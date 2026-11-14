#!/usr/bin/env python3
"""Serve viewer/ from this Mac and forward /api/* to the training server.

Replay files are copied from the training server into recordings/ in the background
(paused while a run is training, so the live stream keeps the link), and played back from
this Mac. Replays therefore need no connection to the training server.

    scripts/tunnel.sh              # Mac:8765 -> GPU machine
    python3 scripts/dev_proxy.py   # http://localhost:8090

The headset reaches it at http://localhost:8090 after `adb reverse tcp:8090 tcp:8090`;
localhost is a secure context, so WebXR works without certificates.
"""
import argparse
import http.client
import json
import os
import mimetypes
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = (Path(__file__).resolve().parent.parent / "viewer").resolve()
RECORDINGS = (Path(__file__).resolve().parent.parent / "recordings").resolve()


class Fetcher(threading.Thread):
    """Copies finished replay files from the training server, newest first."""

    def __init__(self, host, port):
        super().__init__(daemon=True)
        self.host, self.port = host, port
        self.remote = {}  # name -> total bytes, as last listed by the server
        self.progress = {}  # name -> bytes copied so far, for files not yet complete
        RECORDINGS.mkdir(exist_ok=True)

    def _get_json(self, path):
        connection = http.client.HTTPConnection(self.host, self.port, timeout=15)
        try:
            connection.request("GET", path)
            return json.loads(connection.getresponse().read())
        finally:
            connection.close()

    def _training(self):
        try:
            return self._get_json("/api/status").get("state") in ("loading", "training")
        except (OSError, ValueError):
            return True  # unreachable: do not start a download

    def run(self):
        while True:
            try:
                listing = self._get_json("/api/recordings")
                self.remote = {item["name"]: item["bytes"] for item in listing}
                for item in listing:
                    if not (RECORDINGS / f"{item['name']}.lspl").exists():
                        self._copy(item["name"], item["bytes"])
                        break
            except (OSError, ValueError):
                pass
            time.sleep(3)

    def _copy(self, name, total):
        if self._training():
            return
        part = RECORDINGS / f"{name}.lspl.part"
        have = part.stat().st_size if part.exists() else 0
        self.progress[name] = have
        connection = http.client.HTTPConnection(self.host, self.port, timeout=30)
        try:
            connection.request("GET", f"/api/recording/{name}", headers={"Range": f"bytes={have}-"})
            response = connection.getresponse()
            if response.status not in (200, 206):
                return
            checked = time.time()
            with open(part, "ab" if response.status == 206 else "wb") as file:
                if response.status == 200:
                    have = 0
                while chunk := response.read1(1 << 16):
                    file.write(chunk)
                    have += len(chunk)
                    self.progress[name] = have
                    if time.time() - checked > 3:
                        checked = time.time()
                        if self._training():
                            return  # resume after the run, from where this stopped
            if have >= total:
                part.rename(RECORDINGS / f"{name}.lspl")
                self.progress.pop(name, None)
        finally:
            connection.close()

    def listing(self):
        """Local replay files (ready to play) and remote ones still to be copied."""
        local = sorted(RECORDINGS.glob("*.lspl"), key=lambda f: f.stat().st_mtime, reverse=True)
        items = [{"name": f.stem, "bytes": f.stat().st_size, "ready": True} for f in local]
        ready = {item["name"] for item in items}
        waiting = [{"name": name, "bytes": total, "ready": False,
                    "have": self.progress.get(name, 0)}
                   for name, total in self.remote.items() if name not in ready]
        return waiting + items


def make_handler(upstream_host, upstream_port, fetcher):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def _reply(self, code, body=b"", content_type="text/plain", extra=None):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for key, value in (extra or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def _proxy(self, method):
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else None
            try:
                upstream = http.client.HTTPConnection(upstream_host, upstream_port, timeout=60)
                upstream.request(method, self.path, body=body,
                                 headers={"Content-Type": "application/json"})
                response = upstream.getresponse()
                data = response.read()
            except OSError as error:
                return self._reply(502, f"training server unreachable: {error}".encode())
            self._reply(response.status, data,
                        response.getheader("Content-Type", "application/octet-stream"))

        def _proxy_stream(self):
            """Pass the frame stream through as it arrives, without buffering it."""
            try:
                upstream = http.client.HTTPConnection(upstream_host, upstream_port, timeout=30)
                upstream.request("GET", self.path)
                response = upstream.getresponse()
            except OSError as error:
                return self._reply(502, f"training server unreachable: {error}".encode())
            self.send_response(response.status)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            try:
                while chunk := response.read1(1 << 16):
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except OSError:
                pass
            finally:
                upstream.close()

        def _recording(self, name):
            path = (RECORDINGS / name).resolve()
            if path.parent != RECORDINGS or path.suffix != ".lspl" or not path.is_file():
                return self._reply(404, b"not found")
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(path.stat().st_size))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                with open(path, "rb") as file:
                    while chunk := file.read(1 << 16):
                        self.wfile.write(chunk)
            except OSError:
                self.close_connection = True

        def do_GET(self):
            if self.path.startswith("/api/recordings"):
                return self._reply(200, json.dumps(fetcher.listing()).encode(),
                                   "application/json")
            if self.path.startswith("/recordings/"):
                return self._recording(self.path.split("?", 1)[0].rsplit("/", 1)[1])
            if self.path.startswith("/api/stream"):
                return self._proxy_stream()
            if self.path.startswith("/api/"):
                return self._proxy("GET")
            relative = self.path.split("?", 1)[0].lstrip("/") or "index.html"
            target = (ROOT / relative).resolve()
            if target.is_dir():
                target = target / "index.html"
            if ROOT not in target.parents or not target.is_file():
                return self._reply(404, b"not found")
            content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            if target.suffix == ".js":
                content_type = "text/javascript"
            self._reply(200, target.read_bytes(), content_type)

        def do_POST(self):
            if self.path.startswith("/api/"):
                return self._proxy("POST")
            self._reply(404, b"not found")

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8090)))
    parser.add_argument("--upstream", default="127.0.0.1:8765")
    args = parser.parse_args()
    upstream_host, upstream_port = args.upstream.rsplit(":", 1)
    fetcher = Fetcher(upstream_host, int(upstream_port))
    fetcher.start()
    server = ThreadingHTTPServer((args.host, args.port),
                                 make_handler(upstream_host, int(upstream_port), fetcher))
    server.daemon_threads = True
    print(f"Viewer on http://{args.host}:{args.port}  (api -> {args.upstream})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
