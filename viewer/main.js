// Live Splat viewer: follows a 3DGS training run, redrawing the Gaussians as frames arrive.
// Works on a desktop browser (orbit controls) and in WebXR on a headset (passthrough AR).
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { PackedSplats, SparkRenderer, SparkXr, SplatMesh } from "@sparkjsdev/spark";

const HOLOGRAM_RADIUS = 0.45; // metres: the scene is shown as a tabletop hologram
const XR_ANCHOR = new THREE.Vector3(0, 1.05, -0.9); // in front of the viewer, table height

const $ = (id) => document.getElementById(id);
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// ---------------------------------------------------------------- three.js scene

const renderer = new THREE.WebGLRenderer({ antialias: false, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setClearColor(0x000000, 0);
document.body.prepend(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.02, 100);
camera.position.set(0, 0.35, 1.5);

// The camera and Spark share a frame so splat sorting stays correct in XR.
const localFrame = new THREE.Group();
scene.add(localFrame);
const spark = new SparkRenderer({ renderer, maxStdDev: Math.sqrt(5) });
localFrame.add(spark);
localFrame.add(camera);

// stage: where the hologram sits (grabbed and scaled by the user).
// fit: maps the training world (z-up, arbitrary units) into the hologram.
// spin: turns the scene so its photographed side faces the viewer (the board stays put).
const stage = new THREE.Group();
const spin = new THREE.Group();
const fit = new THREE.Group();
const content = new THREE.Group();
scene.add(stage);
stage.add(spin);
spin.add(fit);
fit.add(content);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.target.set(0, 0, 0);

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// Direction, in stage space, from the scene towards where it was photographed from.
const front = new THREE.Vector3(0, 0.3, 1).normalize();
let fittedScene = "";

function fitScene(info) {
  if (!info || !info.radius) return;
  const up = new THREE.Vector3(...(info.up || [0, 0, 1])).normalize();
  fit.quaternion.setFromUnitVectors(up, new THREE.Vector3(0, 1, 0));
  fit.scale.setScalar(HOLOGRAM_RADIUS / info.radius);
  content.position.set(...info.center).multiplyScalar(-1);
  if (info.view) {
    front.set(...info.view).sub(new THREE.Vector3(...info.center))
      .applyQuaternion(fit.quaternion).normalize();
  }
  if (info.name !== fittedScene) {
    fittedScene = info.name;
    faceViewer();
  }
}

// Look at the scene from the side it was captured from.
function faceViewer() {
  if (renderer.xr.isPresenting) {
    // Turn the hologram so that side faces the viewer, who stands towards +z.
    spin.rotation.set(0, -Math.atan2(front.x, front.z), 0);
  } else {
    spin.rotation.set(0, 0, 0);
    camera.position.copy(front).multiplyScalar(1.5);
    controls.target.set(0, 0, 0);
  }
}

// ---------------------------------------------------------------- frames

// One mesh for the whole session. Each frame overwrites its packed array in place:
// the server already sends Spark's 16-byte layout, so there is nothing to parse.
const packed = new PackedSplats({ packedArray: new Uint32Array(2048 * 64 * 4), numSplats: 0 });
const mesh = new SplatMesh({ packedSplats: packed });
content.add(mesh);

// mode: "live" follows the training server; "replay" plays a recorded run from this Mac.
const state = { mode: "live", run: 0, meta: null, status: null, frames: 0, fps: 0,
  replay: null, recordings: [] };
let fpsMark = { time: performance.now(), frames: 0 };

let shownCount = 0;

async function inflate(bytes) {
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("deflate"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

// A frame is the change since the previous one: 16 byte-columns of `count` values each,
// added (mod 256) onto the rows already held. Rows past the previous count start at zero.
async function showFrame(payload, meta) {
  const count = meta.gaussians;
  const planes = await inflate(payload);
  if (planes.byteLength !== count * 16) throw new Error("frame size mismatch");
  const array = packed.ensureSplats(count);
  const rows = new Uint8Array(array.buffer, array.byteOffset, array.byteLength);
  if (meta.reset) shownCount = 0;
  if (count > shownCount) rows.fill(0, shownCount * 16, count * 16);
  for (let column = 0; column < 16; column++) {
    const plane = planes.subarray(column * count, (column + 1) * count);
    for (let i = 0, at = column; i < count; i++, at += 16) rows[at] += plane[i];
  }
  shownCount = count;
  packed.numSplats = count;
  packed.needsUpdate = true;
  mesh.updateVersion();
  state.meta = meta;
  state.run = meta.run;
  state.frames += 1;
  const now = performance.now();
  if (now - fpsMark.time > 1000) {
    state.fps = (state.frames - fpsMark.frames) * 1000 / (now - fpsMark.time);
    fpsMark = { time: now, frames: state.frames };
  }
  render();
}

// Reads exact byte counts from a fetch body. Returns null at a clean end of stream.
function byteReader(stream) {
  const reader = stream.getReader();
  let chunks = [], available = 0;
  return async function read(length) {
    while (available < length) {
      const { done, value } = await reader.read();
      if (done) {
        if (available === 0) return null;
        throw new Error("stream cut short");
      }
      chunks.push(value);
      available += value.byteLength;
    }
    const out = new Uint8Array(length);
    let offset = 0;
    while (offset < length) {
      const head = chunks[0];
      const take = Math.min(head.byteLength, length - offset);
      out.set(head.subarray(0, take), offset);
      offset += take;
      if (take === head.byteLength) chunks.shift();
      else chunks[0] = head.subarray(take);
    }
    available -= length;
    return out;
  };
}

// One message: "LSPL", a u32 header length, a JSON header, then `bytes` of payload.
const decoder = new TextDecoder();
async function readMessage(read) {
  const prefix = await read(8);
  if (!prefix) return null;
  if (decoder.decode(prefix.subarray(0, 4)) !== "LSPL") throw new Error("bad frame");
  const headerLength = new DataView(prefix.buffer).getUint32(4, true);
  const meta = JSON.parse(decoder.decode(await read(headerLength)));
  return { meta, payload: meta.bytes ? await read(meta.bytes) : null };
}

// ---------------------------------------------------------------- live

const viewerId = Math.random().toString(36).slice(2, 10);
let liveAbort = null;

async function followFrames() {
  for (;;) {
    if (state.mode !== "live") {
      await sleep(200);
      continue;
    }
    try {
      liveAbort = new AbortController();
      const response = await fetch(`/api/stream?id=${viewerId}`, { signal: liveAbort.signal });
      if (!response.ok) throw new Error(`stream ${response.status}`);
      const read = byteReader(response.body);
      for (;;) {
        const message = await readMessage(read);
        if (!message) throw new Error("stream closed");
        if (!message.payload || state.mode !== "live") continue;
        const meta = message.meta;
        await showFrame(message.payload, meta);
        // Tell the server this frame is on screen; it keeps only a few frames in flight.
        fetch(`/api/ack?id=${viewerId}&run=${meta.run}&seq=${meta.seq}`, { method: "POST" })
          .catch(() => {});
      }
    } catch (error) {
      if (state.mode !== "live") continue;
      console.warn("frame stream interrupted", error);
      await sleep(1000);
    }
  }
}

async function followStatus() {
  for (;;) {
    try {
      const status = await (await fetch("/api/status")).json();
      if (state.mode === "live" && status.scene && status.scene.radius) fitScene(status.scene);
      const select = $("scene");
      if (!select.options.length && status.scenes) {
        for (const name of status.scenes) select.add(new Option(name, name));
        if (status.config) select.value = status.config.scene;
      }
      state.status = status;
      render();
    } catch (error) {
      state.status = { state: "offline" };
      render();
    }
    await sleep(500);
  }
}

async function startTraining() {
  stopReplay();
  const body = { scene: $("scene").value, max_steps: Number($("steps").value) || 7000,
    pace: $("pace").checked };
  await fetch("/api/start", { method: "POST", body: JSON.stringify(body) });
}

$("start").addEventListener("click", startTraining);
$("stop").addEventListener("click", () => fetch("/api/stop", { method: "POST" }));

// ---------------------------------------------------------------- replay

// A recorded run holds every 20th step. It is read from this Mac, so playback runs at a
// steady rate whatever the link to the training server is doing.
const REPLAY_FPS = 15;
let replayAbort = null;
let replayTicket = 0;
let autoPlayed = "";

async function playReplay(name) {
  const ticket = ++replayTicket;
  state.mode = "replay";
  state.replay = { name, scene: name, max: 1 };
  liveAbort?.abort();
  render();
  while (ticket === replayTicket) {
    try {
      replayAbort = new AbortController();
      const response = await fetch(`/recordings/${name}.lspl`, { signal: replayAbort.signal });
      if (!response.ok) throw new Error(`recording ${response.status}`);
      const read = byteReader(response.body);
      let due = performance.now();
      for (;;) {
        const message = await readMessage(read);
        if (!message || ticket !== replayTicket) break;
        const meta = message.meta;
        if (meta.type === "intro") {
          state.replay.scene = meta.scene.name;
          state.replay.max = meta.config.max_steps;
          fitScene(meta.scene);
        }
        if (!message.payload) continue;
        await showFrame(message.payload, meta);
        due = Math.max(due + 1000 / REPLAY_FPS, performance.now());
        await sleep(due - performance.now());
      }
    } catch (error) {
      if (ticket !== replayTicket) return;
      console.warn("replay interrupted", error);
    }
    if (ticket === replayTicket) await sleep(2500); // hold the finished scene, then loop
  }
}

function stopReplay() {
  if (state.mode !== "replay") return;
  replayTicket += 1;
  replayAbort?.abort();
  state.replay = null;
  state.mode = "live"; // followFrames reconnects and the server resends the full scene
  render();
}

async function followRecordings() {
  for (;;) {
    try {
      state.recordings = await (await fetch("/api/recordings")).json();
      const select = $("replays");
      const names = state.recordings.filter((item) => item.ready).map((item) => item.name);
      if (names.join() !== [...select.options].map((option) => option.value).join()) {
        const chosen = select.value;
        select.replaceChildren(...names.map((name) => new Option(name, name)));
        if (names.includes(chosen)) select.value = chosen;
      }
      // When a run has finished and its recording has arrived, play it back.
      const finished = state.status?.state === "done" ? state.status.recording : "";
      if (finished && finished !== autoPlayed && $("auto").checked && state.mode === "live"
          && names.includes(finished)) {
        autoPlayed = finished;
        select.value = finished;
        playReplay(finished);
      }
      render();
    } catch (error) {
      // proxy restarting: try again
    }
    await sleep(1500);
  }
}

$("play").addEventListener("click", () => {
  if (state.mode === "replay") stopReplay();
  else if ($("replays").value) playReplay($("replays").value);
});

// ---------------------------------------------------------------- readouts

const LABELS = { idle: "Idle. Pick a scene and start.", loading: "Loading images…",
  training: "Training", done: "Done", stopped: "Stopped", error: "Error", offline: "Server offline" };

function readout() {
  const status = state.status || {};
  const meta = state.meta || {};
  // A recording still being copied from the training server, if any.
  const pending = state.recordings.find((item) => !item.ready);
  const copying = pending ? `${Math.round(100 * (pending.have || 0) / pending.bytes)}%` : "";
  if (state.mode === "replay") {
    const max = state.replay.max || 1;
    return {
      replay: true, copying,
      label: "Replay", scene: state.replay.scene,
      step: meta.step ?? 0, max, progress: Math.min(1, (meta.step ?? 0) / max),
      shownStep: meta.step, count: meta.gaussians, psnr: meta.psnr, fps: state.fps,
    };
  }
  const max = status.max_steps || 1;
  return {
    replay: false, copying,
    label: LABELS[status.state] || status.state || "",
    step: status.step ?? 0,
    max,
    progress: Math.min(1, (status.step ?? 0) / max),
    shownStep: meta.step,
    count: meta.gaussians,
    psnr: meta.psnr,
    speed: status.steps_per_sec,
    fps: state.fps,
    elapsed: status.elapsed,
    scene: status.scene?.name,
    error: status.error,
  };
}

function render() {
  const r = readout();
  $("state").textContent = r.scene ? `${r.label} · ${r.scene}` : r.label;
  $("fill").style.width = `${r.progress * 100}%`;
  $("step").textContent = r.replay || state.status?.max_steps ? `${r.step} / ${r.max}` : "–";
  $("play").textContent = r.replay ? "Back to live" : "Replay";
  $("copying").textContent = r.copying ? `Copying the latest recording: ${r.copying}` : "";
  $("count").textContent = r.count ? r.count.toLocaleString() : "–";
  $("psnr").textContent = r.psnr ? `${r.psnr.toFixed(1)} dB` : "–";
  $("speed").textContent = r.speed ? `${r.speed.toFixed(0)} steps/s` : "–";
  $("shown").textContent = r.shownStep != null ? `step ${r.shownStep}` : "–";
  $("fps").textContent = r.fps ? `${r.fps.toFixed(1)} / s` : "–";
  $("elapsed").textContent = r.elapsed ? `${r.elapsed.toFixed(0)} s` : "–";
  drawBoard(r);
}

// A floating board next to the hologram, readable inside the headset.
const boardCanvas = document.createElement("canvas");
boardCanvas.width = 1024;
boardCanvas.height = 384;
const boardTexture = new THREE.CanvasTexture(boardCanvas);
boardTexture.colorSpace = THREE.SRGBColorSpace;
const board = new THREE.Mesh(
  new THREE.PlaneGeometry(0.64, 0.24),
  new THREE.MeshBasicMaterial({ map: boardTexture, transparent: true }),
);
board.position.set(0, -0.42, 0.25);
board.rotation.x = -0.5;
board.visible = false;
stage.add(board);

function drawBoard(r) {
  const g = boardCanvas.getContext("2d");
  const { width: w, height: h } = boardCanvas;
  g.clearRect(0, 0, w, h);
  g.fillStyle = "rgba(10, 12, 17, 0.86)";
  g.beginPath();
  g.roundRect(0, 0, w, h, 36);
  g.fill();
  g.fillStyle = "#8d95a5";
  g.font = "34px system-ui, sans-serif";
  g.fillText(r.scene ? `${r.label} · ${r.scene}` : r.label, 44, 66);
  g.fillStyle = "#eef1f6";
  g.font = "600 84px system-ui, sans-serif";
  g.fillText(`step ${r.step.toLocaleString()}`, 44, 164);
  g.font = "40px system-ui, sans-serif";
  g.fillStyle = "#c8cedb";
  const count = r.count ? `${r.count.toLocaleString()} Gaussians` : "";
  const psnr = r.psnr ? `PSNR ${r.psnr.toFixed(1)} dB` : "";
  g.fillText([count, psnr].filter(Boolean).join("   ·   "), 44, 232);
  g.fillStyle = "#1b1f29";
  g.beginPath();
  g.roundRect(44, 276, w - 88, 22, 11);
  g.fill();
  g.fillStyle = "#7cf3c6";
  g.beginPath();
  g.roundRect(44, 276, Math.max(22, (w - 88) * r.progress), 22, 11);
  g.fill();
  g.fillStyle = "#8d95a5";
  g.font = "28px system-ui, sans-serif";
  const hint = r.copying ? `replay arriving: ${r.copying}`
    : "trigger: restart   ·   grip: move   ·   both grips: resize";
  g.fillText(hint, 44, 348);
  boardTexture.needsUpdate = true;
}

// ---------------------------------------------------------------- WebXR

const xr = new SparkXr({
  renderer,
  mode: "arvr",
  element: $("xr"),
  referenceSpaceType: "local-floor",
  frameBufferScaleFactor: 0.7,
  onEnterXr: () => {
    $("xr").textContent = "Exit AR";
    stage.position.copy(XR_ANCHOR);
    stage.scale.setScalar(1);
    board.visible = true;
    controls.enabled = false;
    setTimeout(faceViewer, 300); // once the session is presenting
  },
  onExitXr: () => {
    $("xr").textContent = "Enter AR";
    stage.position.set(0, 0, 0);
    stage.quaternion.identity();
    stage.scale.setScalar(1);
    board.visible = false;
    controls.enabled = true;
    faceViewer();
  },
});
window.liveSplat = { xr, spark, packed, mesh, content, state, THREE, PackedSplats, SplatMesh };

// Grip moves the hologram; both grips resize it; trigger restarts training.
const hands = [0, 1].map((index) => {
  const grip = renderer.xr.getControllerGrip(index);
  localFrame.add(grip);
  const hand = { grip, squeezing: false };
  grip.addEventListener("squeezestart", () => { hand.squeezing = true; regrab(); });
  grip.addEventListener("squeezeend", () => { hand.squeezing = false; regrab(); });
  renderer.xr.getController(index).addEventListener("select", startTraining);
  return hand;
});
let pinch = null;

function regrab() {
  const held = hands.filter((hand) => hand.squeezing);
  scene.attach(stage);
  pinch = null;
  if (held.length === 1) {
    held[0].grip.attach(stage);
  } else if (held.length === 2) {
    pinch = { distance: handDistance(), scale: stage.scale.x };
  }
}

function handDistance() {
  const a = new THREE.Vector3(), b = new THREE.Vector3();
  hands[0].grip.getWorldPosition(a);
  hands[1].grip.getWorldPosition(b);
  return Math.max(a.distanceTo(b), 0.01);
}

renderer.setAnimationLoop(() => {
  if (pinch) {
    const scale = THREE.MathUtils.clamp(pinch.scale * handDistance() / pinch.distance, 0.2, 8);
    stage.scale.setScalar(scale);
  }
  if (!renderer.xr.isPresenting) controls.update();
  renderer.render(scene, camera);
});

followStatus();
followFrames();
followRecordings();
