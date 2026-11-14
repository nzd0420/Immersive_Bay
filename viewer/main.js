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

// Two ways to place a scene:
//   hologram   scaled to fit a table top, turned to face the viewer
//   room       real size and where it was scanned, so the viewer stands inside it
// Room scale needs a scan in metres with a known floor, which a Quest scan provides.
let sceneInfo = null; // from the server's status, or a recording's intro message
let roomWanted = true;
let solidBackground = false;

// In AR the real room shows wherever nothing opaque is drawn, and three.js always clears
// to transparent there, whatever clear colour is set. So hiding the room takes a real
// object: a black sphere around the viewer, drawn before everything else.
const backdrop = new THREE.Mesh(
  new THREE.SphereGeometry(40, 24, 12),
  new THREE.MeshBasicMaterial({ color: 0x000000, side: THREE.BackSide, depthTest: false, depthWrite: false }),
);
backdrop.renderOrder = -1000;
backdrop.frustumCulled = false;
backdrop.visible = false;
camera.add(backdrop);

function setBackdrop(solid) {
  solidBackground = solid;
  backdrop.visible = renderer.xr.isPresenting && solidBackground;
}

// The capture world has y down and z forward; three.js has y up and z towards the viewer.
const CAPTURE_TO_THREE = new THREE.Matrix4().makeScale(1, -1, -1);
// Direction, in stage space, from the scene towards where it was photographed from.
const front = new THREE.Vector3(0, 0.3, 1).normalize();

const inRoom = () => roomWanted && !!sceneInfo?.to_world;
// A scene scanned in this AR session is already in its coordinates: nothing to line up.
const anchored = () => inRoom() && !!sceneInfo.anchored;

function fitScene(info) {
  if (!info || !info.radius) return;
  const changed = info.name !== sceneInfo?.name || !!info.to_world !== !!sceneInfo?.to_world;
  sceneInfo = info;
  layout(changed);
}

function layout(replace = false) {
  const info = sceneInfo;
  if (!info) return;
  if (inRoom()) {
    const toWorld = new THREE.Matrix4().set(...info.to_world.flat());
    toWorld.premultiply(CAPTURE_TO_THREE).decompose(fit.position, fit.quaternion, fit.scale);
    if (!info.anchored) fit.position.y += info.floor; // put the scanned floor at y = 0
    content.position.set(0, 0, 0);
    spin.rotation.set(0, 0, 0);
  } else {
    const up = new THREE.Vector3(...(info.up || [0, 0, 1])).normalize();
    fit.position.set(0, 0, 0);
    fit.quaternion.setFromUnitVectors(up, new THREE.Vector3(0, 1, 0));
    fit.scale.setScalar(HOLOGRAM_RADIUS / info.radius);
    content.position.set(...info.center).multiplyScalar(-1);
    if (info.view) {
      front.set(...info.view).sub(new THREE.Vector3(...info.center))
        .applyQuaternion(fit.quaternion).normalize();
    }
  }
  if (replace) placeStage();
}

// Where the whole scene sits, and where a desktop camera looks from.
function placeStage() {
  const presenting = renderer.xr.isPresenting;
  scene.attach(stage);
  stage.scale.setScalar(1);
  stage.quaternion.identity();
  stage.position.set(0, 0, 0);
  if (inRoom()) {
    const saved = presenting && !sceneInfo.anchored ? loadAlignment() : null;
    if (saved) {
      stage.position.set(saved.x, saved.y, saved.z);
      stage.rotation.set(0, saved.yaw, 0);
    }
    if (!presenting) {
      // Stand where the scan was taken from and look at the middle of the scene.
      scene.updateMatrixWorld(true);
      camera.position.copy(content.localToWorld(new THREE.Vector3(...sceneInfo.view)));
      controls.target.copy(content.localToWorld(new THREE.Vector3(...sceneInfo.center)));
    }
  } else if (presenting) {
    stage.position.copy(XR_ANCHOR);
    // Turn the hologram so its photographed side faces the viewer, who stands towards +z.
    spin.rotation.set(0, -Math.atan2(front.x, front.z), 0);
  } else {
    spin.rotation.set(0, 0, 0);
    camera.position.copy(front).multiplyScalar(1.5);
    controls.target.set(0, 0, 0);
  }
  placeBoard();
  setBackdrop(solidBackground);
}

// The viewer lines the room up with reality by hand; remember it for this scene.
function loadAlignment() {
  try {
    return JSON.parse(localStorage.getItem(`align:${sceneInfo.name}`));
  } catch (error) {
    return null;
  }
}

function saveAlignment() {
  try {
    const { x, y, z } = stage.position;
    localStorage.setItem(`align:${sceneInfo.name}`, JSON.stringify({ x, y, z, yaw: stage.rotation.y }));
  } catch (error) {
    // storage unavailable: the alignment just lasts for this session
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
  // The first frame of a room-scale reconstruction: black out the real room, so what
  // is seen is only what was rebuilt. (B/Y brings the room back.)
  if (!mesh.visible && inRoom()) setBackdrop(true);
  mesh.visible = true;
  markers.visible = false;
  depthCloud.visible = false;
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
let mutedRun = -1; // a run whose remaining frames are to be ignored
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
        const meta = message.meta;
        // After a reset, frames of the run that was stopped may still be on their way.
        if (!message.payload || state.mode !== "live" || meta.run === mutedRun) continue;
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
    pace: $("pace").checked, record: $("auto").checked };
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

// ---------------------------------------------------------------- scan

// Scanning happens inside the AR session: the page photographs through the headset's
// passthrough cameras and notes where the head was for each photo. The Mac turns that
// into a training scene. Because the same session then shows the result, the Gaussians
// come back exactly where the real things are.
const SCAN = { everyMs: 450, size: 1024, maxPhotos: 240, maxTurn: 50 };
const scan = { cameras: [], active: false, pending: "", id: "", photos: 0, sending: 0,
  poses: [], last: null, turning: 0, note: "", timebase: "raf",
  numbers: { left: 0, right: 0 }, poseAt: 0, timing: null };

// Small dots where photos were taken, so coverage is visible while scanning.
const markers = new THREE.Group();
scene.add(markers);
const markerGeometry = new THREE.ConeGeometry(0.022, 0.04, 4).rotateX(-Math.PI / 2).rotateZ(Math.PI / 4);
const markerMaterial = new THREE.MeshBasicMaterial({ color: 0x7cf3c6, wireframe: true });

// What the headset's depth sensor sees while scanning, as points on the real surfaces.
// They show coverage as it builds up, and become the starting points for training.
const DEPTH = { grid: 56, everyMs: 350, maxPoints: 400000, near: 0.25, far: 4.5 };
const depthPositions = new Float32Array(DEPTH.maxPoints * 3);
const depthColours = new Float32Array(DEPTH.maxPoints * 3);
const depthGeometry = new THREE.BufferGeometry();
depthGeometry.setAttribute("position", new THREE.BufferAttribute(depthPositions, 3));
depthGeometry.setAttribute("color", new THREE.BufferAttribute(depthColours, 3));
depthGeometry.setDrawRange(0, 0);
const depthCloud = new THREE.Points(depthGeometry, new THREE.PointsMaterial(
  { size: 0.014, sizeAttenuation: true, vertexColors: true, transparent: true, opacity: 0.9 }));
depthCloud.frustumCulled = false;
depthCloud.visible = false; // shown while scanning
scene.add(depthCloud);
const depth = { count: 0, last: 0, reader: null, info: null, stats: null, error: "" };

// Copies a coarse grid of the depth map into a small RGBA target that can be read back.
function depthReader(gl) {
  const compile = (type, source) => {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
    return shader;
  };
  const program = gl.createProgram();
  gl.attachShader(program, compile(gl.VERTEX_SHADER, `#version 300 es
    out vec2 uv;
    void main() {
      vec2 corner = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
      uv = corner;
      gl_Position = vec4(corner * 2.0 - 1.0, 0.0, 1.0);
    }`));
  gl.attachShader(program, compile(gl.FRAGMENT_SHADER, `#version 300 es
    precision highp float;
    precision highp sampler2DArray;
    uniform sampler2DArray depthMap;
    in vec2 uv;
    out vec4 colour;
    void main() {
      float d = clamp(texture(depthMap, vec3(uv, 0.0)).r, 0.0, 1.0) * 65535.0;
      colour = vec4(floor(d / 256.0) / 255.0, mod(floor(d), 256.0) / 255.0, 0.0, 1.0);
    }`));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
  const target = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, target);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, DEPTH.grid, DEPTH.grid, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  const framebuffer = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, target, 0);
  return { program, framebuffer, vertexArray: gl.createVertexArray(),
    uniform: gl.getUniformLocation(program, "depthMap"),
    pixels: new Uint8Array(DEPTH.grid * DEPTH.grid * 4) };
}

// Reads the left eye's depth map and returns its points in room coordinates.
function readDepth(frame, pose) {
  const session = renderer.xr.getSession();
  if (!session.enabledFeatures?.includes("depth-sensing")) return null;
  const view = pose.views[0];
  const info = renderer.xr.getBinding()?.getDepthInformation(view);
  if (!info || info.isValid === false || !info.texture) return null;
  const gl = renderer.getContext();
  depth.reader ||= depthReader(gl);
  const reader = depth.reader;
  gl.bindFramebuffer(gl.FRAMEBUFFER, reader.framebuffer);
  gl.viewport(0, 0, DEPTH.grid, DEPTH.grid);
  gl.disable(gl.DEPTH_TEST);
  gl.disable(gl.BLEND);
  gl.disable(gl.SCISSOR_TEST);
  gl.useProgram(reader.program);
  gl.bindVertexArray(reader.vertexArray);
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D_ARRAY, info.texture);
  gl.uniform1i(reader.uniform, 0);
  gl.drawArrays(gl.TRIANGLES, 0, 3);
  gl.readPixels(0, 0, DEPTH.grid, DEPTH.grid, gl.RGBA, gl.UNSIGNED_BYTE, reader.pixels);
  gl.bindTexture(gl.TEXTURE_2D_ARRAY, null);
  renderer.resetState(); // three.js caches GL state: tell it everything changed

  const near = info.depthNear ?? session.renderState.depthNear;
  const far = info.depthFar ?? session.renderState.depthFar;
  const projection = view.projectionMatrix;
  const toRoom = new THREE.Matrix4().fromArray(view.transform.matrix);
  const point = new THREE.Vector3();
  const points = [];
  let valid = 0, nearest = Infinity, farthest = 0;
  for (let row = 0; row < DEPTH.grid; row++) {
    for (let column = 0; column < DEPTH.grid; column++) {
      const at = (row * DEPTH.grid + column) * 4;
      const d = (reader.pixels[at] * 256 + reader.pixels[at + 1]) / 65535;
      if (d <= 0.0005 || d >= 0.9995) continue;
      const ndcZ = 2 * d - 1;
      // distance in front of the eye, from a depth-buffer value
      const distance = Number.isFinite(far)
        ? 2 * near * far / (far + near - ndcZ * (far - near)) : near / (1 - d);
      if (!(distance > DEPTH.near && distance < DEPTH.far)) continue;
      valid += 1;
      nearest = Math.min(nearest, distance);
      farthest = Math.max(farthest, distance);
      const ndcX = (column + 0.5) / DEPTH.grid * 2 - 1;
      const ndcY = (row + 0.5) / DEPTH.grid * 2 - 1;
      point.set(distance * (ndcX + projection[8]) / projection[0],
                distance * (ndcY + projection[9]) / projection[5], -distance).applyMatrix4(toRoom);
      points.push(point.x, point.y, point.z);
    }
  }
  depth.info = { near, far, size: [info.width, info.height], type: info.textureType, valid };
  depth.stats = { valid, nearest: +nearest.toFixed(2), farthest: +farthest.toFixed(2) };
  return new Float32Array(points);
}

// Adds freshly sensed points to the cloud and sends them to the Mac with their time.
function senseDepth(time, frame, pose) {
  if (time - depth.last < DEPTH.everyMs || scan.turning > SCAN.maxTurn) return;
  depth.last = time;
  let points = null;
  try {
    points = readDepth(frame, pose);
  } catch (error) {
    depth.error = String(error);
    return;
  }
  if (!points || !points.length) return;
  const room = Math.min(points.length / 3, DEPTH.maxPoints - depth.count);
  const colour = new THREE.Color();
  for (let i = 0; i < room; i++) {
    const at = (depth.count + i) * 3;
    depthPositions.set(points.subarray(i * 3, i * 3 + 3), at);
    // colour by height, so floors, tables and walls read as different bands
    colour.setHSL(0.55 - 0.45 * THREE.MathUtils.clamp(points[i * 3 + 1] / 2.2, 0, 1), 0.85, 0.6);
    depthColours.set([colour.r, colour.g, colour.b], at);
  }
  depth.count += room;
  depthGeometry.setDrawRange(0, depth.count);
  depthGeometry.attributes.position.needsUpdate = true;
  depthGeometry.attributes.color.needsUpdate = true;
  scan.sending += 1;
  fetch(`/scan/depth?id=${scan.id}&t=${scan.last.t}`, { method: "POST", body: points })
    .catch(() => {}).finally(() => { scan.sending -= 1; });
}

// Several things can ask for the cameras at once (the button, the tab regaining focus
// after the permission prompt). They must all share one attempt: opening twice gave two
// streams per camera, whose photos overwrote each other under the same number.
let camerasOpening = null;
let camerasOpeningSince = 0;

// One attempt at a time: a second one started while the first still holds a camera
// fails with "Could not start video source", and then cancels the first.
// An attempt older than 30 s is taken to be stuck and may be replaced.
function openCameras() {
  if (scan.cameras.length) return Promise.resolve();
  if (!camerasOpening || performance.now() - camerasOpeningSince > 30000) {
    camerasOpeningSince = performance.now();
    const promise = openCamerasOnce().finally(() => {
      if (camerasOpening === promise) camerasOpening = null;
    });
    camerasOpening = promise;
  }
  return camerasOpening;
}

function closeCameras() {
  for (const camera of scan.cameras) camera.video.srcObject?.getTracks().forEach((track) => track.stop());
  scan.cameras.length = 0;
}
// A reload must hand the cameras back at once, or the new page cannot open them.
window.addEventListener("pagehide", closeCameras);

// Tells the Mac what happened on the headset, so problems can be read without a cable.
function report(what, detail) {
  fetch("/debug/log", { method: "POST", body: JSON.stringify({ [what]: detail, at: Date.now() }) })
    .catch(() => {});
}
window.addEventListener("error", (event) => report("error", String(event.message)));
window.addEventListener("unhandledrejection", (event) => report("rejection", String(event.reason)));

const within = (promise, ms, what) => Promise.race([promise,
  new Promise((_, reject) => setTimeout(() => reject(new Error(`${what} did not answer in ${ms / 1000} s`)), ms))]);

// Opens one camera, trying again if it is still being released by whoever had it last.
async function openCamera(deviceId, side) {
  let failure = null;
  for (let attempt = 0; attempt < 4; attempt++) {
    if (attempt) await sleep(900);
    let stream = null;
    try {
      stream = await within(navigator.mediaDevices.getUserMedia({
        video: { deviceId: { exact: deviceId }, width: { ideal: 1728 }, height: { ideal: 1728 } },
      }), 10000, `the ${side} camera`);
      const video = document.createElement("video");
      video.muted = true;
      video.playsInline = true;
      video.srcObject = stream;
      await within(video.play(), 6000, `the ${side} camera picture`);
      return video;
    } catch (error) {
      stream?.getTracks().forEach((track) => track.stop());
      failure = error;
    }
  }
  throw failure;
}

async function openCamerasOnce() {
  let devices = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput");
  if (!devices.some((device) => device.label)) {
    // Labels are hidden until the camera permission is given: ask for it.
    const probe = await within(navigator.mediaDevices.getUserMedia({ video: true }), 30000, "the camera");
    probe.getTracks().forEach((track) => track.stop());
    await sleep(600); // let the camera be released before it is opened again
    devices = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput");
  }
  devices = devices.filter((device) => /facing back/.test(device.label));
  report("cameras_found", devices.map((device) => device.label));
  for (const device of devices) {
    // The Quest browser lists the passthrough cameras as "camera 1" (left), "camera 2" (right).
    const side = /camera 1\b/.test(device.label) ? "left" : /camera 2\b/.test(device.label) ? "right" : "";
    if (!side || scan.cameras.some((other) => other.side === side)) continue;
    const video = await openCamera(device.deviceId, side);
    const camera = { side, video, last: 0 };
    scan.cameras.push(camera);
    const onFrame = (now, metadata) => {
      video.requestVideoFrameCallback(onFrame);
      // No photo without a fresh head pose: the AR frame loop stops while the system
      // menu is up or the headset is off, and the cameras keep running.
      const tracked = performance.now() - scan.poseAt < 150;
      const ready = scan.active && tracked && now - camera.last >= SCAN.everyMs
        && scan.sending < 6 && scan.photos < SCAN.maxPhotos && scan.turning < SCAN.maxTurn;
      if (!ready) return;
      camera.last = now;
      photograph(camera, metadata.captureTime ?? now);
    };
    video.requestVideoFrameCallback(onFrame);
  }
  report("cameras_open", scan.cameras.map((camera) => camera.side));
  render();
}

async function photograph(camera, time) {
  const n = scan.numbers[camera.side]++;
  scan.photos += 1;
  scan.sending += 1;
  const started = performance.now();
  if (scan.last && camera.side === "left") {
    const dot = new THREE.Mesh(markerGeometry, markerMaterial);
    dot.position.fromArray(scan.last.p);
    dot.quaternion.fromArray(scan.last.q);
    markers.add(dot);
  }
  try {
    const bitmap = await createImageBitmap(camera.video,
      { resizeWidth: SCAN.size, resizeHeight: SCAN.size, resizeQuality: "medium" });
    const canvas = new OffscreenCanvas(SCAN.size, SCAN.size);
    canvas.getContext("2d").drawImage(bitmap, 0, 0);
    bitmap.close();
    const blob = await canvas.convertToBlob({ type: "image/jpeg", quality: 0.88 });
    const encoded = performance.now();
    const query = `id=${scan.id}&side=${camera.side}&n=${n}&t=${time}&w=${SCAN.size}&h=${SCAN.size}`;
    await fetch(`/scan/frame?${query}`, { method: "POST", body: blob });
    // how long a photo takes to prepare and to send, for tuning
    scan.timing = { encodeMs: Math.round(encoded - started), sendMs: Math.round(performance.now() - encoded),
      bytes: blob.size };
  } catch (error) {
    console.warn("photo not sent", error);
  } finally {
    scan.sending -= 1;
  }
}

// Called every AR frame: the head pose history, and how fast the head is turning.
function recordPose(time, frame) {
  const pose = frame?.getViewerPose(renderer.xr.getReferenceSpace());
  if (!pose) return;
  const { position: p, orientation: q } = pose.transform;
  if (frame.predictedDisplayTime !== undefined) scan.timebase = "predicted";
  const t = frame.predictedDisplayTime ?? time;
  if (scan.last && t > scan.last.t) {
    const dot = Math.abs(q.x * scan.last.q[0] + q.y * scan.last.q[1] + q.z * scan.last.q[2] + q.w * scan.last.q[3]);
    const speed = 2 * Math.acos(Math.min(1, dot)) * 180 / Math.PI / ((t - scan.last.t) / 1000);
    scan.turning = 0.8 * scan.turning + 0.2 * speed;
  }
  scan.last = { t, p: [p.x, p.y, p.z], q: [q.x, q.y, q.z, q.w] };
  scan.poseAt = performance.now();
  if (!scan.active) return;
  scan.poses.push([t, p.x, p.y, p.z, q.x, q.y, q.z, q.w]);
  senseDepth(time, frame, pose);
}

function startScan() {
  if (!scan.cameras.length) {
    scan.note = "cameras are off: press Enable scanning on the page first";
    render();
    return;
  }
  stopReplay();
  fetch("/api/stop", { method: "POST" }).catch(() => {});
  const now = new Date();
  const two = (value) => String(value).padStart(2, "0");
  scan.id = `scan-${two(now.getHours())}${two(now.getMinutes())}${two(now.getSeconds())}`;
  scan.photos = 0;
  scan.poses = [];
  scan.note = "";
  scan.pending = "";
  for (const camera of scan.cameras) camera.last = 0;
  scan.numbers = { left: 0, right: 0 };
  markers.clear();
  markers.visible = true;
  depth.count = 0;
  depth.error = "";
  depthGeometry.setDrawRange(0, 0);
  depthCloud.visible = true;
  mesh.visible = false;
  setBackdrop(false); // the real room has to be visible to scan it
  roomWanted = true;
  $("room").checked = true;
  scan.active = true;
  placeBoard();
  render();
}

async function finishScan() {
  scan.active = false;
  scan.pending = "sending";
  render();
  while (scan.sending) await sleep(50);
  const options = { steps: Number($("steps").value) || 3000, timebase: scan.timebase,
    record: $("auto").checked };
  try {
    await fetch(`/scan/finish?id=${scan.id}`,
      { method: "POST", body: JSON.stringify({ poses: scan.poses, options }) });
    scan.pending = "building";
  } catch (error) {
    scan.pending = "";
    scan.note = "could not reach the Mac";
  }
  render();
}

let scanToggled = 0;
// Back to a clean start: stop any training, forget the scene on screen and any scan in
// progress, put the settings back, and show the real room, ready for a new scan.
async function resetAll() {
  stopReplay();
  mutedRun = state.status?.run ?? state.run;
  scan.active = false;
  scan.pending = "";
  scan.photos = 0;
  scan.poses = [];
  scan.note = "";
  scanToggled = 0;
  markers.clear();
  depth.count = 0;
  depthGeometry.setDrawRange(0, 0);
  depthCloud.visible = false;
  packed.numSplats = 0;
  packed.needsUpdate = true;
  mesh.updateVersion();
  mesh.visible = false;
  shownCount = 0;
  state.meta = null;
  sceneInfo = null;
  roomWanted = true;
  $("room").checked = true;
  $("pace").checked = false;
  $("auto").checked = false;
  $("steps").value = 3000;
  setBackdrop(false);
  placeStage();
  render();
  await Promise.all([
    fetch("/api/reset", { method: "POST" }).catch(() => {}),
    fetch("/scan/reset", { method: "POST" }).catch(() => {}),
  ]);
  openCameras().catch(() => {});
  render();
}
$("reset").addEventListener("click", resetAll);

function toggleScan() {
  // One pull of the trigger can register twice; a scan that short has nothing in it.
  if (performance.now() - scanToggled < 2500) return;
  scanToggled = performance.now();
  if (scan.active) finishScan();
  else if (!scan.pending) startScan();
}

// Follows the Mac while it builds, uploads and starts training the scan.
async function followScan() {
  for (;;) {
    if (scan.pending && scan.pending !== "sending") {
      try {
        const status = await (await fetch("/scan/status")).json();
        if (status.id === scan.id) {
          if (status.state === "error") {
            scan.note = `scan failed: ${status.message}`;
            scan.pending = "";
            placeBoard();
          } else if (status.state === "training") {
            scan.pending = "";
            placeBoard();
          } else {
            scan.pending = status.state;
          }
          render();
        }
      } catch (error) {
        // proxy busy: ask again
      }
    }
    await sleep(600);
  }
}

$("cameras").addEventListener("click", () => {
  scan.note = "opening the cameras…";
  render();
  openCameras().then(() => {
    scan.note = scan.cameras.length ? "" : "no passthrough camera found in this browser";
    render();
  }).catch((error) => {
    scan.note = /video source/.test(error.message)
      ? "the cameras are in use: close other tabs of this page, then press again"
      : `camera: ${error.message}`;
    report("camera_error", `${error.name}: ${error.message}`);
    render();
  });
});
$("scan").addEventListener("click", toggleScan);
// Once the camera permission has been given, open the cameras without being asked:
// on load, when the tab comes to the front, and on entering AR. A tab in the background
// or another tab holding the cameras makes this fail, so it is tried again.
let openingCameras = false;
async function autoOpenCameras() {
  if (scan.cameras.length || openingCameras) return;
  openingCameras = true;
  try {
    const permission = await navigator.permissions.query({ name: "camera" });
    if (permission.state === "granted") await openCameras();
  } catch (error) {
    console.warn("cameras not opened yet", error);
  }
  openingCameras = false;
}
autoOpenCameras();
document.addEventListener("visibilitychange", autoOpenCameras);
window.addEventListener("focus", autoOpenCameras);

// ---------------------------------------------------------------- readouts

const LABELS = { idle: "Idle. Pick a scene and start.", loading: "Loading images…",
  training: "Training", done: "Done", stopped: "Stopped", error: "Error", offline: "Server offline" };

const SCAN_STEPS = { sending: "Sending the last photos", building: "Working out camera poses",
  uploading: "Uploading to the GPU", starting: "Loading on the GPU" };

function readout() {
  const status = state.status || {};
  const meta = state.meta || {};
  if (scan.active || scan.pending) {
    return {
      scanning: true, copying: "", step: 0, max: 1, replay: false,
      progress: scan.active ? scan.photos / SCAN.maxPhotos : 0,
      label: scan.active ? "Scanning" : "Preparing", scene: scan.id,
      big: scan.active ? `${scan.photos} photos` : SCAN_STEPS[scan.pending] || scan.pending,
      count: depth.count,
      small: !scan.active ? "the scene will appear around you"
        : scan.turning > SCAN.maxTurn ? "turning too fast: slow down"
        : scan.photos >= SCAN.maxPhotos ? "enough photos: pull the trigger"
        : "move slowly around things   ·   trigger: finish",
    };
  }
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
  $("scan").textContent = scan.active ? `Finish scan (${scan.photos})` : "Scan";
  $("scan").disabled = !scan.cameras.length || (!!scan.pending && !scan.active);
  $("cameras").hidden = scan.cameras.length > 0;
  $("scanNote").textContent = scan.note || (r.scanning ? `${r.label}: ${r.big}`
    : scan.cameras.length ? `Cameras ready: ${scan.cameras.map((c) => c.side).join(", ")}. In AR, pull the trigger to scan.`
    : "");
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
board.visible = false;

// Beside the hologram, or on the left controller like a wristwatch when at room scale.
function placeBoard() {
  if ((scan.active || scan.pending) && renderer.xr.isPresenting) {
    // While scanning, keep the readout in front of the eyes: it is the only feedback.
    camera.add(board);
    board.position.set(0, -0.2, -0.75);
    board.rotation.set(-0.35, 0, 0);
    board.scale.setScalar(0.62);
  } else if (inRoom() && renderer.xr.isPresenting) {
    const hand = hands.find((item) => item.handedness === "left") || hands[0];
    hand.grip.add(board);
    board.position.set(0, 0.09, 0.03);
    board.rotation.set(-1.1, 0, 0);
    board.scale.setScalar(0.34);
  } else {
    stage.add(board);
    board.position.set(0, -0.42, 0.25);
    board.rotation.set(-0.5, 0, 0);
    board.scale.setScalar(1);
  }
}

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
  g.font = r.scanning && r.big.length > 14 ? "600 56px system-ui, sans-serif" : "600 84px system-ui, sans-serif";
  g.fillText(r.scanning ? r.big : `step ${r.step.toLocaleString()}`, 44, 164);
  g.font = "40px system-ui, sans-serif";
  g.fillStyle = "#c8cedb";
  const count = r.count ? `${r.count.toLocaleString()} Gaussians` : "";
  const psnr = r.psnr ? `PSNR ${r.psnr.toFixed(1)} dB` : "";
  g.fillText(r.scanning ? r.small : [count, psnr].filter(Boolean).join("   ·   "), 44, 232);
  if (r.scanning && scan.active) {
    g.fillStyle = "#7cf3c6";
    g.fillText(depth.count ? `${depth.count.toLocaleString()} depth points` : "no depth yet", 560, 164);
  }
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
  const hint = scan.note ? scan.note
    : r.copying ? `replay arriving: ${r.copying}`
    : inRoom() ? "trigger: new scan   ·   B/Y: real room on/off   ·   A/X: table top   ·   stick: reset"
    : "trigger: scan   ·   grip: move   ·   both grips: resize   ·   stick: reset";
  g.fillText(hint, 44, 348);
  boardTexture.needsUpdate = true;
}

// ---------------------------------------------------------------- WebXR

// Asking for depth sensing has kept the AR session from starting on the Quest 3S, so it
// is off unless the page is opened with ?depth in the address.
const DEPTH_REQUESTED = new URLSearchParams(location.search).has("depth");

const xr = new SparkXr({
  renderer,
  mode: "arvr",
  element: $("xr"),
  referenceSpaceType: "local-floor",
  frameBufferScaleFactor: 0.7,
  sessionInit: DEPTH_REQUESTED ? {
    optionalFeatures: ["depth-sensing"],
    depthSensing: { usagePreference: ["cpu-optimized", "gpu-optimized"],
                    dataFormatPreference: ["luminance-alpha", "float32", "unsigned-short"] },
  } : {},
  onEnterXr: () => {
    $("xr").textContent = "Exit AR";
    board.visible = true;
    controls.enabled = false;
    setTimeout(() => {
      // Entering AR with a room-scale scene already loaded: show it on black.
      solidBackground = inRoom() && mesh.visible && packed.numSplats > 0;
      placeStage();
    }, 300); // once the session is presenting
    autoOpenCameras();
  },
  onExitXr: () => {
    $("xr").textContent = "Enter AR";
    board.visible = false;
    controls.enabled = true;
    setTimeout(placeStage, 300);
  },
});
// three.js would use the depth map to hide virtual things behind real surfaces. The
// reconstruction sits exactly on those surfaces, so it must not be hidden by them.
renderer.xr.getDepthSensingMesh = () => null;

window.liveSplat = { xr, spark, packed, mesh, content, state, scan, depth, THREE, PackedSplats, SplatMesh };

// Hologram: grip moves it, both grips resize it. Room: grip slides the whole scene, both
// grips also turn it about the vertical, to line it up with the real room.
// Trigger starts and finishes a scan. A/X switches hologram and room; B/Y hides the real
// room; pressing a thumbstick resets everything, ready for a new scan.
const hands = [0, 1].map((index) => {
  const grip = renderer.xr.getControllerGrip(index);
  localFrame.add(grip);
  const hand = { grip, squeezing: false, handedness: "", pressed: [] };
  grip.addEventListener("connected", (event) => { hand.handedness = event.data.handedness; });
  grip.addEventListener("squeezestart", () => { hand.squeezing = true; regrab(); });
  grip.addEventListener("squeezeend", () => { hand.squeezing = false; regrab(); });
  renderer.xr.getController(index).addEventListener("select", toggleScan);
  return hand;
});
let pinch = null;
let drag = null;

// Midpoint of the gripping hands, and the heading of the line between two of them.
function grasp(held) {
  const a = held[0].grip.getWorldPosition(new THREE.Vector3());
  if (held.length < 2) return { middle: a, heading: 0 };
  const b = held[1].grip.getWorldPosition(new THREE.Vector3());
  return { middle: a.clone().add(b).multiplyScalar(0.5), heading: Math.atan2(b.x - a.x, b.z - a.z) };
}

function regrab() {
  const held = hands.filter((hand) => hand.squeezing);
  scene.attach(stage);
  pinch = null;
  if (inRoom()) {
    if (drag && !held.length) saveAlignment();
    drag = held.length
      ? { held, ...grasp(held), position: stage.position.clone(), yaw: stage.rotation.y }
      : null;
    return;
  }
  drag = null;
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

function setRoom(wanted) {
  roomWanted = wanted;
  $("room").checked = wanted;
  drag = pinch = null;
  layout(true);
  render();
}
$("room").addEventListener("change", () => setRoom($("room").checked));

const BUTTONS = {
  4: () => setRoom(!roomWanted), // A / X
  5: () => setBackdrop(!solidBackground), // B / Y
  3: resetAll, // thumbstick press
};

function pollButtons() {
  const sources = renderer.xr.getSession()?.inputSources || [];
  let index = 0;
  for (const source of sources) {
    const hand = hands[index++];
    if (!hand || !source.gamepad) continue;
    for (const [button, action] of Object.entries(BUTTONS)) {
      const down = !!source.gamepad.buttons[button]?.pressed;
      if (down && !hand.pressed[button]) action();
      hand.pressed[button] = down;
    }
  }
}

const VERTICAL = new THREE.Vector3(0, 1, 0);

renderer.setAnimationLoop((time, frame) => {
  if (renderer.xr.isPresenting) {
    pollButtons();
    recordPose(time, frame);
  }
  if (pinch) {
    const scale = THREE.MathUtils.clamp(pinch.scale * handDistance() / pinch.distance, 0.2, 8);
    stage.scale.setScalar(scale);
  }
  if (drag) {
    const now = grasp(drag.held);
    const turn = drag.held.length === 2 ? now.heading - drag.heading : 0;
    stage.rotation.set(0, drag.yaw + turn, 0);
    stage.position.copy(drag.position).sub(drag.middle).applyAxisAngle(VERTICAL, turn)
      .add(now.middle);
  }
  if (!renderer.xr.isPresenting) controls.update();
  renderer.render(scene, camera);
});

followStatus();
followFrames();
followRecordings();
followScan();
