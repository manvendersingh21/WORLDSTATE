/**
 * WORLDSTATE — 3D Digital Twin module (isolated, self-contained).
 *
 * ES module exporting `mountTwin(container, { assetsBase, dataBase })`.
 * Replays the simulator's recorded per-frame world poses (MuJoCo Z-up, meters)
 * of the Panda arm + cube, synchronized to WORLDSTATE states via setTime().
 *
 * No network access except `assetsBase` (GLBs) and `dataBase` (twin JSON).
 * Three.js is vendored locally under ./vendor/ — no CDN at runtime.
 */

import * as THREE from './vendor/three.module.js';
import { OrbitControls } from './vendor/OrbitControls.js';
import { GLTFLoader } from './vendor/GLTFLoader.js';
import { RoomEnvironment } from './vendor/RoomEnvironment.js';

const LINK_ORDER = [
  'link0', 'link1', 'link2', 'link3', 'link4', 'link5', 'link6', 'link7',
  'hand', 'leftfinger', 'rightfinger',
];

const BG_COLOR = 0x080a0d;
const REFERENCE_RUN = 'normal_16'; // normal episode used for the "expected" path

// ---------------------------------------------------------------------------

function injectStylesheet() {
  if (document.querySelector('link[data-twin-css], style[data-twin-css]')) return;
  const link = document.createElement('link');
  link.rel = 'stylesheet';
  link.href = new URL('./twin.css', import.meta.url).href;
  link.setAttribute('data-twin-css', '');
  document.head.appendChild(link);
}

function el(tag, attrs = {}, text) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (text != null) node.textContent = text;
  return node;
}

function quatFromWxyz(q, out) {
  out.set(q[1], q[2], q[3], q[0]); // wxyz -> THREE xyzw
  return out;
}

// ---------------------------------------------------------------------------

export function mountTwin(container, options = {}) {
  injectStylesheet();

  const assetsBase = (options.assetsBase ?? new URL('./assets', import.meta.url).href).replace(/\/$/, '');
  const dataBase = (options.dataBase ?? new URL('./data', import.meta.url).href).replace(/\/$/, '');

  // ----- DOM -----------------------------------------------------------------
  const root = el('div', { class: 'twin-root', 'data-state': '', 'data-status': 'loading' });

  const hud = el('div', { class: 'twin-hud' });

  const topLeft = el('div', { class: 'twin-topleft' });
  const stateChip = el('div', { class: 'twin-state-chip' });
  stateChip.appendChild(el('span', { class: 'twin-chip-dot' }));
  const stateLabel = el('span', { id: 'twin-state-label' }, '—');
  stateChip.appendChild(stateLabel);
  topLeft.appendChild(stateChip);

  const divergence = el('div', { id: 'twin-divergence', class: 'twin-divergence' });
  divergence.style.display = 'none';
  topLeft.appendChild(divergence);
  hud.appendChild(topLeft);

  const controlsBox = el('div', { class: 'twin-viewbar' });
  const presetDefs = [
    ['perspective', 'Perspective'],
    ['front', 'Front'],
    ['side', 'Side'],
    ['top', 'Top'],
  ];
  const presetButtons = {};
  for (const [key, label] of presetDefs) {
    const b = el('button', { type: 'button', class: 'twin-btn', 'data-preset': key }, label);
    presetButtons[key] = b;
    controlsBox.appendChild(b);
  }
  const resetBtn = el('button', { type: 'button', class: 'twin-btn twin-btn-reset', id: 'twin-reset' }, 'Reset View');
  controlsBox.appendChild(resetBtn);
  hud.appendChild(controlsBox);

  const noData = el('div', { id: 'twin-no-data', class: 'twin-no-data' }, 'twin data unavailable');
  noData.style.display = 'none';
  hud.appendChild(noData);

  hud.appendChild(el('div', { class: 'twin-provenance' },
    'Digital twin: replay of the simulator\u2019s recorded trajectory, synchronized to WORLDSTATE states'));

  root.appendChild(hud);
  container.appendChild(root);

  // ----- Renderer / scene ----------------------------------------------------
  const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.domElement.id = 'twin-canvas';
  renderer.domElement.className = 'twin-canvas';
  root.insertBefore(renderer.domElement, hud);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(BG_COLOR);

  const pmrem = new THREE.PMREMGenerator(renderer);
  scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
  scene.environmentIntensity = 0.45;

  // MuJoCo data is Z-up; keep the scene Z-up and tell the camera.
  const camera = new THREE.PerspectiveCamera(36, 1, 0.05, 50);
  camera.up.set(0, 0, 1);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.minDistance = 0.35;
  controls.maxDistance = 7;

  const PRESETS = {
    perspective: { pos: [1.3, -1.1, 1.7], target: [0, 0, 0.85], fov: 36 },
    // Front ~= dataset "frontview" camera (pos [1.6,0,1.45], fovy 28, looking slightly down at the cell)
    front: { pos: [1.6, 0, 1.45], target: [0, 0, 1.0], fov: 28 },
    side: { pos: [0, -2.05, 1.15], target: [0, 0, 0.85], fov: 32 },
    top: { pos: [0.02, -0.14, 2.9], target: [0, 0, 0.8], fov: 34 },
  };

  function setPreset(name) {
    const p = PRESETS[name] || PRESETS.perspective;
    camera.position.set(...p.pos);
    camera.fov = p.fov;
    camera.updateProjectionMatrix();
    controls.target.set(...p.target);
    controls.update();
    for (const [key, btn] of Object.entries(presetButtons)) {
      btn.classList.toggle('twin-btn-active', key === (PRESETS[name] ? name : 'perspective'));
    }
  }
  function resetView() { setPreset('perspective'); }
  setPreset('perspective');

  // ----- Lights --------------------------------------------------------------
  const keyLight = new THREE.DirectionalLight(0xffffff, 2.6);
  keyLight.position.set(1.5, -1.1, 2.9);
  keyLight.castShadow = true;
  keyLight.shadow.mapSize.set(2048, 2048);
  keyLight.shadow.camera.near = 0.5;
  keyLight.shadow.camera.far = 7;
  keyLight.shadow.camera.left = -1.4;
  keyLight.shadow.camera.right = 1.4;
  keyLight.shadow.camera.top = 1.4;
  keyLight.shadow.camera.bottom = -1.4;
  keyLight.shadow.bias = -0.0002;
  keyLight.shadow.radius = 4;
  keyLight.target.position.set(0, 0, 0.8);
  scene.add(keyLight, keyLight.target);

  const fillLight = new THREE.DirectionalLight(0xbdd0e4, 0.55);
  fillLight.position.set(-1.8, 1.5, 1.4);
  scene.add(fillLight);

  // ----- Static cell ---------------------------------------------------------
  const disposables = [];
  function track(obj) { disposables.push(obj); return obj; }

  const floorMat = track(new THREE.MeshStandardMaterial({ color: 0x0c0f13, roughness: 0.96, metalness: 0.0 }));
  const floor = new THREE.Mesh(track(new THREE.CircleGeometry(3.2, 48)), floorMat);
  floor.receiveShadow = true;
  scene.add(floor);

  const tableMat = track(new THREE.MeshStandardMaterial({ color: 0x2a2e33, roughness: 0.55, metalness: 0.25 }));
  const legMat = track(new THREE.MeshStandardMaterial({ color: 0x1b1e22, roughness: 0.6, metalness: 0.35 }));

  // World constants from the agreed schema (overridden by data/index.json when loaded).
  const world = {
    tableTopZ: 0.8,
    tableFull: [0.8, 0.8, 0.05],
    padCenter: [0.0, 0.12, 0.801],
    padHalf: [0.04, 0.04, 0.001],
    padRgb: [0.15, 0.7, 0.25],
    cubeHalf: [0.02035, 0.02152, 0.02147],
    robotBase: [-0.56, 0.0, 0.912],
  };

  const cellGroup = new THREE.Group();
  scene.add(cellGroup);

  function buildCell() {
    cellGroup.clear();
    const [tw, td, th] = world.tableFull;
    const top = new THREE.Mesh(track(new THREE.BoxGeometry(tw, td, th)), tableMat);
    top.position.set(0, 0, world.tableTopZ - th / 2);
    top.castShadow = true;
    top.receiveShadow = true;
    cellGroup.add(top);

    const legH = world.tableTopZ - th;
    const legGeo = track(new THREE.BoxGeometry(0.06, 0.06, legH));
    for (const sx of [-1, 1]) for (const sy of [-1, 1]) {
      const leg = new THREE.Mesh(legGeo, legMat);
      leg.position.set(sx * (tw / 2 - 0.06), sy * (td / 2 - 0.06), legH / 2);
      leg.castShadow = true;
      leg.receiveShadow = true;
      cellGroup.add(leg);
    }

    const padMat = track(new THREE.MeshStandardMaterial({
      color: new THREE.Color(...world.padRgb), roughness: 0.75, metalness: 0.05,
    }));
    const pad = new THREE.Mesh(
      track(new THREE.BoxGeometry(world.padHalf[0] * 2, world.padHalf[1] * 2, world.padHalf[2] * 2)),
      padMat);
    pad.position.set(...world.padCenter);
    pad.receiveShadow = true;
    cellGroup.add(pad);

    // Pedestal under the robot base (the recorded base pose floats at ~0.91 m)
    const [bx, by, bz] = world.robotBase;
    const pedestal = new THREE.Mesh(track(new THREE.BoxGeometry(0.3, 0.3, bz)), legMat);
    pedestal.position.set(bx, by, bz / 2);
    pedestal.castShadow = true;
    pedestal.receiveShadow = true;
    cellGroup.add(pedestal);
    const basePlate = new THREE.Mesh(track(new THREE.BoxGeometry(0.42, 0.42, 0.02)), tableMat);
    basePlate.position.set(bx, by, 0.01);
    basePlate.receiveShadow = true;
    cellGroup.add(basePlate);
  }
  buildCell();

  const cubeMat = track(new THREE.MeshStandardMaterial({ color: 0xb13030, roughness: 0.5, metalness: 0.08 }));
  let cube = new THREE.Mesh(
    track(new THREE.BoxGeometry(world.cubeHalf[0] * 2, world.cubeHalf[1] * 2, world.cubeHalf[2] * 2)),
    cubeMat);
  cube.castShadow = true;
  cube.receiveShadow = true;
  cube.visible = false;
  scene.add(cube);

  // Divergence marker (subtle red ring + dot at the cube's divergence position)
  const markerGroup = new THREE.Group();
  const ringMat = track(new THREE.MeshBasicMaterial({ color: 0xff5050, transparent: true, opacity: 0.8, side: THREE.DoubleSide }));
  const ring = new THREE.Mesh(track(new THREE.RingGeometry(0.045, 0.055, 48)), ringMat);
  const dotMat = track(new THREE.MeshBasicMaterial({ color: 0xff5050, transparent: true, opacity: 0.9 }));
  const dot = new THREE.Mesh(track(new THREE.SphereGeometry(0.008, 16, 12)), dotMat);
  markerGroup.add(ring, dot);
  markerGroup.visible = false;
  scene.add(markerGroup);

  // Trajectory overlays (expected = amber from a normal episode, actual = cyan)
  const expectedLineMat = track(new THREE.LineBasicMaterial({ color: 0xc9a227, transparent: true, opacity: 0.55 }));
  const actualLineMat = track(new THREE.LineBasicMaterial({ color: 0x37c3d6, transparent: true, opacity: 0.7 }));
  const divergeDotMat = track(new THREE.MeshBasicMaterial({ color: 0xff5050 }));
  let expectedLine = null;
  let actualLine = null;
  let divergeDot = null;

  // ----- Robot meshes ----------------------------------------------------------
  // The GLBs carry their own PBR white/graphite materials (baked from the MJCF),
  // but ship positions only — compute smooth vertex normals so PBR shading works.
  const linkObjects = new Array(LINK_ORDER.length).fill(null);
  const loader = new GLTFLoader();
  let disposed = false;

  const assetsReady = Promise.all(LINK_ORDER.map(async (name, i) => {
    try {
      const gltf = await loader.loadAsync(`${assetsBase}/${name}.glb`);
      if (disposed) return;
      const obj = gltf.scene;
      obj.traverse((child) => {
        if (child.isMesh) {
          child.castShadow = true;
          child.receiveShadow = true;
          if (!child.geometry.getAttribute('normal')) {
            child.geometry.computeVertexNormals();
          }
          if (child.material) {
            child.material.flatShading = false;
            child.material.needsUpdate = true;
            track(child.material);
          }
          track(child.geometry);
        }
      });
      obj.visible = false;
      linkObjects[i] = obj;
      scene.add(obj);
    } catch (err) {
      console.warn(`[twin] failed to load asset ${name}.glb`, err);
    }
  })).then(() => { applyTime(currentTime, true); });

  // ----- Run data / analysis state ---------------------------------------------
  let runData = null;        // current run twin JSON
  let referenceData = null;  // normal episode used for the expected path
  let analysis = null;
  let currentTime = 0;
  let playing = false;
  let loadSeq = 0;

  const fps = () => (runData ? runData.fps : 20);
  const duration = () => (runData ? (runData.n_frames - 1) / runData.fps : 8.95);

  const tmpQa = new THREE.Quaternion();
  const tmpQb = new THREE.Quaternion();
  const tmpVa = new THREE.Vector3();
  const tmpVb = new THREE.Vector3();

  function lerpPose(obj, pa, pb, qa, qb, alpha) {
    tmpVa.set(pa[0], pa[1], pa[2]);
    tmpVb.set(pb[0], pb[1], pb[2]);
    obj.position.copy(tmpVa.lerp(tmpVb, alpha));
    quatFromWxyz(qa, tmpQa);
    quatFromWxyz(qb, tmpQb);
    obj.quaternion.copy(tmpQa.slerp(tmpQb, alpha));
  }

  function stateAt(t) {
    if (!analysis || !Array.isArray(analysis.sequence) || analysis.sequence.length === 0) return null;
    let cur = null;
    for (const s of analysis.sequence) {
      if (s.t <= t + 1e-9) cur = s; else break;
    }
    return cur || analysis.sequence[0];
  }

  function applyTime(t, force = false) {
    currentTime = Math.max(0, Math.min(t, duration()));
    if (runData) {
      const frames = runData.frames;
      const f = currentTime * runData.fps;
      const i0 = Math.min(Math.floor(f), frames.length - 1);
      const i1 = Math.min(i0 + 1, frames.length - 1);
      const alpha = Math.min(Math.max(f - i0, 0), 1);
      const fa = frames[i0];
      const fb = frames[i1];
      for (let i = 0; i < LINK_ORDER.length; i++) {
        const obj = linkObjects[i];
        if (!obj) continue;
        obj.visible = true;
        lerpPose(obj, fa.link_pos[i], fb.link_pos[i], fa.link_quat[i], fb.link_quat[i], alpha);
      }
      cube.visible = true;
      lerpPose(cube, fa.cube_pos, fb.cube_pos, fa.cube_quat, fb.cube_quat, alpha);
    }

    // State label
    const st = stateAt(currentTime);
    const name = st ? st.name : '—';
    if (stateLabel.textContent !== name || force) {
      stateLabel.textContent = name;
      root.setAttribute('data-state', st ? st.name : '');
    }

    // Divergence marker + label
    const fds = analysis && analysis.first_divergence_s != null ? analysis.first_divergence_s : null;
    const diverged = fds != null && currentTime >= fds - 1e-9;
    divergence.style.display = diverged ? 'flex' : 'none';
    markerGroup.visible = diverged && runData != null;
  }

  function clearLine(line) {
    if (!line) return null;
    scene.remove(line);
    line.geometry.dispose();
    return null;
  }

  function buildTrajectories() {
    expectedLine = clearLine(expectedLine);
    actualLine = clearLine(actualLine);
    if (divergeDot) { scene.remove(divergeDot); divergeDot.geometry.dispose(); divergeDot = null; }

    const fds = analysis && analysis.first_divergence_s != null ? analysis.first_divergence_s : null;
    const anomalous = analysis && analysis.status && analysis.status !== 'normal';
    if (!runData || !anomalous) return;

    const toPoints = (frames) => frames.map((f) => new THREE.Vector3(...f.eef_pos));
    if (referenceData && referenceData.id !== runData.id) {
      const g = new THREE.BufferGeometry().setFromPoints(toPoints(referenceData.frames));
      expectedLine = new THREE.Line(g, expectedLineMat);
      scene.add(expectedLine);
    }
    const ga = new THREE.BufferGeometry().setFromPoints(toPoints(runData.frames));
    actualLine = new THREE.Line(ga, actualLineMat);
    scene.add(actualLine);

    if (fds != null) {
      const fi = Math.min(Math.round(fds * runData.fps), runData.frames.length - 1);
      const fr = runData.frames[fi];
      divergeDot = new THREE.Mesh(new THREE.SphereGeometry(0.01, 16, 12), divergeDotMat);
      divergeDot.position.set(...fr.eef_pos);
      scene.add(divergeDot);
      markerGroup.position.set(fr.cube_pos[0], fr.cube_pos[1], fr.cube_pos[2] + 0.001);
      ring.position.z = world.cubeHalf[2] + 0.004;
    }
  }

  async function fetchReference() {
    if (referenceData || !runData || runData.id === REFERENCE_RUN) return;
    try {
      const res = await fetch(`${dataBase}/${REFERENCE_RUN}.json`);
      if (res.ok) {
        referenceData = await res.json();
        buildTrajectories();
      }
    } catch { /* expected path is optional */ }
  }

  function syncStatus() {
    if (!runData) return; // keep 'loading' / 'no-data'
    root.setAttribute('data-status', analysis && analysis.status ? analysis.status : 'ready');
  }

  // ----- Public API --------------------------------------------------------------
  async function load(runId) {
    const seq = ++loadSeq;
    root.setAttribute('data-status', 'loading');
    noData.style.display = 'none';
    let data = null;
    try {
      const res = await fetch(`${dataBase}/${runId}.json`);
      if (res.ok) data = await res.json();
    } catch { /* treated as missing data below */ }
    if (seq !== loadSeq || disposed) return;

    if (!data || !Array.isArray(data.frames) || data.frames.length === 0) {
      runData = null;
      cube.visible = false;
      for (const obj of linkObjects) if (obj) obj.visible = false;
      expectedLine = clearLine(expectedLine);
      actualLine = clearLine(actualLine);
      markerGroup.visible = false;
      root.setAttribute('data-status', 'no-data');
      noData.style.display = 'flex';
      applyTime(0, true);
      return;
    }

    runData = data;
    syncStatus();
    buildTrajectories();
    fetchReference();
    applyTime(0, true);
  }

  function setAnalysis(a) {
    analysis = a || null;
    if (analysis && analysis.first_divergence_s != null) {
      divergence.textContent = `REALITY DIVERGED \u2014 ${Number(analysis.first_divergence_s).toFixed(3)}s`;
    } else {
      divergence.textContent = '';
    }
    syncStatus();
    buildTrajectories();
    applyTime(currentTime, true);
  }

  function setTime(seconds) {
    applyTime(Number(seconds) || 0);
  }

  function play() { playing = true; }
  function pause() { playing = false; }

  // ----- Render loop ---------------------------------------------------------------
  let rafId = 0;
  let lastNow = performance.now();
  function tick(now) {
    rafId = requestAnimationFrame(tick);
    const dt = Math.min((now - lastNow) / 1000, 0.1);
    lastNow = now;
    if (playing && runData) {
      let t = currentTime + dt;
      if (t > duration()) t = 0; // loop
      applyTime(t);
    }
    if (markerGroup.visible) {
      const s = 1 + 0.06 * Math.sin(now * 0.004);
      ring.scale.set(s, s, 1);
    }
    controls.update();
    renderer.render(scene, camera);
  }
  rafId = requestAnimationFrame(tick);

  // ----- Sizing ---------------------------------------------------------------------
  function resize() {
    const w = container.clientWidth || 1;
    const h = container.clientHeight || 1;
    renderer.setSize(w, h, false);
    renderer.domElement.style.width = '100%';
    renderer.domElement.style.height = '100%';
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  const ro = new ResizeObserver(resize);
  ro.observe(container);
  resize();

  // ----- UI wiring --------------------------------------------------------------------
  for (const [key, btn] of Object.entries(presetButtons)) {
    btn.addEventListener('click', () => setPreset(key));
  }
  resetBtn.addEventListener('click', resetView);

  // ----- Dispose ----------------------------------------------------------------------
  function dispose() {
    if (disposed) return;
    disposed = true;
    cancelAnimationFrame(rafId);
    ro.disconnect();
    controls.dispose();
    expectedLine = clearLine(expectedLine);
    actualLine = clearLine(actualLine);
    if (divergeDot) { scene.remove(divergeDot); divergeDot.geometry.dispose(); divergeDot = null; }
    for (const obj of disposables) {
      if (obj && typeof obj.dispose === 'function') obj.dispose();
    }
    pmrem.dispose();
    renderer.dispose();
    root.remove();
  }

  return {
    load,
    setTime,
    setAnalysis,
    play,
    pause,
    setPreset,
    resetView,
    dispose,
    // Non-contractual helpers (used by the standalone demo page)
    getTime: () => currentTime,
    getDuration: duration,
    isPlaying: () => playing,
    ready: assetsReady,
    element: root,
  };
}

export default mountTwin;
