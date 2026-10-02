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
import { RoundedBoxGeometry } from './vendor/RoundedBoxGeometry.js';

const LINK_ORDER = [
  'link0', 'link1', 'link2', 'link3', 'link4', 'link5', 'link6', 'link7',
  'hand', 'leftfinger', 'rightfinger',
];

const BG_COLOR = 0x0a0c0f;
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

// ----- Procedural textures (industrial cell look, zero network) -------------

function makeConcreteTexture() {
  const s = 1024;
  const c = document.createElement('canvas');
  c.width = c.height = s;
  const g = c.getContext('2d');
  g.fillStyle = '#313437';
  g.fillRect(0, 0, s, s);
  // Large soft tonal blotches (trowel / cure variation)
  for (let i = 0; i < 70; i++) {
    const x = Math.random() * s, y = Math.random() * s, r = 60 + Math.random() * 190;
    const dark = Math.random() < 0.55;
    const grad = g.createRadialGradient(x, y, 0, x, y, r);
    grad.addColorStop(0, dark ? 'rgba(0,0,0,0.06)' : 'rgba(255,255,255,0.04)');
    grad.addColorStop(1, 'rgba(0,0,0,0)');
    g.fillStyle = grad;
    g.beginPath(); g.arc(x, y, r, 0, Math.PI * 2); g.fill();
  }
  // Fine aggregate speckle
  for (let i = 0; i < 15000; i++) {
    const a = 0.02 + Math.random() * 0.05;
    g.fillStyle = Math.random() < 0.5 ? `rgba(255,255,255,${a})` : `rgba(0,0,0,${a})`;
    g.fillRect(Math.random() * s, Math.random() * s, 1.4, 1.4);
  }
  // Saw-cut joint lines: 2x2 tiles per texture repeat
  g.strokeStyle = 'rgba(10,11,12,0.8)';
  g.lineWidth = 4;
  for (const p of [0, s / 2, s]) {
    g.beginPath(); g.moveTo(p, 0); g.lineTo(p, s); g.stroke();
    g.beginPath(); g.moveTo(0, p); g.lineTo(s, p); g.stroke();
  }
  g.strokeStyle = 'rgba(255,255,255,0.045)';
  g.lineWidth = 1.5;
  for (const p of [3, s / 2 + 3]) {
    g.beginPath(); g.moveTo(p, 0); g.lineTo(p, s); g.stroke();
    g.beginPath(); g.moveTo(0, p); g.lineTo(s, p); g.stroke();
  }
  const tex = new THREE.CanvasTexture(c);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

function makeBrushedTexture() {
  // Luminance map used as roughnessMap: base ~0.72 with horizontal streaks.
  const w = 512, h = 512;
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  const g = c.getContext('2d');
  g.fillStyle = '#b8b8b8';
  g.fillRect(0, 0, w, h);
  for (let i = 0; i < 2600; i++) {
    const y = Math.random() * h;
    const x = Math.random() * w;
    const len = 30 + Math.random() * 190;
    const a = 0.03 + Math.random() * 0.09;
    g.strokeStyle = Math.random() < 0.5 ? `rgba(255,255,255,${a})` : `rgba(40,40,40,${a})`;
    g.lineWidth = 0.8 + Math.random() * 0.9;
    g.beginPath(); g.moveTo(x, y); g.lineTo(x + len, y); g.stroke();
  }
  const tex = new THREE.CanvasTexture(c);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  return tex;
}

function makeHazardTexture() {
  // Muted industrial yellow/black diagonal tape, tileable along X.
  const w = 128, h = 64;
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  const g = c.getContext('2d');
  g.fillStyle = '#17171a';
  g.fillRect(0, 0, w, h);
  g.fillStyle = '#8f7b25';
  const stripe = 32;
  for (let x = -h; x < w + h; x += stripe * 2) {
    g.beginPath();
    g.moveTo(x, h); g.lineTo(x + h, 0); g.lineTo(x + h + stripe, 0); g.lineTo(x + stripe, h);
    g.closePath(); g.fill();
  }
  // grime
  for (let i = 0; i < 900; i++) {
    g.fillStyle = `rgba(0,0,0,${0.04 + Math.random() * 0.1})`;
    g.fillRect(Math.random() * w, Math.random() * h, 1.5, 1.5);
  }
  const tex = new THREE.CanvasTexture(c);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

function makeShadowBlobTexture() {
  const s = 256;
  const c = document.createElement('canvas');
  c.width = c.height = s;
  const g = c.getContext('2d');
  const grad = g.createRadialGradient(s / 2, s / 2, 0, s / 2, s / 2, s / 2);
  grad.addColorStop(0, 'rgba(0,0,0,0.85)');
  grad.addColorStop(0.55, 'rgba(0,0,0,0.38)');
  grad.addColorStop(1, 'rgba(0,0,0,0)');
  g.fillStyle = grad;
  g.fillRect(0, 0, s, s);
  return new THREE.CanvasTexture(c);
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
  const stateLabel = el('span', { id: 'twin-state-label' }, '');
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
  renderer.toneMappingExposure = 1.12;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.domElement.id = 'twin-canvas';
  renderer.domElement.className = 'twin-canvas';
  root.insertBefore(renderer.domElement, hud);

  const maxAniso = Math.min(renderer.capabilities.getMaxAnisotropy() || 1, 8);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(BG_COLOR);
  // Subtle depth haze: industrial hall falling off into darkness.
  scene.fog = new THREE.Fog(BG_COLOR, 4.5, 12.5);

  const pmrem = new THREE.PMREMGenerator(renderer);
  scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
  scene.environmentIntensity = 0.55;

  // MuJoCo data is Z-up; keep the scene Z-up and tell the camera.
  const camera = new THREE.PerspectiveCamera(38, 1, 0.05, 50);
  camera.up.set(0, 0, 1);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.minDistance = 0.35;
  controls.maxDistance = 7;

  const PRESETS = {
    // Default framing: whole arm + cell visible with clear headroom at the top
    // (arm reaches link z ~1.56 m when upright at t=0; keep it below the top 10%).
    // Tight 3/4 view on the workbench: cube, gripper and pad fill the middle of the frame.
    perspective: { pos: [0.78, -0.72, 1.16], target: [-0.02, 0.02, 0.86], fov: 38 },
    // Front ~= dataset "frontview" camera (pos [1.6,0,1.45], fovy 28, looking slightly down at the cell)
    front: { pos: [1.6, 0, 1.45], target: [0, 0, 1.0], fov: 28 },
    side: { pos: [0.02, -1.15, 1.02], target: [-0.04, 0.02, 0.86], fov: 36 },
    top: { pos: [0.02, -0.06, 1.85], target: [0, 0.02, 0.82], fov: 35 },
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
  // Key: high bay luminaire — slightly warm, tight high-res soft shadows.
  const keyLight = new THREE.DirectionalLight(0xfff1e0, 2.7);
  keyLight.position.set(1.7, -1.3, 3.1);
  keyLight.castShadow = true;
  keyLight.shadow.mapSize.set(2048, 2048);
  keyLight.shadow.camera.near = 0.5;
  keyLight.shadow.camera.far = 8;
  keyLight.shadow.camera.left = -1.6;
  keyLight.shadow.camera.right = 1.6;
  keyLight.shadow.camera.top = 1.6;
  keyLight.shadow.camera.bottom = -1.6;
  keyLight.shadow.bias = -0.0002;
  keyLight.shadow.normalBias = 0.01;
  keyLight.shadow.radius = 5;
  keyLight.target.position.set(0, 0, 0.8);
  scene.add(keyLight, keyLight.target);

  // Cool sky fill from the opposite side.
  const fillLight = new THREE.DirectionalLight(0xbdd0e4, 0.45);
  fillLight.position.set(-1.8, 1.5, 1.4);
  scene.add(fillLight);

  // Faint cool rim to separate the white shell from the dark hall.
  const rimLight = new THREE.DirectionalLight(0x8fa8c8, 0.5);
  rimLight.position.set(-1.4, 2.2, 2.3);
  scene.add(rimLight);

  // ----- Static cell ---------------------------------------------------------
  const disposables = [];
  function track(obj) { disposables.push(obj); return obj; }

  const concreteTex = track(makeConcreteTexture());
  concreteTex.repeat.set(7, 7);
  concreteTex.anisotropy = maxAniso;
  const concreteBump = track(makeConcreteTexture());
  concreteBump.repeat.set(7, 7);
  concreteBump.anisotropy = maxAniso;
  const brushedTex = track(makeBrushedTexture());
  brushedTex.anisotropy = maxAniso;
  const hazardTex = track(makeHazardTexture());
  hazardTex.anisotropy = maxAniso;
  const blobTex = track(makeShadowBlobTexture());

  // Polished concrete slab with saw-cut joints, fading into the haze.
  const floorMat = track(new THREE.MeshStandardMaterial({
    color: 0x97999c,
    map: concreteTex,
    bumpMap: concreteBump,
    bumpScale: 0.4,
    roughness: 0.88,
    metalness: 0.0,
    envMapIntensity: 0.7,
  }));
  const floor = new THREE.Mesh(track(new THREE.PlaneGeometry(14, 14)), floorMat);
  floor.receiveShadow = true;
  scene.add(floor);

  // Floor safety marking: muted yellow/black tape square around the cell.
  const hazardMat = track(new THREE.MeshStandardMaterial({
    map: hazardTex, roughness: 0.85, metalness: 0.0,
    polygonOffset: true, polygonOffsetFactor: -1,
  }));
  const TAPE = { half: 1.5, w: 0.09 };
  for (let i = 0; i < 4; i++) {
    const horizontal = i < 2;
    const len = TAPE.half * 2 + TAPE.w;
    const geo = track(new THREE.PlaneGeometry(len, TAPE.w));
    const strip = new THREE.Mesh(geo, hazardMat);
    const off = (i % 2 === 0 ? 1 : -1) * TAPE.half;
    if (horizontal) strip.position.set(0, off, 0.002);
    else { strip.position.set(off, 0, 0.002); strip.rotation.z = Math.PI / 2; }
    strip.receiveShadow = true;
    scene.add(strip);
  }

  // Industrial steel bench + graphite structure.
  const tableMat = track(new THREE.MeshStandardMaterial({
    color: 0x484d53,
    roughness: 0.62,
    metalness: 0.72,
    roughnessMap: brushedTex,
    envMapIntensity: 0.9,
  }));
  const legMat = track(new THREE.MeshStandardMaterial({
    color: 0x33383e, roughness: 0.48, metalness: 0.65, envMapIntensity: 0.85,
  }));

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

    // Thin darker rim under the top (bench apron)
    const apron = new THREE.Mesh(track(new THREE.BoxGeometry(tw - 0.02, td - 0.02, 0.035)), legMat);
    apron.position.set(0, 0, world.tableTopZ - th - 0.0175);
    apron.castShadow = true;
    cellGroup.add(apron);

    const legH = world.tableTopZ - th;
    const legGeo = track(new THREE.BoxGeometry(0.055, 0.055, legH));
    for (const sx of [-1, 1]) for (const sy of [-1, 1]) {
      const leg = new THREE.Mesh(legGeo, legMat);
      leg.position.set(sx * (tw / 2 - 0.07), sy * (td / 2 - 0.07), legH / 2);
      leg.castShadow = true;
      leg.receiveShadow = true;
      cellGroup.add(leg);
    }

    // Matte rubber placement mat, slightly inset into the bench top.
    const padMat = track(new THREE.MeshStandardMaterial({
      color: new THREE.Color(world.padRgb[0] * 0.5, world.padRgb[1] * 0.55, world.padRgb[2] * 0.5),
      roughness: 0.95,
      metalness: 0.0,
      envMapIntensity: 0.5,
    }));
    const pad = new THREE.Mesh(
      track(new THREE.BoxGeometry(world.padHalf[0] * 2, world.padHalf[1] * 2, world.padHalf[2] * 2)),
      padMat);
    pad.position.set(world.padCenter[0], world.padCenter[1], world.padCenter[2] - world.padHalf[2] * 0.5);
    pad.receiveShadow = true;
    cellGroup.add(pad);

    // Robot pedestal: steel column + floor flange (recorded base floats at ~0.91 m).
    const [bx, by, bz] = world.robotBase;
    const pedestal = new THREE.Mesh(track(new THREE.CylinderGeometry(0.13, 0.15, bz, 32)), legMat);
    pedestal.rotation.x = Math.PI / 2; // cylinder Y-axis -> Z-up
    pedestal.position.set(bx, by, bz / 2);
    pedestal.castShadow = true;
    pedestal.receiveShadow = true;
    cellGroup.add(pedestal);

    const flange = new THREE.Mesh(track(new THREE.CylinderGeometry(0.21, 0.23, 0.025, 32)), tableMat);
    flange.rotation.x = Math.PI / 2;
    flange.position.set(bx, by, 0.0125);
    flange.castShadow = true;
    flange.receiveShadow = true;
    cellGroup.add(flange);

    // Mounting plate right under the arm base.
    const plate = new THREE.Mesh(track(new THREE.CylinderGeometry(0.105, 0.105, 0.016, 32)), tableMat);
    plate.rotation.x = Math.PI / 2;
    plate.position.set(bx, by, bz + 0.008 - 0.016);
    plate.castShadow = true;
    cellGroup.add(plate);

    // Soft contact shadow under the pedestal.
    const pedBlobMat = track(new THREE.MeshBasicMaterial({
      map: blobTex, transparent: true, opacity: 0.55, depthWrite: false,
    }));
    const pedBlob = new THREE.Mesh(track(new THREE.PlaneGeometry(0.85, 0.85)), pedBlobMat);
    pedBlob.position.set(bx, by, 0.0015);
    pedBlob.renderOrder = 1;
    cellGroup.add(pedBlob);

    // Soft contact shadow under the bench.
    const benchBlobMat = track(new THREE.MeshBasicMaterial({
      map: blobTex, transparent: true, opacity: 0.4, depthWrite: false,
    }));
    const benchBlob = new THREE.Mesh(track(new THREE.PlaneGeometry(tw * 1.5, td * 1.5)), benchBlobMat);
    benchBlob.position.set(0, 0, 0.001);
    benchBlob.renderOrder = 1;
    cellGroup.add(benchBlob);
  }
  buildCell();

  // Work object: red anodized block with beveled edges.
  const cubeMat = track(new THREE.MeshStandardMaterial({
    color: 0x9c2723,
    roughness: 0.48,
    metalness: 0.35,
    envMapIntensity: 0.9,
  }));
  function makeCubeGeometry() {
    const bevel = Math.min(world.cubeHalf[0], world.cubeHalf[1], world.cubeHalf[2]) * 0.22;
    return track(new RoundedBoxGeometry(
      world.cubeHalf[0] * 2, world.cubeHalf[1] * 2, world.cubeHalf[2] * 2, 3, bevel));
  }
  let cube = new THREE.Mesh(makeCubeGeometry(), cubeMat);
  cube.castShadow = true;
  cube.receiveShadow = true;
  cube.visible = false;
  scene.add(cube);

  // Moving contact shadow under the cube (fades as the gripper lifts it).
  const cubeBlobMat = track(new THREE.MeshBasicMaterial({
    map: blobTex, transparent: true, opacity: 0.5, depthWrite: false,
  }));
  const cubeBlob = new THREE.Mesh(track(new THREE.PlaneGeometry(0.095, 0.095)), cubeBlobMat);
  cubeBlob.renderOrder = 1;
  cubeBlob.visible = false;
  scene.add(cubeBlob);

  // Divergence marker (subtle red ring + dot at the cube's divergence position)
  const markerGroup = new THREE.Group();
  const ringMat = track(new THREE.MeshBasicMaterial({ color: 0xff5050, transparent: true, opacity: 0.75, side: THREE.DoubleSide }));
  const ring = new THREE.Mesh(track(new THREE.RingGeometry(0.048, 0.0545, 64)), ringMat);
  const dotMat = track(new THREE.MeshBasicMaterial({ color: 0xff5050, transparent: true, opacity: 0.9 }));
  const dot = new THREE.Mesh(track(new THREE.SphereGeometry(0.008, 16, 12)), dotMat);
  markerGroup.add(ring, dot);
  markerGroup.visible = false;
  scene.add(markerGroup);

  // Trajectory overlays (expected = amber from a normal episode, actual = cyan)
  const expectedLineMat = track(new THREE.LineBasicMaterial({ color: 0xc9a227, transparent: true, opacity: 0.5 }));
  const actualLineMat = track(new THREE.LineBasicMaterial({ color: 0x37c3d6, transparent: true, opacity: 0.65 }));
  const divergeDotMat = track(new THREE.MeshBasicMaterial({ color: 0xff5050 }));
  let expectedLine = null;
  let actualLine = null;
  let divergeDot = null;

  // ----- World constants from the dataset index --------------------------------
  function rebuildWorldDependent() {
    buildCell();
    const old = cube.geometry;
    cube.geometry = makeCubeGeometry();
    if (old) old.dispose();
    applyTime(currentTime, true);
  }
  (async () => {
    try {
      const res = await fetch(`${dataBase}/index.json`);
      if (!res.ok) return;
      const idx = await res.json();
      if (disposed || !idx || !idx.world) return;
      const w = idx.world;
      if (w.table_top_z != null) world.tableTopZ = w.table_top_z;
      if (Array.isArray(w.table_full_size)) world.tableFull = w.table_full_size;
      if (Array.isArray(w.pad_center)) world.padCenter = w.pad_center;
      if (Array.isArray(w.pad_half_size)) world.padHalf = w.pad_half_size;
      if (Array.isArray(w.pad_rgba)) world.padRgb = w.pad_rgba.slice(0, 3);
      if (Array.isArray(w.cube_half_size)) world.cubeHalf = w.cube_half_size;
      if (Array.isArray(w.robot_base_pos)) world.robotBase = w.robot_base_pos;
      rebuildWorldDependent();
    } catch { /* index.json is optional; defaults match the agreed schema */ }
  })();

  // ----- Robot meshes ----------------------------------------------------------
  // The GLBs carry baked PBR colors; restyle per material family so the arm reads
  // as glossy white painted metal + dark graphite joints + brushed-steel fingers.
  function styleRobotMaterial(linkName, mat) {
    if (!mat || !mat.color) return;
    const hsl = { h: 0, s: 0, l: 0 };
    mat.color.getHSL(hsl);
    if (linkName === 'leftfinger' || linkName === 'rightfinger') {
      // Brushed aluminum gripper fingers.
      mat.color.set(0xafb4ba);
      mat.metalness = 0.85;
      mat.roughness = 0.42;
      mat.roughnessMap = brushedTex;
      mat.envMapIntensity = 0.95;
    } else if (hsl.l >= 0.45) {
      // Glossy white painted shell.
      mat.roughness = 0.34;
      mat.metalness = 0.12;
      mat.envMapIntensity = 1.05;
    } else {
      // Dark graphite joints / flanges.
      mat.roughness = 0.46;
      mat.metalness = 0.62;
      mat.envMapIntensity = 0.85;
    }
    mat.flatShading = false;
    mat.needsUpdate = true;
  }

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
            styleRobotMaterial(name, child.material);
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
  let currentFrame = null; // [callouts hook] interpolated frame at currentTime
  const mix3 = (a, b, s) => [a[0] + (b[0] - a[0]) * s, a[1] + (b[1] - a[1]) * s, a[2] + (b[2] - a[2]) * s]; // [callouts hook]
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

  function updateCubeShadow() {
    if (!cube.visible) { cubeBlob.visible = false; return; }
    const restZ = world.tableTopZ + world.cubeHalf[2];
    const lift = Math.max(0, cube.position.z - restZ);
    const k = Math.max(0, 1 - lift / 0.22);
    cubeBlob.visible = k > 0.02;
    if (!cubeBlob.visible) return;
    cubeBlob.position.set(cube.position.x, cube.position.y, world.tableTopZ + 0.0015);
    const sc = 1 + lift * 2.4;
    cubeBlob.scale.set(sc, sc, 1);
    cubeBlobMat.opacity = 0.5 * k;
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
      currentFrame = { ...fa, t: currentTime, cube_pos: mix3(fa.cube_pos, fb.cube_pos, alpha), eef_pos: mix3(fa.eef_pos, fb.eef_pos, alpha) }; // [callouts hook]
      for (let i = 0; i < LINK_ORDER.length; i++) {
        const obj = linkObjects[i];
        if (!obj) continue;
        obj.visible = true;
        lerpPose(obj, fa.link_pos[i], fb.link_pos[i], fa.link_quat[i], fb.link_quat[i], alpha);
      }
      cube.visible = true;
      lerpPose(cube, fa.cube_pos, fb.cube_pos, fa.cube_quat, fb.cube_quat, alpha);
    }
    updateCubeShadow();

    // State label
    // Before the first segment starts, show the first state rather than a placeholder.
    const seq = analysis && analysis.sequence ? analysis.sequence : [];
    const st = stateAt(currentTime) || (seq.length ? seq[0] : null);
    const name = st ? st.name : '';
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
      cubeBlob.visible = false;
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
  // [callouts hook] screen projection and per-frame subscription for callouts.js
  const frameSubs = new Set();
  const projVec = new THREE.Vector3();
  function project(world) {
    projVec.set(world[0], world[1], world[2]).project(camera);
    const w = container.clientWidth || 1;
    const h = container.clientHeight || 1;
    return { x: (projVec.x * 0.5 + 0.5) * w, y: (projVec.y * -0.5 + 0.5) * h, visible: projVec.z > -1 && projVec.z < 1 };
  }
  function getFrame() { return currentFrame; }
  function onFrame(cb) { frameSubs.add(cb); return () => frameSubs.delete(cb); }

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
      const s = 1 + 0.05 * Math.sin(now * 0.0035);
      ring.scale.set(s, s, 1);
      ringMat.opacity = 0.6 + 0.18 * (0.5 + 0.5 * Math.sin(now * 0.0035));
    }
    controls.update();
    renderer.render(scene, camera);
    for (const cb of frameSubs) cb(currentTime, currentFrame); // [callouts hook]
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
    if (cube.geometry) cube.geometry.dispose();
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
    project,   // [callouts hook]
    getFrame,  // [callouts hook]
    onFrame,   // [callouts hook]
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
