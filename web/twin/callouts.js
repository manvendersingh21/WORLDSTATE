/**
 * WORLDSTATE — in-scene contextual callouts for the 3D digital twin.
 *
 * ES module exporting `mountCallouts(twin, container)`.
 * Draws graphite-glass panels anchored to 3D objects (cube, gripper, target pad)
 * with thin leader lines, driven entirely by the twin's per-frame data plus the
 * analysis payload. Nothing here is hard-coded per run: every number and step
 * time is measured from `frames` (cube_pos / eef_pos / gripper_closed) and the
 * analysis sequence.
 *
 * Requires three small additive hooks on the twin (see callouts.hook.md):
 *   twin.project(worldXYZ) -> { x, y, visible }   container pixel coords
 *   twin.getFrame()        -> current interpolated frame object
 *   twin.onFrame(cb)       -> unsubscribe; cb(t, frame) each render
 */

const PAD_WORLD = [0.0, 0.12, 0.801];

const ALIGN_XY = 0.015;   // "aligned above cube" tolerance (1.5 cm)
const LIFT_Z = 0.015;     // cube counts as lifted after a 1.5 cm rise
const CARRY_XY = 0.03;    // cube travelled 3 cm from the grasp pose
const PAD_REACH = 0.03;   // cube within 3 cm of the pad centre counts as reached

const MAX_VISIBLE = 3;
const STEP_HOLD = 2.1;    // seconds a step callout stays up
const FAIL_HOLD = 2.6;

// --------------------------------------------------------------------------- utils

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text != null) node.textContent = text;
  return node;
}

function injectStylesheet() {
  if (document.querySelector('link[data-callouts-css]')) return;
  const link = document.createElement('link');
  link.rel = 'stylesheet';
  link.href = new URL('./callouts.css', import.meta.url).href;
  link.setAttribute('data-callouts-css', '');
  document.head.appendChild(link);
}

const xyDist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
const secs = (t) => `${Number(t).toFixed(3)}s`;
const cmOf = (m) => (Math.abs(m) * 100).toFixed(1);
const prettyClass = (s) => String(s).replace(/[_-]+/g, ' ').trim();

// --------------------------------------------------------------------------- content

/**
 * Measure the run from its own frames. Returns indices/values only — no copy.
 */
function measure(frames, fps) {
  const z0 = frames[0].cube_pos[2];
  const cube0 = frames[0].cube_pos;

  let iAligned = -1, iGrasp = -1, iLift = -1, iCarry = -1, iPlaced = -1, iRelease = -1;
  let maxRise = 0, maxJump = 0, iJump = -1, padMin = Infinity, iPad = -1;

  for (let i = 0; i < frames.length; i++) {
    const f = frames[i];
    const rise = f.cube_pos[2] - z0;
    if (rise > maxRise) maxRise = rise;

    if (i > 0) {
      const j = xyDist(frames[i - 1].cube_pos, f.cube_pos);
      if (j > maxJump) { maxJump = j; iJump = i; }
    }
    const dPad = xyDist(f.cube_pos, PAD_WORLD);
    if (dPad < padMin) { padMin = dPad; iPad = i; }

    if (iAligned < 0 && !f.gripper_closed && xyDist(f.eef_pos, f.cube_pos) < ALIGN_XY) iAligned = i;
    if (iGrasp < 0 && f.gripper_closed) iGrasp = i;
    if (iGrasp >= 0 && i > iGrasp) {
      if (iLift < 0 && rise > LIFT_Z) iLift = i;
      if (iRelease < 0 && !f.gripper_closed) iRelease = i;
      if (iLift >= 0 && iCarry < 0 && xyDist(f.cube_pos, frames[iGrasp].cube_pos) > CARRY_XY) iCarry = i;
      if (iLift >= 0 && iPlaced < 0 && rise < 0.01 && !f.gripper_closed) iPlaced = i;
    }
  }

  const t = (i) => (i < 0 ? null : i / fps);
  return {
    z0, cube0, fps,
    tAligned: t(iAligned), tGrasp: t(iGrasp), tLift: t(iLift), tCarry: t(iCarry),
    tPlaced: t(iPlaced), tRelease: t(iRelease), tJump: t(iJump), tPadNearest: t(iPad),
    maxRise, maxJump, padMin,
    lifted: maxRise > LIFT_Z,
    padReached: padMin < PAD_REACH,
    duration: (frames.length - 1) / fps,
  };
}

function supportText(analysis) {
  const s = analysis && analysis.support;
  if (!s) return null;
  const k = s.k != null && s.k > 0 ? s.k : (s.expected_k != null ? s.expected_k : s.k);
  if (k == null || s.n == null) return null;
  return `${k}/${s.n}`;
}

/**
 * Build the callout timeline from the analysis payload + measured frames.
 */
function buildCallouts(analysis, m) {
  const out = [];
  const status = analysis && analysis.status ? analysis.status : 'normal';
  const failing = status === 'novel' || status === 'known_failure';
  const seq = Array.isArray(analysis && analysis.sequence) ? analysis.sequence : [];
  const stateAt = (t) => {
    let cur = null;
    for (const s of seq) { if (s.t <= t + 1e-9) cur = s; else break; }
    return cur ? cur.name : null;
  };
  const step = (n) => `STEP ${n}`;
  const add = (c) => { if (c.t != null) out.push(c); };

  let n = 0;

  // ----- shared approach / align steps (both normal and failing runs) -------
  add({
    id: 'approach', t: 0, hold: Math.max(STEP_HOLD, (m.tAligned ?? STEP_HOLD)),
    kind: 'step', anchor: 'gripper', status: 'ok', priority: 1,
    eyebrow: `${step(++n)} · ${secs(0)}`,
    title: 'Approaching cube',
    detail: stateAt(0) ? `state · ${stateAt(0)}` : null,
  });
  if (m.tAligned != null) {
    add({
      id: 'aligned', t: m.tAligned, hold: STEP_HOLD,
      kind: 'step', anchor: 'cube', status: 'ok', priority: 1,
      eyebrow: `${step(++n)} · ${secs(m.tAligned)}`,
      title: 'Aligned above cube',
      detail: `gripper open · within ${(ALIGN_XY * 100).toFixed(1)} cm of cube centre`,
    });
  }

  if (!failing) {
    // ----- nominal cycle ----------------------------------------------------
    if (m.tGrasp != null) {
      add({
        id: 'grasp', t: m.tGrasp, hold: STEP_HOLD,
        kind: 'step', anchor: 'gripper', status: 'ok', priority: 1,
        eyebrow: `${step(++n)} · ${secs(m.tGrasp)}`,
        title: 'Grasp: fingers closing \u2713',
        detail: stateAt(m.tGrasp) ? `state · ${stateAt(m.tGrasp)}` : null,
      });
    }
    if (m.tLift != null) {
      add({
        id: 'lift', t: m.tLift, hold: STEP_HOLD,
        kind: 'step', anchor: 'cube', status: 'ok', priority: 2,
        eyebrow: `${step(++n)} · ${secs(m.tLift)}`,
        title: `Lift: cube secured (+${cmOf(m.maxRise)} cm)`,
        detail: 'part clear of the surface',
      });
    }
    if (m.tCarry != null) {
      add({
        id: 'carry', t: m.tCarry, hold: Math.max(STEP_HOLD, (m.tPlaced != null ? m.tPlaced - m.tCarry : STEP_HOLD)),
        kind: 'step', anchor: 'cube', status: 'ok', priority: 1,
        eyebrow: `${step(++n)} · ${secs(m.tCarry)}`,
        title: 'Carrying to target',
        detail: stateAt(m.tCarry) ? `state · ${stateAt(m.tCarry)}` : null,
      });
    }
    if (m.tPlaced != null && m.padReached) {
      add({
        id: 'placed', t: m.tPlaced, hold: STEP_HOLD,
        kind: 'step', anchor: 'pad', status: 'ok', priority: 2,
        eyebrow: `${step(++n)} · ${secs(m.tPlaced)}`,
        title: 'Placed on target pad \u2713',
        detail: `${cmOf(m.padMin)} cm from pad centre`,
      });
    }
    const sup = supportText(analysis);
    const completeAt = m.tPlaced != null ? m.tPlaced + 0.45 : m.duration * 0.85;
    const n_ref = analysis && analysis.support ? analysis.support.n : null;
    add({
      id: 'complete', t: completeAt, hold: Infinity,
      kind: 'summary', anchor: 'scene', status: 'ok', priority: 4,
      eyebrow: 'CYCLE',
      title: sup ? `Cycle complete \u2014 matches learned process (${sup})` : 'Cycle complete \u2014 matches learned process',
      detail: n_ref != null
        ? `Consistent with ${n_ref} reference runs \u00b7 cube lifted +${cmOf(m.maxRise)} cm, placed ${cmOf(m.padMin)} cm from pad centre`
        : null,
    });
    return out;
  }

  // ----- divergence --------------------------------------------------------
  const statusKind = status === 'known_failure' ? 'known' : 'failure';
  const fds = analysis.first_divergence_s != null ? Number(analysis.first_divergence_s)
    : (m.tJump != null ? m.tJump : 0);

  add({
    id: 'displaced', t: fds, hold: FAIL_HOLD,
    kind: 'failure', anchor: 'cube', status: statusKind, priority: 5,
    eyebrow: `DIVERGENCE · ${secs(fds)}`,
    title: `Cube displaced ${cmOf(m.maxJump)} cm sideways`,
    detail: 'part left the taught pickup pose',
  });
  add({
    id: 'targeting', t: fds + 0.12, hold: FAIL_HOLD,
    kind: 'failure', anchor: 'gripper', status: statusKind, priority: 5,
    eyebrow: `DIVERGENCE · ${secs(fds)}`,
    title: 'Gripper still targeting original position',
    detail: stateAt(fds) ? `observed · ${stateAt(fds)}` : null,
  });

  if (m.tGrasp != null && !m.lifted) {
    add({
      id: 'empty', t: m.tGrasp, hold: FAIL_HOLD,
      kind: 'failure', anchor: 'gripper', status: statusKind, priority: 6,
      eyebrow: `GRASP · ${secs(m.tGrasp)}`,
      title: 'Closed on empty air \u2717 \u2014 cube not lifted',
      detail: `cube rise ${cmOf(m.maxRise)} cm`,
    });
  }
  if (!m.padReached) {
    const padAt = m.tGrasp != null ? m.tGrasp + FAIL_HOLD * 0.55 : fds + 1.5;
    add({
      id: 'pad', t: padAt, hold: Infinity,
      kind: 'failure', anchor: 'pad', status: statusKind, priority: 6,
      eyebrow: 'TARGET',
      title: 'Target never reached',
      detail: `cube stayed ${cmOf(m.padMin)} cm away`,
    });
  }

  const exp = (analysis.expected || {}).to || '\u2014';
  const obs = (analysis.observed || {}).to || '\u2014';
  const sup = analysis.support || {};
  add({
    id: 'summary', t: fds + 0.3, hold: Infinity,
    kind: 'summary', anchor: 'scene', status: statusKind, priority: 7,
    eyebrow: status === 'known_failure' ? 'KNOWN FAILURE' : 'UNSEEN PATTERN',
    title: `Expected ${exp} \u00b7 Observed ${obs}`,
    detail: sup.n != null ? `Seen in ${sup.k != null ? sup.k : 0}/${sup.n} runs` : null,
  });
  if (status === 'known_failure' && analysis.known_class) {
    add({
      id: 'memory', t: fds + 0.5, hold: Infinity,
      kind: 'memory', anchor: 'scene', status: 'known', priority: 6,
      eyebrow: 'MEMORY',
      title: `Recognized from memory: ${prettyClass(analysis.known_class)}`,
      detail: analysis.known_distance != null ? `distance ${analysis.known_distance}` : null,
    });
  }
  return out;
}

// --------------------------------------------------------------------------- layout

// Where a card sits relative to its anchor, in pixels, so it never covers the
// cube or the gripper.
const OFFSETS = {
  gripper: [-168, -104],
  cube: [172, 92],
  pad: [186, 34],
};

function cardRect(card) {
  return { w: card.offsetWidth || 240, h: card.offsetHeight || 72 };
}

/** Point where the segment card-centre -> anchor exits the card box. */
function edgePoint(cx, cy, w, h, ax, ay) {
  const dx = ax - cx;
  const dy = ay - cy;
  if (dx === 0 && dy === 0) return [cx, cy];
  const hw = w / 2 + 4;
  const hh = h / 2 + 4;
  const sx = dx === 0 ? Infinity : hw / Math.abs(dx);
  const sy = dy === 0 ? Infinity : hh / Math.abs(dy);
  const s = Math.min(sx, sy, 1);
  return [cx + dx * s, cy + dy * s];
}

// --------------------------------------------------------------------------- mount

export function mountCallouts(twin, container) {
  injectStylesheet();

  if (getComputedStyle(container).position === 'static') container.style.position = 'relative';

  const layer = el('div', 'co-layer');
  layer.setAttribute('data-callouts', '');
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('class', 'co-leaders');
  layer.appendChild(svg);
  container.appendChild(layer);

  let analysis = null;
  let frames = null;
  let runId = null;
  let measured = null;
  let callouts = [];
  const nodes = new Map();   // id -> { card, title, eyebrow, detail, leader, dot }
  let disposed = false;

  function clearNodes() {
    for (const n of nodes.values()) { n.card.remove(); n.leader.remove(); n.dot.remove(); }
    nodes.clear();
  }

  function nodeFor(c) {
    let n = nodes.get(c.id);
    if (n) return n;
    const card = el('div', 'co-card');
    card.setAttribute('data-kind', c.kind);
    card.setAttribute('data-anchor', c.anchor);
    card.setAttribute('data-status', c.status);
    card.setAttribute('data-id', c.id);
    card.setAttribute('data-visible', 'false');
    const eyebrow = el('div', 'co-eyebrow', c.eyebrow || '');
    const title = el('div', 'co-title', c.title || '');
    card.appendChild(eyebrow);
    card.appendChild(title);
    let detail = null;
    if (c.detail) { detail = el('div', 'co-detail', c.detail); card.appendChild(detail); }
    layer.appendChild(card);

    const leader = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    leader.setAttribute('class', 'co-leader');
    leader.setAttribute('data-id', c.id);
    leader.setAttribute('data-status', c.status);
    const dot = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    dot.setAttribute('class', 'co-anchor-dot');
    dot.setAttribute('r', '2.6');
    svg.appendChild(leader);
    svg.appendChild(dot);

    n = { card, eyebrow, title, detail, leader, dot };
    nodes.set(c.id, n);
    return n;
  }

  function rebuild() {
    clearNodes();
    callouts = (analysis && frames && frames.length) ? buildCallouts(analysis, measured) : [];
    for (const c of callouts) nodeFor(c);
  }

  function anchorPoint(c, frame) {
    if (c.anchor === 'scene') return null;
    let world = null;
    if (c.anchor === 'cube') world = frame && frame.cube_pos;
    else if (c.anchor === 'gripper') world = frame && frame.eef_pos;
    else if (c.anchor === 'pad') world = PAD_WORLD;
    if (!world || typeof twin.project !== 'function') return null;
    const p = twin.project(world);
    if (!p || !p.visible) return null;
    return p;
  }

  function render(t, frame) {
    if (disposed || !callouts.length) return;
    const W = container.clientWidth || 1;
    const H = container.clientHeight || 1;
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);

    const active = callouts
      .filter((c) => t >= c.t - 1e-6 && t < c.t + c.hold)
      .sort((a, b) => (b.priority - a.priority) || (b.t - a.t))
      .slice(0, MAX_VISIBLE);
    const activeIds = new Set(active.map((c) => c.id));

    // Scene-anchored summary cards stack at the bottom-left of the view.
    let sceneSlot = 0;
    const placed = [];

    for (const c of active) {
      const n = nodeFor(c);
      const { w, h } = cardRect(n.card);
      let cx, cy, ap = null;

      if (c.anchor === 'scene') {
        cx = 18 + w / 2;
        cy = H - 46 - h / 2 - sceneSlot * (h + 10);
        sceneSlot += 1;
      } else {
        ap = anchorPoint(c, frame);
        if (!ap) { n.card.setAttribute('data-visible', 'false'); n.leader.setAttribute('d', ''); n.dot.setAttribute('r', '0'); continue; }
        const [ox, oy] = OFFSETS[c.anchor] || [150, -80];
        cx = ap.x + ox;
        cy = ap.y + oy;
        // Flip the offset if it would leave the viewport.
        if (cx + w / 2 > W - 12) cx = ap.x - Math.abs(ox);
        if (cx - w / 2 < 12) cx = ap.x + Math.abs(ox);
        if (cy - h / 2 < 12) cy = ap.y + Math.abs(oy);
        if (cy + h / 2 > H - 12) cy = ap.y - Math.abs(oy);
        cx = Math.min(Math.max(cx, w / 2 + 12), W - w / 2 - 12);
        cy = Math.min(Math.max(cy, h / 2 + 12), H - h / 2 - 12);

        // Keep cards from stacking on top of each other.
        for (const p of placed) {
          if (Math.abs(cx - p.cx) < (w + p.w) / 2 - 8 && Math.abs(cy - p.cy) < (h + p.h) / 2 + 8) {
            cy = p.cy + (cy >= p.cy ? 1 : -1) * ((h + p.h) / 2 + 10);
            cy = Math.min(Math.max(cy, h / 2 + 12), H - h / 2 - 12);
          }
        }
      }

      placed.push({ cx, cy, w, h });
      n.card.style.transform = `translate(${Math.round(cx - w / 2)}px, ${Math.round(cy - h / 2)}px)`;
      n.card.setAttribute('data-visible', 'true');

      if (ap) {
        const [ex, ey] = edgePoint(cx, cy, w, h, ap.x, ap.y);
        // Thin elbow: short horizontal run off the card, then straight to the anchor.
        const kx = ex + Math.sign(ap.x - cx) * 14;
        n.leader.setAttribute('d', `M ${ex.toFixed(1)} ${ey.toFixed(1)} L ${kx.toFixed(1)} ${ey.toFixed(1)} L ${ap.x.toFixed(1)} ${ap.y.toFixed(1)}`);
        n.dot.setAttribute('cx', ap.x.toFixed(1));
        n.dot.setAttribute('cy', ap.y.toFixed(1));
        n.dot.setAttribute('r', '2.6');
      } else {
        n.leader.setAttribute('d', '');
        n.dot.setAttribute('r', '0');
      }
    }

    for (const [id, n] of nodes) {
      if (!activeIds.has(id)) {
        n.card.setAttribute('data-visible', 'false');
        n.leader.setAttribute('d', '');
        n.dot.setAttribute('r', '0');
      }
    }
  }

  const unsubscribe = typeof twin.onFrame === 'function'
    ? twin.onFrame((t, frame) => render(t, frame || (typeof twin.getFrame === 'function' ? twin.getFrame() : null)))
    : () => {};

  return {
    setAnalysis(a) {
      analysis = a || null;
      rebuild();
      render(typeof twin.getTime === 'function' ? twin.getTime() : 0,
        typeof twin.getFrame === 'function' ? twin.getFrame() : null);
    },
    setRun(id, runFrames) {
      runId = id;
      frames = Array.isArray(runFrames) && runFrames.length ? runFrames : null;
      measured = frames ? measure(frames, frames.length > 1 ? Math.round(1 / (frames[1].t - frames[0].t)) : 20) : null;
      rebuild();
    },
    dispose() {
      if (disposed) return;
      disposed = true;
      unsubscribe();
      clearNodes();
      layer.remove();
    },
    // Non-contractual helpers for the demo page / tests.
    getCallouts: () => callouts.map((c) => ({ ...c })),
    getMeasurements: () => (measured ? { ...measured, runId } : null),
    element: layer,
  };
}

export default mountCallouts;
