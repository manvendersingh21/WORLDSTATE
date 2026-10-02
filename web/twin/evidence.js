/**
 * WORLDSTATE live evidence module.
 *
 * Pulls real YOLO detections, divergence analysis, and Cosmos Reason
 * narration from the WORLDSTATE API so 3D callouts can show live values.
 *
 * Exports:
 *   loadEvidence(runId, {base=''}) -> Promise<Evidence>
 *   formatTag(obj, label)          -> "YOLO · cube 0.97"
 *
 * Evidence = {
 *   atTime(t) -> {gripper:{conf,cx,cy}|null, part:{conf,cx,cy}|null,
 *                 fixture:{conf}|null, frame},
 *   analysis,                       // raw /analysis payload
 *   cosmos: {source, model, sentences, failureSentence},
 *   summary: {meanConf:{gripper,part,fixture},
 *             detectionRate:{gripper,part,fixture}},
 * }
 */

const LABELS = ["gripper", "part", "fixture"];
const FAILURE_RE = /fail|shift|miss|empty|not/i;

async function fetchJson(url) {
  const res = await fetch(url, { headers: { Accept: "application/json" } });
  if (!res.ok) {
    throw new Error(`GET ${url} failed: ${res.status}`);
  }
  return res.json();
}

/** Split a narrative into trimmed, nonempty sentences. */
function splitSentences(text) {
  if (!text) return [];
  return String(text)
    .split(/(?<=[.!?])\s+/)
    .map((s) => s.trim())
    .filter(Boolean);
}

/** Highest-confidence object with the given label in one frame, or null. */
function bestObject(frame, label) {
  let best = null;
  for (const obj of frame.objects || []) {
    if (obj.label !== label) continue;
    if (!best || obj.conf > best.conf) best = obj;
  }
  return best;
}

/** formatTag({conf:0.973}, 'cube') -> "YOLO · cube 0.97". '' for null obj. */
export function formatTag(obj, label) {
  if (!obj || typeof obj.conf !== "number") return "";
  return `YOLO · ${label} ${obj.conf.toFixed(2)}`;
}

export async function loadEvidence(runId, { base = "" } = {}) {
  const root = `${base}/api/episodes/${encodeURIComponent(runId)}`;
  const [episode, analysis, narration] = await Promise.all([
    fetchJson(root),
    fetchJson(`${root}/analysis`),
    fetchJson(`${root}/narration`),
  ]);

  const frames = episode.frames || [];

  // Per-label mean confidence and detection rate over all frames.
  const meanConf = {};
  const detectionRate = {};
  for (const label of LABELS) {
    let sum = 0;
    let hits = 0;
    for (const frame of frames) {
      const obj = bestObject(frame, label);
      if (obj) {
        sum += obj.conf;
        hits += 1;
      }
    }
    meanConf[label] = hits ? sum / hits : 0;
    detectionRate[label] = frames.length ? hits / frames.length : 0;
  }

  const sentences = splitSentences(narration.narrative);
  const failureSentence = sentences.find((s) => FAILURE_RE.test(s)) || null;

  function atTime(t) {
    if (!frames.length) {
      return { gripper: null, part: null, fixture: null, frame: null };
    }
    let best = frames[0];
    let bestDist = Math.abs((frames[0].time || 0) - t);
    for (const frame of frames) {
      const dist = Math.abs((frame.time || 0) - t);
      if (dist < bestDist) {
        best = frame;
        bestDist = dist;
      }
    }
    const gripper = bestObject(best, "gripper");
    const part = bestObject(best, "part");
    const fixture = bestObject(best, "fixture");
    return {
      gripper: gripper ? { conf: gripper.conf, cx: gripper.cx, cy: gripper.cy } : null,
      part: part ? { conf: part.conf, cx: part.cx, cy: part.cy } : null,
      fixture: fixture ? { conf: fixture.conf } : null,
      frame: best.frame,
    };
  }

  return {
    atTime,
    analysis,
    cosmos: {
      source: narration.source,
      model: narration.model,
      sentences,
      failureSentence,
    },
    summary: { meanConf, detectionRate },
    fps: episode.fps,
    duration: episode.duration,
  };
}
