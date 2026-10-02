const state = {
  model: null,
  episodes: [],
  current: null,
  analysis: null,
  health: null,
  narration: null,
  narrationExpanded: false,
  justRemembered: false,
};

const $ = (id) => document.getElementById(id);

async function api(path, opts) {
  const response = await fetch(path, opts);
  if (!response.ok) throw new Error((await response.text()) || response.statusText);
  const type = response.headers.get("content-type") || "";
  if (type.includes("application/json")) return response.json();
  return response;
}

/* ---------- labels ---------- */

const SHORT_NODE_NAMES = {
  "At pickup": "Pickup",
  "Grasping part": "Grasp",
  "Lifting part": "Lift",
  "Carrying part": "Carry",
  "Descending": "Descend",
  "Placing part": "Place",
  "Seated in fixture": "Seated",
};

function titleCase(text) {
  return String(text)
    .replaceAll("_", " ")
    .replace(/\b\w/g, (ch) => ch.toUpperCase());
}

function shortNodeName(name) {
  return SHORT_NODE_NAMES[name] || titleCase(name);
}

function runLabel(ep) {
  if (ep.id === "miss_unseen") return "Unseen failure";
  if (ep.id === "miss_eval_left") return "Failure (left)";
  if (ep.id === "miss_eval_right") return "Failure (right)";
  const suffix = (ep.id.match(/(\d+)$/) || [])[1];
  if (ep.role === "eval_failure") return suffix ? `Failure ${suffix}` : "Failure";
  if (ep.role === "heldout_normal") return suffix ? `Normal run (${suffix})` : "Normal run";
  if (ep.role === "upload" || ep.role === "similar_hidden") return suffix ? `Uploaded run (${suffix.slice(-4)})` : "Uploaded run";
  if (ep.role === "reference") return suffix ? `Reference ${suffix}` : "Reference";
  return ep.id;
}

/* ---------- boot ---------- */

async function boot() {
  state.health = await api("/api/health");
  state.model = await api("/api/model");
  state.episodes = await api("/api/episodes");
  renderHeader();
  renderRunSelect();
  renderGraph();
  const first =
    state.episodes.find((ep) => ep.id === "miss_unseen") ||
    state.episodes.find((ep) => ep.role === "unseen_failure") ||
    state.episodes[0];
  if (first) await selectEpisode(first.id);
}

function renderHeader() {
  const health = state.health || {};
  const adapters = health.adapters || {};
  const cosmos = health.cosmos || {};
  const memory = health.memory || {};
  $("sys-live").classList.toggle("on", Boolean(health.ok));
  $("sys-yolo").classList.toggle("on", adapters.perception === "yolo" || Boolean(adapters.yolo_installed));
  $("sys-cosmos").classList.toggle("on", Boolean(cosmos.available || cosmos.succeeded));
  $("sys-memory").classList.toggle("on", Boolean(health.ok && (memory.embedder || memory.vector_index || Object.keys(memory).length)));
  const version = state.model?.version ?? health.version ?? 0;
  $("version-chip").textContent = `WORLD MODEL v${version}`;
}

function renderRunSelect() {
  const select = $("run-select");
  select.innerHTML = "";
  const groups = [
    ["Failures", (ep) => ep.role === "unseen_failure" || ep.role === "eval_failure"],
    ["Normal runs", (ep) => ep.role === "heldout_normal"],
    ["Uploads", (ep) => ep.role === "upload" || ep.role === "similar_hidden"],
    ["Reference runs", (ep) => ep.role === "reference"],
  ];
  const seen = new Set();
  for (const [label, match] of groups) {
    const members = state.episodes.filter((ep) => !seen.has(ep.id) && match(ep));
    members.forEach((ep) => seen.add(ep.id));
    if (!members.length) continue;
    const group = document.createElement("optgroup");
    group.label = label;
    for (const ep of members.sort((a, b) => a.id.localeCompare(b.id))) {
      const option = document.createElement("option");
      option.value = ep.id;
      option.textContent = runLabel(ep);
      group.append(option);
    }
    select.append(group);
  }
  if (state.current) select.value = state.current;
}

/* ---------- selection + analysis ---------- */

async function selectEpisode(id) {
  state.current = id;
  state.justRemembered = false;
  state.narration = null;
  state.narrationExpanded = false;
  if ($("run-select").value !== id) $("run-select").value = id;
  const detail = await api(`/api/episodes/${id}`);
  const video = $("video");
  video.src = `/api/episodes/${id}/video`;
  video.dataset.frames = JSON.stringify(detail.frames || []);
  // Loop and resume playback so the run stays alive on screen (and keeps
  // driving the twin clock). Autoplay may be blocked unmuted; that's fine.
  video.loop = true;
  video.play().catch(() => {});
  state.analysis = await api(`/api/episodes/${id}/analysis`);
  renderAnalysis();
  seekToStory();
  drawOverlay();
  if (twinState.instance) syncTwinRun().catch(() => {});
}

const STATUS_WORDS = { normal: "NORMAL", novel: "NOVEL FAILURE", known_failure: "KNOWN FAILURE" };

function headlineFor(analysis) {
  if (analysis.status === "normal") return "MATCHES LEARNED PROCESS";
  const hint = `${analysis.known_class || ""} ${analysis.observed?.to || ""}`.toLowerCase();
  if (/slid|displac|shift/.test(hint)) return "OBJECT DISPLACED AFTER ALIGNMENT";
  return "PROCESS LEFT THE LEARNED PATH";
}

function renderAnalysis() {
  const analysis = state.analysis;
  if (!analysis) return;
  const status = analysis.status;
  const badge = $("status-badge");
  const word = $("status-word");
  badge.dataset.status = status;
  badge.textContent = STATUS_WORDS[status] || status;
  word.dataset.status = status;
  word.textContent = STATUS_WORDS[status] || status;
  $("event-headline").textContent = headlineFor(analysis);

  const divergenceBlock = $("divergence-block");
  if (status !== "normal" && analysis.first_divergence_s != null) {
    divergenceBlock.classList.remove("hidden");
    divergenceBlock.classList.toggle("known", status === "known_failure");
    $("divergence-time").textContent = `${Number(analysis.first_divergence_s).toFixed(3)}s`;
  } else {
    divergenceBlock.classList.add("hidden");
  }

  const normalScore = $("normal-score");
  if (status === "normal") {
    normalScore.classList.remove("hidden");
    normalScore.textContent = `Consistent with ${analysis.support?.n ?? state.model?.n_reference ?? ""} reference runs · score ${analysis.score.toFixed(2)}`;
  } else {
    normalScore.classList.add("hidden");
  }

  $("field-expected").textContent = analysis.expected?.to || "—";
  $("field-observed").textContent = analysis.observed?.to || "—";
  $("field-support").textContent = analysis.support ? `${analysis.support.k} / ${analysis.support.n}` : "—";

  const remember = $("remember-btn");
  remember.classList.toggle("hidden", status !== "novel");
  remember.disabled = status !== "novel";
  remember.textContent = "Remember this failure";
  $("remember-confirm").classList.toggle("hidden", !state.justRemembered);

  renderDrawer(analysis);
  loadNarration(analysis.episode_id);
  renderGraph();
  renderTimeline(analysis);
}

function renderDrawer(analysis) {
  $("drawer-why").textContent = analysis.why || "";
  $("drawer-expected-path").textContent = (analysis.expected_path || []).map((step) => step.name).join(" → ");
  $("drawer-recovery-path").textContent = (analysis.recovery_path || []).map((step) => step.name).join(" → ");
  $("drawer-scores").textContent = [
    `score ${analysis.score?.toFixed(3)}`,
    `threshold ${analysis.threshold?.toFixed(3)}`,
    `P(expected) ${analysis.expected?.p?.toFixed(2)}`,
    analysis.support?.text,
  ].filter(Boolean).join(" · ");
  $("drawer-internal").textContent = [
    `run ${analysis.episode_id}`,
    `model v${analysis.version}`,
    analysis.known_class ? `class ${analysis.known_class}` : null,
  ].filter(Boolean).join(" · ");
  const adapters = state.health?.adapters || {};
  const memory = state.health?.memory || {};
  $("drawer-adapters").textContent = [
    `Perception ${adapters.perception || "classical"}`,
    `Embedder ${memory.embedder || adapters.text_embedder || "hashing"}`,
    `Index ${memory.vector_index || adapters.vector_index || "numpy"}`,
  ].join(" · ");
}

/* ---------- narration ---------- */

function splitSentences(text) {
  return (String(text || "").match(/[^.!?]+[.!?]+/g) || [String(text || "")]).map((s) => s.trim()).filter(Boolean);
}

async function loadNarration(episodeId) {
  const text = $("cosmos-text");
  const attribution = $("cosmos-attribution");
  const expand = $("cosmos-expand");
  text.textContent = "Analyzing with the event model…";
  attribution.textContent = "";
  attribution.classList.remove("cosmos");
  expand.classList.add("hidden");
  try {
    const data = await api(`/api/episodes/${encodeURIComponent(episodeId)}/narration`);
    if (state.current && state.current !== episodeId) return;
    const sentences = splitSentences(data.narrative);
    state.narration = {
      excerpt: sentences.slice(0, 3).join(" "),
      full: sentences.join(" "),
      hasMore: sentences.length > 3,
      source: data.source,
    };
    state.narrationExpanded = false;
    renderNarration();
  } catch (error) {
    text.textContent = "Event narration unavailable.";
  }
}

function renderNarration() {
  const narration = state.narration;
  if (!narration) return;
  const text = $("cosmos-text");
  const attribution = $("cosmos-attribution");
  const expand = $("cosmos-expand");
  text.textContent = state.narrationExpanded ? narration.full : narration.excerpt;
  text.classList.toggle("clamp", !state.narrationExpanded);
  expand.classList.toggle("hidden", !narration.hasMore);
  expand.textContent = state.narrationExpanded ? "Show less" : "Show full";
  if (narration.source === "cosmos-reason") {
    attribution.textContent = "NVIDIA Cosmos Reason";
    attribution.classList.add("cosmos");
  } else {
    attribution.textContent = "Kinematic summary";
    attribution.classList.remove("cosmos");
  }
}

$("cosmos-expand").addEventListener("click", () => {
  state.narrationExpanded = !state.narrationExpanded;
  renderNarration();
});

/* ---------- state graph ---------- */

function escapeXml(value) {
  return String(value).replace(/[&<>]/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[ch]));
}

function renderGraph() {
  const svg = $("graph");
  const model = state.model;
  if (!model?.nodes) {
    svg.innerHTML = "";
    svg.dataset.branch = "none";
    return;
  }
  const analysis = state.analysis;
  const nominal = model.nodes.filter((node) => node.kind !== "failure");
  const failures = model.nodes.filter((node) => node.kind === "failure");
  const width = 1360;
  const yMain = 62;
  const yBranch = 150;
  const rMain = 30;
  const rBranch = 24;
  const margin = 64;
  const xOf = {};
  nominal.forEach((node, index) => {
    xOf[node.id] = margin + index * ((width - margin * 2) / Math.max(nominal.length - 1, 1));
  });

  const isNormal = analysis?.status === "normal";
  const divergeFrom = analysis && analysis.status !== "normal"
    ? nominal.find((node) => node.name === analysis.expected?.from)
    : null;
  const activeId = divergeFrom?.id || null;

  let markup = "";

  // learned path edges (muted slate; green when this run traversed them normally)
  for (const edge of model.edges || []) {
    if (edge.kind !== "nominal" || xOf[edge.src] == null || xOf[edge.dst] == null) continue;
    markup += `<line class="edge-nominal${isNormal ? " traversed" : ""}" x1="${xOf[edge.src] + rMain}" y1="${yMain}" x2="${xOf[edge.dst] - rMain}" y2="${yMain}"/>`;
  }

  // branch geometry: leave the main circle at ~45° and curve away so the
  // path never strikes through node labels (labels sit above the nodes).
  const branchStart = (srcX) => `M ${srcX + rMain * 0.7} ${yMain + rMain * 0.7} Q ${srcX + 42} ${yBranch}, `;

  // known failure branches (blue)
  for (const node of failures) {
    const inEdge = (model.edges || []).find((edge) => edge.dst === node.id && edge.kind === "failure");
    const srcX = inEdge && xOf[inEdge.src] != null ? xOf[inEdge.src] : width / 2;
    const x = srcX + 150;
    xOf[node.id] = x;
    markup += `<path class="branch-known" d="${branchStart(srcX)}${x - rBranch - 4} ${yBranch}"/>`;
    markup += `<circle class="branch-known" cx="${x}" cy="${yBranch}" r="${rBranch}"/>`;
    markup += `<text class="node-label branch-known" x="${x + rBranch + 10}" y="${yBranch + 5}" text-anchor="start">${escapeXml(shortNodeName(node.name))}</text>`;
    const recovery = (model.edges || []).find((edge) => edge.src === node.id && edge.kind === "recovery");
    if (recovery && xOf[recovery.dst] != null) {
      markup += `<path class="branch-known edge-recovery" d="M ${x + rBranch} ${yBranch - 6} Q ${(x + xOf[recovery.dst]) / 2} ${yBranch - 10}, ${xOf[recovery.dst]} ${yMain + rMain + 4}"/>`;
    }
  }

  // live novel branch (red)
  if (analysis?.status === "novel" && divergeFrom) {
    const srcX = xOf[divergeFrom.id];
    const x = srcX + 150;
    markup += `<path class="branch-novel pulse" d="${branchStart(srcX)}${x - rBranch - 4} ${yBranch}"/>`;
    markup += `<circle class="branch-novel pulse" cx="${x}" cy="${yBranch}" r="${rBranch}"/>`;
    markup += `<text class="node-label branch-novel" x="${x + rBranch + 10}" y="${yBranch + 5}" text-anchor="start">${escapeXml(analysis.observed?.to || "Novel state")}</text>`;
  }

  // nominal nodes (labels above the circles, clear of branch paths)
  for (const node of nominal) {
    const x = xOf[node.id];
    const active = node.id === activeId;
    markup += `<g class="node-nominal${active ? " active" : ""}${isNormal ? " traversed" : ""}">`;
    markup += `<circle cx="${x}" cy="${yMain}" r="${rMain}"${active ? ' class="pulse"' : ""}/>`;
    markup += `<text class="node-label" x="${x}" y="${yMain - rMain - 12}">${escapeXml(shortNodeName(node.name))}</text>`;
    markup += `</g>`;
  }

  svg.innerHTML = markup;
  svg.dataset.branch = analysis?.status === "novel" ? "novel" : analysis?.status === "known_failure" ? "known" : "none";
  $("graph-meta").textContent = `${nominal.length} learned states · ${model.n_reference} reference runs`;
}

/* ---------- video, overlay, timeline ---------- */

function renderTimeline(analysis) {
  const bar = $("timeline");
  bar.innerHTML = "";
  if (!analysis || analysis.first_divergence_s == null) return;
  const video = $("video");
  const duration = video.duration || 8;
  const position = `${(analysis.first_divergence_s / duration) * 100}%`;
  const mark = document.createElement("i");
  mark.style.left = position;
  const tickLabel = document.createElement("span");
  tickLabel.className = "tick-label";
  tickLabel.style.left = position;
  tickLabel.textContent = `${Number(analysis.first_divergence_s).toFixed(3)}s`;
  bar.append(mark, tickLabel);
  bar.onclick = (event) => {
    const rect = bar.getBoundingClientRect();
    video.currentTime = ((event.clientX - rect.left) / rect.width) * (video.duration || duration);
  };
}

function drawOverlay() {
  const video = $("video");
  const canvas = $("overlay");
  if (!$("tracking-toggle").checked) return;
  const frames = JSON.parse(video.dataset.frames || "[]");
  if (!frames.length || !video.videoWidth) return;
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  let frame = frames[0];
  for (const candidate of frames) {
    if (candidate.time <= video.currentTime) frame = candidate;
  }
  const colors = { gripper: "#FF8276", part: "#6FB3FF", fixture: "#8FD77A" };
  ctx.lineWidth = 2;
  ctx.font = "12px Inter, sans-serif";
  ctx.globalAlpha = 0.85;
  for (const obj of frame.objects || []) {
    const x = (obj.cx - obj.w / 2) * canvas.width;
    const y = (obj.cy - obj.h / 2) * canvas.height;
    ctx.strokeStyle = colors[obj.label] || "#fff";
    ctx.strokeRect(x, y, obj.w * canvas.width, obj.h * canvas.height);
    const labelWidth = ctx.measureText(obj.label).width;
    const baseline = Math.max(12, y - 4);
    ctx.fillStyle = "rgba(6, 8, 11, 0.85)";
    ctx.fillRect(x, baseline - 11, labelWidth + 8, 15);
    ctx.fillStyle = ctx.strokeStyle;
    ctx.fillText(obj.label, x + 4, baseline);
  }
  ctx.globalAlpha = 1;
}

$("video").addEventListener("timeupdate", drawOverlay);

$("tracking-toggle").addEventListener("change", (event) => {
  const canvas = $("overlay");
  canvas.style.display = event.target.checked ? "" : "none";
  if (event.target.checked) drawOverlay();
});

function seekToStory() {
  const video = $("video");
  const analysis = state.analysis;
  if (!video || !analysis || !Number.isFinite(video.duration)) return;
  if (analysis.first_divergence_s == null || analysis.status === "normal") {
    video.currentTime = analysis.status === "normal" ? Math.min(6.4, video.duration - 0.2) : 0;
  } else {
    video.currentTime = Math.min(video.duration - 0.25, analysis.first_divergence_s + 1.15);
  }
  renderTimeline(analysis);
}

$("video").addEventListener("loadedmetadata", seekToStory);

/* ---------- actions ---------- */

$("run-select").addEventListener("change", (event) => selectEpisode(event.target.value));

$("remember-btn").addEventListener("click", async () => {
  if (!state.current) return;
  const button = $("remember-btn");
  button.disabled = true;
  button.textContent = "Remembering…";
  try {
    const payload = await api(`/api/episodes/${state.current}/remember`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ label: "displaced_after_align" }),
    });
    state.model = payload.model;
    state.analysis = payload.analysis;
    state.health = await api("/api/health");
    state.justRemembered = true;
    renderHeader();
    renderAnalysis();
    for (const id of ["status-badge", "status-word"]) {
      const el = $(id);
      el.classList.remove("flip");
      void el.offsetWidth;
      el.classList.add("flip");
    }
  } catch (error) {
    button.disabled = false;
    button.textContent = "Remember this failure";
  }
});

/* ---------- search ---------- */

function humanizeHit(hit) {
  const snippet = hit.snippet || "";
  const lower = snippet.toLowerCase();
  const diverge = snippet.match(/anomaly at ([\d.]+)s/);
  let finding;
  let isNormal = false;
  const subs = [];
  if (lower.includes("normal run")) {
    finding = "Placed on target";
    isNormal = true;
  } else if (lower.includes("displaced") || /displaced/.test(String(hit.kind || ""))) {
    finding = "Displaced after alignment";
  } else if (lower.includes("failed grasp") || lower.includes("novel transition") || String(hit.kind || "").startsWith("miss")) {
    finding = "Failed grasp";
  } else if (hit.next_state) {
    finding = `Next ${hit.next_state}`;
  } else {
    finding = titleCase(hit.kind || hit.role || "Run");
  }
  const episode = state.episodes.find((ep) => ep.id === hit.id);
  const title = episode ? `${runLabel(episode)} — ${finding}` : finding;
  if (isNormal) subs.push("Matches learned process");
  if (diverge) subs.push(`${Number(diverge[1]).toFixed(3)}s divergence`);
  if (finding === "Failed grasp" && lower.includes("displaced")) subs.push("Displaced after alignment");
  else if (finding === "Failed grasp" && lower.includes("cube moved unexpectedly")) subs.push("Cube moved unexpectedly");
  return { title, sub: subs.join(" · ") };
}

function renderHits(hits, note) {
  const host = $("search-results");
  host.innerHTML = "";
  if (note) {
    const head = document.createElement("p");
    head.className = "result-note";
    head.textContent = note;
    host.append(head);
  }
  for (const hit of hits) {
    const card = document.createElement("article");
    card.className = "result-card";
    card.dataset.episode = hit.id;
    const main = document.createElement("div");
    main.className = "result-main";
    const { title, sub } = humanizeHit(hit);
    const titleEl = document.createElement("div");
    titleEl.className = "result-title";
    titleEl.textContent = title;
    main.append(titleEl);
    if (sub) {
      const subEl = document.createElement("div");
      subEl.className = "result-sub";
      subEl.textContent = sub;
      main.append(subEl);
    }
    const idEl = document.createElement("span");
    idEl.className = "result-id";
    idEl.textContent = hit.id;
    card.append(main, idEl);
    if (state.episodes.some((ep) => ep.id === hit.id)) {
      card.addEventListener("click", () => {
        selectEpisode(hit.id);
        window.scrollTo({ top: 0, behavior: "smooth" });
      });
    }
    host.append(card);
  }
  if (!hits.length) {
    const empty = document.createElement("p");
    empty.className = "result-note";
    empty.textContent = "No matching runs in memory.";
    host.append(empty);
  }
}

async function askMemory(query) {
  $("search-input").value = query;
  const hits = await api(`/api/search?q=${encodeURIComponent(query)}&k=5`);
  renderHits(hits, `World memory results for “${query}”`);
}

$("search-btn").addEventListener("click", () => {
  const typed = $("search-input").value.trim();
  askMemory(typed || "what unusual things happened");
});

$("search-input").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && event.target.value.trim()) askMemory(event.target.value.trim());
});

document.querySelectorAll(".search-chip").forEach((chip) => {
  chip.addEventListener("click", () => askMemory(chip.dataset.q));
});

/* ---------- uploads ---------- */

$("upload-similar-btn").addEventListener("click", async () => {
  const button = $("upload-similar-btn");
  button.disabled = true;
  button.textContent = "Uploading…";
  try {
    const blob = await (await fetch("/api/demo/similar-file")).blob();
    const body = new FormData();
    body.append("file", blob, "similar_run.mp4");
    const payload = await api("/api/upload", { method: "POST", body });
    state.episodes = await api("/api/episodes");
    renderRunSelect();
    await selectEpisode(payload.episode.id);
  } finally {
    button.disabled = false;
    button.textContent = "Upload similar run";
  }
});

$("file-input").addEventListener("change", async (event) => {
  const file = event.target.files?.[0];
  if (!file) return;
  const body = new FormData();
  body.append("file", file, file.name);
  const payload = await api("/api/upload", { method: "POST", body });
  state.episodes = await api("/api/episodes");
  renderRunSelect();
  await selectEpisode(payload.episode.id);
});

/* ---------- 3D digital twin view ---------- */

const twinState = { instance: null, loadedRun: null, raf: 0, view: "camera" };

async function ensureTwin() {
  if (twinState.instance) return twinState.instance;
  // Lazy: the twin module (and its vendored three.js + GLBs) is only fetched
  // the first time the 3D DIGITAL TWIN toggle is clicked.
  const mod = await import("/static/twin/twin.js");
  twinState.instance = mod.mountTwin($("twin-mount"), {
    assetsBase: "/static/twin/assets",
    dataBase: "/static/twin/data",
  });
  window.__twin = twinState.instance;
  return twinState.instance;
}

async function syncTwinRun() {
  const twin = twinState.instance;
  if (!twin || !state.current) return;
  if (twinState.loadedRun !== state.current) {
    twinState.loadedRun = state.current;
    // Runs without twin data show the module's own "twin data unavailable" note.
    await twin.load(state.current);
    twin.setAnalysis(state.analysis);
    // Re-apply the default perspective after load so the whole arm is framed.
    twin.resetView();
    twin.setTime($("video").currentTime || 0);
  } else {
    twin.setAnalysis(state.analysis);
  }
}

function startTwinClock() {
  cancelAnimationFrame(twinState.raf);
  const video = $("video");
  const step = () => {
    if (twinState.view !== "twin") return;
    if (twinState.instance) twinState.instance.setTime(video.currentTime || 0);
    twinState.raf = requestAnimationFrame(step);
  };
  twinState.raf = requestAnimationFrame(step);
}

async function setView(view) {
  twinState.view = view;
  const zone = document.querySelector(".video-zone");
  zone.dataset.mode = view;
  document.querySelectorAll("#view-toggle button").forEach((btn) => {
    const active = btn.dataset.view === view;
    btn.classList.toggle("active", active);
    btn.setAttribute("aria-pressed", String(active));
  });
  const mount = $("twin-mount");
  const provenance = $("twin-provenance");
  if (view === "twin") {
    mount.classList.remove("hidden");
    try {
      await ensureTwin();
      await syncTwinRun();
      startTwinClock();
      // The hidden video is the twin's clock — keep it running.
      $("video").play().catch(() => {});
      provenance.classList.add("hidden");
    } catch (error) {
      provenance.textContent = "3D twin failed to load — switch back to CAMERA.";
      provenance.classList.remove("hidden");
    }
  } else {
    // Fully restore the camera view; the hidden video kept playing all along.
    cancelAnimationFrame(twinState.raf);
    mount.classList.add("hidden");
    provenance.classList.add("hidden");
    drawOverlay();
  }
}

document.querySelectorAll("#view-toggle button").forEach((btn) => {
  btn.addEventListener("click", () => setView(btn.dataset.view));
});

boot().catch((error) => {
  $("event-headline").textContent = error.message;
});
