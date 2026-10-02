const state = { model: null, episodes: [], current: null, analysis: null, health: null };

const $ = (id) => document.getElementById(id);

async function api(path, opts) {
  const response = await fetch(path, opts);
  if (!response.ok) throw new Error((await response.text()) || response.statusText);
  const type = response.headers.get("content-type") || "";
  if (type.includes("application/json")) return response.json();
  return response;
}

function roleLabel(role) {
  return {
    unseen_failure: "Unseen failure",
    heldout_normal: "Held-out normal",
    reference: "Reference",
    eval_failure: "Held-out failure",
    upload: "Uploaded",
    similar_hidden: "Similar",
  }[role] || role;
}

async function boot() {
  state.health = await api("/api/health");
  state.model = await api("/api/model");
  state.episodes = await api("/api/episodes");
  renderChrome();
  renderEpisodes();
  renderGraph();
  const unseen = state.episodes.find((ep) => ep.role === "unseen_failure");
  if (unseen) await selectEpisode(unseen.id);
}

function renderChrome() {
  const version = state.model?.version || state.health?.version || 0;
  const n = state.model?.n_reference || 0;
  $("version").textContent = `world model v${version} · ${n} reference runs`;
  const fallback = state.health?.fallback;
  $("dataset-note").textContent = fallback
    ? `Exylos front-camera clips don’t share one path (agreement ${fallback.real_path_agreement}, AUROC ${Number(fallback.real_auc).toFixed(2)}). Graph below is the repeated cell.`
    : "Learned from repeated runs of the same cell. No SOP and no failure labels.";
  const adapters = state.health?.adapters || {};
  const cosmos = state.health?.cosmos || {};
  const memory = state.health?.memory || {};
  const cosmosLine = cosmos.succeeded
    ? `Cosmos Reason calls succeeded (${cosmos.succeeded})`
    : `Cosmos offline (${cosmos.skipped_reason || cosmos.last_error || "no key"})`;
  $("adapters").textContent = [
    `Perception ${adapters.perception || "classical"}`,
    `Embedder ${memory.embedder || adapters.text_embedder || "hashing"}`,
    `Index ${memory.vector_index || adapters.vector_index || "numpy"}`,
    cosmosLine,
    adapters.vast ? "VAST remote on" : "VAST local",
  ].join(" · ");
}

function renderEpisodes() {
  const list = $("episode-list");
  list.innerHTML = "";
  const order = ["unseen_failure", "heldout_normal", "eval_failure", "upload", "reference"];
  const ranked = [...state.episodes].sort((a, b) => order.indexOf(a.role) - order.indexOf(b.role) || a.id.localeCompare(b.id));
  for (const ep of ranked) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "ep" + (ep.role.includes("fail") ? " failure" : "") + (state.current === ep.id ? " active" : "");
    button.dataset.episode = ep.id;
    const title = document.createElement("span");
    title.textContent = ep.id;
    const sub = document.createElement("small");
    sub.textContent = roleLabel(ep.role);
    button.append(title, sub);
    button.addEventListener("click", () => selectEpisode(ep.id));
    list.append(button);
  }
}

async function selectEpisode(id) {
  state.current = id;
  renderEpisodes();
  const detail = await api(`/api/episodes/${id}`);
  const video = $("player");
  video.src = `/api/episodes/${id}/video`;
  video.dataset.frames = JSON.stringify(detail.frames || []);
  state.analysis = await api(`/api/episodes/${id}/analysis`);
  renderAnalysis();
  seekToStory();
  drawOverlay();
}

function renderAnalysis() {
  const analysis = state.analysis;
  const alert = $("alert");
  alert.classList.remove("hidden", "known", "normal");
  if (!analysis) {
    alert.classList.add("hidden");
    return;
  }
  const badge = (word, detail) => {
    alert.innerHTML = "";
    const strong = document.createElement("b");
    strong.className = "badge";
    strong.textContent = word;
    alert.append(strong, document.createTextNode(detail));
  };
  if (analysis.status === "novel") {
    badge("NOVEL FAILURE", `New transition at ${analysis.first_divergence_s}s · ${analysis.expected.from} → ${analysis.observed.to} · ${analysis.support.text} · score ${analysis.score.toFixed(2)}`);
  } else if (analysis.status === "known_failure") {
    alert.classList.add("known");
    badge("KNOWN FAILURE", `${analysis.known_class?.replaceAll("_", " ")} · recognized from memory · world model v${analysis.version}`);
  } else {
    alert.classList.add("normal");
    badge("NORMAL", `Matches the learned process · score ${analysis.score.toFixed(2)}`);
  }
  $("score-line").textContent = analysis.status === "known_failure"
    ? "KNOWN FAILURE"
    : analysis.status === "novel"
      ? "NOVEL FAILURE"
      : "NORMAL";
  const facts = $("facts");
  facts.innerHTML = "";
  const rows = [
    ["Score", analysis.score.toFixed(2)],
    ["Divergence", analysis.first_divergence_s == null ? "none" : `${analysis.first_divergence_s}s`],
    ["Expected", `${analysis.expected.from} → ${analysis.expected.to}`],
    ["Observed", analysis.observed.to],
    ["Support", analysis.support.text],
    ["P(expected)", analysis.expected.p.toFixed(2)],
  ];
  for (const [key, value] of rows) {
    const wrap = document.createElement("div");
    const dt = document.createElement("dt");
    dt.textContent = key;
    const dd = document.createElement("dd");
    dd.textContent = value;
    wrap.append(dt, dd);
    facts.append(wrap);
  }
  $("why").textContent = analysis.why;
  loadNarration(analysis.episode_id);
  $("expected-path").textContent = analysis.expected_path.map((step) => step.name).join(" → ");
  $("recovery-path").textContent = analysis.recovery_path.map((step) => step.name).join(" → ");
  $("video-stub").textContent = analysis.video_generation?.note || "";
  $("search-btn").disabled = false;
  $("remember-btn").disabled = analysis.status === "normal";
  $("remember-btn").textContent = analysis.status === "known_failure" ? "Failure class saved" : "Remember this failure";
  renderSequence(analysis);
  renderTimeline(analysis);
  renderGraph();
}

function renderSequence(analysis) {
  const host = $("sequence");
  host.innerHTML = "";
  for (const step of analysis.sequence) {
    const chip = document.createElement("span");
    chip.textContent = step.name;
    if (step.state === "novel" || step.name === analysis.observed.to) chip.className = "bad";
    host.append(chip);
  }
}

function renderTimeline(analysis) {
  const bar = $("timeline");
  bar.innerHTML = "";
  if (analysis.first_divergence_s == null) return;
  const video = $("player");
  const duration = video.duration || 8;
  const mark = document.createElement("i");
  mark.style.left = `${(analysis.first_divergence_s / duration) * 100}%`;
  bar.append(mark);
  bar.onclick = (event) => {
    const rect = bar.getBoundingClientRect();
    video.currentTime = ((event.clientX - rect.left) / rect.width) * (video.duration || duration);
  };
}

function renderGraph() {
  const svg = $("graph");
  const model = state.model;
  if (!model?.nodes) {
    svg.innerHTML = "";
    return;
  }
  const nominal = model.nodes.filter((node) => node.kind !== "failure");
  const failures = model.nodes.filter((node) => node.kind === "failure");
  const width = 980;
  const y = 78;
  const xOf = {};
  nominal.forEach((node, index) => {
    xOf[node.id] = 70 + index * ((width - 140) / Math.max(nominal.length - 1, 1));
  });
  failures.forEach((node) => {
    const edge = (model.edges || []).find((item) => item.dst === node.id && item.kind === "failure");
    xOf[node.id] = edge && xOf[edge.src] ? xOf[edge.src] + 40 : width / 2;
  });
  const analysis = state.analysis;
  const expected = new Set((analysis?.expected_path || []).map((step) => step.id));
  let markup = "";
  for (const edge of model.edges || []) {
    if (xOf[edge.src] == null || xOf[edge.dst] == null) continue;
    const x1 = xOf[edge.src];
    const x2 = xOf[edge.dst];
    const y1 = edge.kind === "failure" || edge.kind === "recovery" ? 150 : y;
    const y2 = edge.kind === "failure" ? 150 : edge.kind === "recovery" ? y : y;
    const color = edge.kind === "failure" ? "#ef6f5e" : edge.kind === "recovery" ? "#8ec8d8" : expected.has(edge.src) && expected.has(edge.dst) ? "#e6a23c" : "#8d927c";
    const dash = edge.kind === "recovery" ? 'stroke-dasharray="5 4"' : "";
    const mid = (x1 + x2) / 2;
    markup += `<path d="M ${x1} ${y} C ${mid} ${y1}, ${mid} ${y2}, ${x2} ${edge.kind === "failure" ? 150 : y}" fill="none" stroke="${color}" stroke-width="2" ${dash}/>`;
    if (edge.kind === "nominal") {
      markup += `<text x="${mid}" y="${y - 14}" fill="#b1ab99" font-size="11" text-anchor="middle">${Number(edge.p).toFixed(2)} · ${edge.episodes}</text>`;
    }
  }
  if (analysis?.status === "novel") {
    const from = nominal.find((node) => node.name === analysis.expected.from);
    if (from) {
      const x1 = xOf[from.id];
      markup += `<path d="M ${x1} ${y} C ${x1 + 30} 150, ${x1 + 50} 150, ${x1 + 70} 150" fill="none" stroke="#ef6f5e" stroke-width="2"/>`;
      markup += `<circle cx="${x1 + 70}" cy="150" r="16" fill="#3a221c" stroke="#ef6f5e"/>`;
      markup += `<text x="${x1 + 70}" y="182" fill="#ffd0c8" font-size="11" text-anchor="middle">${escapeXml(analysis.observed.to)}</text>`;
    }
  }
  for (const node of model.nodes) {
    const x = xOf[node.id];
    const ny = node.kind === "failure" ? 150 : y;
    const active = expected.has(node.id);
    markup += `<circle cx="${x}" cy="${ny}" r="16" fill="${node.kind === "failure" ? "#3a221c" : active ? "#3a2e16" : "#22261c"}" stroke="${node.kind === "failure" ? "#ef6f5e" : active ? "#e6a23c" : "#c5d67a"}" stroke-width="2"/>`;
    markup += `<text x="${x}" y="${ny + 34}" fill="#f3efe2" font-size="11" text-anchor="middle">${escapeXml(graphLabel(node.name))}</text>`;
  }
  svg.innerHTML = markup;
  $("graph-meta").textContent = `${nominal.length} discovered states · clusterer ${model.clusterer || ""}`;
}

function graphLabel(name) {
  return String(name).replace(" part", "").replace(" in fixture", "");
}

function escapeXml(value) {
  return String(value).replace(/[&<>]/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[ch]));
}

function drawOverlay() {
  const video = $("player");
  const canvas = $("overlay");
  const frames = JSON.parse(video.dataset.frames || "[]");
  if (!frames.length || !video.videoWidth) return;
  const rect = video.getBoundingClientRect();
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.style.width = `${rect.width}px`;
  canvas.style.height = `${rect.height}px`;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  let frame = frames[0];
  for (const candidate of frames) {
    if (candidate.time <= video.currentTime) frame = candidate;
  }
  const colors = { gripper: "#ef6f5e", part: "#7ea2ff", fixture: "#c5d67a" };
  ctx.lineWidth = 4;
  ctx.font = "16px sans-serif";
  for (const obj of frame.objects || []) {
    const x = (obj.cx - obj.w / 2) * canvas.width;
    const y = (obj.cy - obj.h / 2) * canvas.height;
    ctx.strokeStyle = colors[obj.label] || "#fff";
    ctx.strokeRect(x, y, obj.w * canvas.width, obj.h * canvas.height);
    ctx.fillStyle = ctx.strokeStyle;
    ctx.fillText(obj.label, x, Math.max(16, y - 4));
  }
}

$("player").addEventListener("timeupdate", drawOverlay);
function seekToStory() {
  const video = $("player");
  const analysis = state.analysis;
  if (!video || !analysis || !Number.isFinite(video.duration)) return;
  if (analysis.first_divergence_s == null || analysis.status === "normal") {
    video.currentTime = analysis.status === "normal" ? Math.min(6.4, video.duration - 0.2) : 0;
  } else {
    video.currentTime = Math.min(video.duration - 0.25, analysis.first_divergence_s + 1.15);
  }
  renderTimeline(analysis);
}

$("player").addEventListener("loadedmetadata", seekToStory);

$("learn-btn").addEventListener("click", async () => {
  $("learn-btn").disabled = true;
  $("learn-btn").textContent = "Learning…";
  try {
    state.model = await api("/api/learn", { method: "POST" });
    state.health = await api("/api/health");
    state.episodes = await api("/api/episodes");
    renderChrome();
    renderEpisodes();
    renderGraph();
  } finally {
    $("learn-btn").disabled = false;
    $("learn-btn").textContent = "Learn process";
  }
});

$("remember-btn").addEventListener("click", async () => {
  if (!state.current) return;
  $("remember-btn").disabled = true;
  const payload = await api(`/api/episodes/${state.current}/remember`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ label: "displaced_after_align" }),
  });
  state.model = payload.model;
  state.analysis = payload.analysis;
  state.health = await api("/api/health");
  renderChrome();
  renderAnalysis();
});

function renderHits(cards, label) {
  const host = $("search-results");
  host.innerHTML = "";
  if (label) {
    const head = document.createElement("p");
    head.className = "stub";
    head.textContent = label;
    host.append(head);
  }
  for (const hit of cards) {
    const card = document.createElement("article");
    card.className = "hit";
    const title = document.createElement("strong");
    title.textContent = hit.next_state ? `${hit.id} · next ${hit.next_state}` : `${hit.id} · ${hit.kind || hit.role || ""}`;
    const body = document.createElement("div");
    body.textContent = hit.snippet || "";
    card.append(title, body);
    if (state.episodes.some((ep) => ep.id === hit.id)) {
      card.addEventListener("click", () => selectEpisode(hit.id));
    }
    host.append(card);
  }
  if (!cards.length) host.append(Object.assign(document.createElement("p"), { className: "stub", textContent: "No matching runs in memory." }));
}

async function askMemory(query) {
  $("search-q").value = query;
  const hits = await api(`/api/search?q=${encodeURIComponent(query)}&k=5`);
  renderHits(hits, `Memory results for “${query}” (SQLite + FAISS)`);
}

$("search-btn").addEventListener("click", async () => {
  const typed = $("search-q").value.trim();
  if (typed) return askMemory(typed);
  const analogous = state.analysis?.analogous || [];
  if (analogous.length) return renderHits(analogous, "Reference runs that shared this prefix");
  return askMemory(state.analysis?.expected?.from || "unusual transition");
});

$("search-q").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && event.target.value.trim()) askMemory(event.target.value.trim());
});

document.querySelectorAll(".chips button").forEach((chip) => {
  chip.addEventListener("click", () => askMemory(chip.dataset.q));
});

$("upload-similar-btn").addEventListener("click", async () => {
  $("upload-similar-btn").disabled = true;
  $("upload-similar-btn").textContent = "Uploading…";
  try {
    const blob = await (await fetch("/api/demo/similar-file")).blob();
    const body = new FormData();
    body.append("file", blob, "similar_run.mp4");
    const payload = await api("/api/upload", { method: "POST", body });
    state.episodes = await api("/api/episodes");
    await selectEpisode(payload.episode.id);
  } finally {
    $("upload-similar-btn").disabled = false;
    $("upload-similar-btn").textContent = "Upload similar run";
  }
});

$("file-input").addEventListener("change", async (event) => {
  const file = event.target.files?.[0];
  if (!file) return;
  const body = new FormData();
  body.append("file", file, file.name);
  const payload = await api("/api/upload", { method: "POST", body });
  state.episodes = await api("/api/episodes");
  await selectEpisode(payload.episode.id);
});

boot().catch((error) => {
  $("why").textContent = error.message;
});

async function loadNarration(episodeId) {
  const text = $("narration");
  const badge = $("narration-source");
  text.textContent = "Asking the event model…";
  badge.textContent = "";
  badge.className = "source";
  try {
    const res = await fetch(`/api/episodes/${encodeURIComponent(episodeId)}/narration`);
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    if (state.current && state.current !== episodeId) return;
    text.textContent = data.narrative;
    if (data.source === "cosmos-reason") {
      badge.textContent = `from Cosmos Reason · ${data.model}`;
      badge.classList.add("cosmos");
    } else {
      badge.textContent = "kinematic fallback";
      badge.title = data.reason || "";
    }
  } catch (error) {
    text.textContent = "";
    badge.textContent = "unavailable";
  }
}
