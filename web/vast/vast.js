(function () {
  "use strict";

  var clips = [];
  var grid = document.getElementById("grid");
  var empty = document.getElementById("empty");
  var meta = document.getElementById("meta");
  var q = document.getElementById("q");
  var detail = document.getElementById("detail");
  var detailBody = document.getElementById("detail-body");

  // ---------- helpers ----------
  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  }
  function esc(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }
  function fmtNum(x, d) {
    var n = Number(x);
    return isFinite(n) ? n.toFixed(d) : null;
  }
  function sentences(text) {
    var m = String(text || "").match(/[^.!?]+[.!?]+["')\]]*|[^.!?]+$/g);
    return (m || []).map(function (s) { return s.trim(); }).filter(Boolean);
  }
  function shortCaption(text) {
    var s = sentences(text);
    if (s.length <= 3) return { text: String(text || "").trim(), truncated: false };
    var n = s.slice(0, 2).join(" ").length < 140 ? 3 : 2;
    return { text: s.slice(0, n).join(" "), truncated: true };
  }

  // ---------- stemming + ranking ----------
  var STOP = {};
  ("a an the of in on at to and or is are was were be been with near by for from into as it its this that there " +
   "these those some any while who which what where when then than very can could also").split(" ")
    .forEach(function (w) { STOP[w] = 1; });
  function stem(w) {
    w = w.toLowerCase();
    if (w.length > 4 && /ies$/.test(w)) return w.slice(0, -3) + "y";
    if (w.length > 5 && /ing$/.test(w)) { w = w.slice(0, -3); if (/(.)\1$/.test(w)) w = w.slice(0, -1); return w; }
    if (w.length > 4 && /ed$/.test(w)) { w = w.slice(0, -2); if (/(.)\1$/.test(w)) w = w.slice(0, -1); return w; }
    if (w.length > 3 && /es$/.test(w) && /(s|x|z|ch|sh)es$/.test(w)) return w.slice(0, -2);
    if (w.length > 3 && /s$/.test(w) && !/ss$/.test(w)) return w.slice(0, -1);
    return w;
  }
  var SYN = { people: "person", men: "person", man: "person", woman: "person", women: "person", worker: "person",
    pedestrian: "person", human: "person", walk: "walk", lift: "forklift", "fork-lift": "forklift", empty: "empty",
    vacant: "empty", clear: "empty", corridor: "aisle", lane: "aisle" };
  function terms(text) {
    var out = [];
    String(text || "").toLowerCase().split(/[^a-z0-9-]+/).forEach(function (w) {
      if (!w || STOP[w]) return;
      var s = stem(w);
      out.push(SYN[s] || SYN[w] || s);
    });
    return out;
  }
  function score(clip, qTerms) {
    if (!qTerms.length) return 0;
    var tf = clip._tf;
    var total = 0, hit = 0;
    qTerms.forEach(function (t) {
      if (tf[t]) { hit++; total += 1 + Math.log(tf[t]); }
    });
    return hit === 0 ? 0 : (hit / qTerms.length) * 2 + total * 0.25;
  }

  // ---------- YOLO detections (defensive) ----------
  var CLS_KEYS = ["class", "label", "name", "class_name", "category", "object", "type"];
  var CONF_KEYS = ["conf", "confidence", "score", "prob", "probability"];
  function pick(o, keys) {
    for (var i = 0; i < keys.length; i++) if (o[keys[i]] != null) return o[keys[i]];
    return undefined;
  }
  function summarizeDetections(det) {
    if (det == null) return null;
    var counts = {}, maxConf = {}, seen = 0;
    function add(cls, conf, n) {
      cls = String(cls).trim();
      if (!cls || cls.length > 40) return;
      counts[cls] = (counts[cls] || 0) + (n || 1);
      var c = conf == null || conf === "" ? NaN : Number(conf);
      if (isFinite(c)) { if (c > 1 && c <= 100) c = c / 100; if (maxConf[cls] == null || c > maxConf[cls]) maxConf[cls] = c; }
      seen++;
    }
    function walk(node, depth) {
      if (node == null || depth > 12) return;
      if (Array.isArray(node)) { node.forEach(function (x) { walk(x, depth + 1); }); return; }
      if (typeof node !== "object") return;
      var cls = pick(node, CLS_KEYS);
      if (typeof cls === "string" || typeof cls === "number") {
        var conf = pick(node, CONF_KEYS);
        var cnt = Number(node.count);
        add(cls, conf, isFinite(cnt) && cnt > 0 && conf == null ? cnt : 1);
      }
      // count maps: {"object_counts": {"person": 3, "forklift": 1}} / {"classes": {...}}
      ["object_counts", "counts", "class_counts", "classes"].forEach(function (k) {
        var v = node[k];
        if (v && typeof v === "object" && !Array.isArray(v)) {
          Object.keys(v).forEach(function (name) {
            var n = Number(v[name]);
            if (isFinite(n) && n > 0) add(name, null, n);
          });
        }
      });
      Object.keys(node).forEach(function (k) {
        var v = node[k];
        if (v && typeof v === "object" && ["object_counts", "counts", "class_counts"].indexOf(k) < 0) walk(v, depth + 1);
      });
    }
    try { walk(det, 0); } catch (e) { return null; }
    if (!seen) return null;
    return Object.keys(counts).map(function (k) {
      return { cls: k, count: counts[k], conf: maxConf[k] };
    }).sort(function (a, b) { return b.count - a.count; });
  }
  function renderYolo(summary, container) {
    container.appendChild(el("div", "ylabel", "YOLO11 detections"));
    if (!summary || !summary.length) { container.appendChild(el("span", "ynone", "no detections sidecar")); return; }
    summary.slice(0, 8).forEach(function (d) {
      var c = el("span", "ychip");
      var conf = d.conf != null ? " · max " + d.conf.toFixed(2) : "";
      c.innerHTML = esc(d.cls) + ' <span class="c">×' + d.count + esc(conf) + "</span>";
      container.appendChild(c);
    });
  }

  // ---------- lazy video ----------
  var io = "IntersectionObserver" in window ? new IntersectionObserver(function (entries) {
    entries.forEach(function (e) {
      if (e.isIntersecting) { attachSrc(e.target); io.unobserve(e.target); }
    });
  }, { rootMargin: "300px" }) : null;
  function attachSrc(v) {
    if (v.dataset.src && !v.getAttribute("src")) { v.src = v.dataset.src; }
  }

  // ---------- rendering ----------
  function highlight(text, qTerms) {
    var safe = esc(text);
    if (!qTerms.length) return safe;
    var set = {}; qTerms.forEach(function (t) { set[t] = 1; });
    return safe.replace(/[A-Za-z][A-Za-z0-9-]*/g, function (w) {
      var s = stem(w); s = SYN[s] || SYN[w.toLowerCase()] || s;
      return set[s] && !STOP[w.toLowerCase()] ? "<mark>" + w + "</mark>" : w;
    });
  }
  function clipName(c, i) {
    var f = String(c.file || "").split("/").pop();
    return f || "clip " + i;
  }
  function render() {
    var qTerms = terms(q.value);
    var list = clips.map(function (c) { return { c: c, s: score(c, qTerms) }; });
    if (qTerms.length) {
      list.sort(function (a, b) { return b.s - a.s || (Number(b.c.similarity) || 0) - (Number(a.c.similarity) || 0); });
    }
    var matched = list.filter(function (x) { return x.s > 0; }).length;
    meta.textContent = qTerms.length
      ? matched + " of " + clips.length + " clips match “" + q.value.trim() + "”."
      : clips.length + " clips" + (window.__vastQuery ? " · VSS query: “" + window.__vastQuery + "”" : "") + ".";
    grid.innerHTML = "";
    list.forEach(function (x) { grid.appendChild(card(x.c, x.s, qTerms)); });
    document.querySelectorAll(".chip").forEach(function (ch) {
      ch.classList.toggle("active", ch.dataset.q === q.value.trim());
    });
  }
  function card(c, s, qTerms) {
    var node = el("article", "card");
    node.tabIndex = 0;
    if (qTerms.length && s === 0) node.style.opacity = "0.45";

    var vid = el("div", "vid");
    var v = document.createElement("video");
    v.muted = true; v.playsInline = true; v.controls = true; v.preload = "metadata"; v.loop = true;
    v.setAttribute("muted", "");
    v.dataset.src = c.file;
    v.addEventListener("error", function () {
      if (!vid.querySelector(".ph")) vid.appendChild(el("div", "ph", "video unavailable"));
    });
    if (io) io.observe(v); else attachSrc(v);
    vid.appendChild(v);
    node.appendChild(vid);
    node.addEventListener("mouseenter", function () { attachSrc(v); var p = v.play(); if (p && p.catch) p.catch(function () {}); });
    node.addEventListener("mouseleave", function () { v.pause(); });
    v.addEventListener("click", function (e) { e.stopPropagation(); });

    var body = el("div", "card-body");
    var head = el("div", "card-head");
    head.appendChild(el("span", "name", clipName(c, 0)));
    var right = el("span");
    var sim = fmtNum(c.similarity, 2);
    right.innerHTML = (sim != null ? '<span class="sim">VSS similarity ' + sim + "</span>" : "") +
      (qTerms.length ? ' <span class="score">· match ' + s.toFixed(2) + "</span>" : "");
    head.appendChild(right);
    body.appendChild(head);

    body.appendChild(el("div", "attr", "Cosmos Reason (VAST pipeline)"));
    var full = String(c.cosmos_caption || "").trim();
    var cap = el("p", "caption");
    if (!full) { cap.textContent = "No Cosmos caption in export."; cap.style.color = "var(--faint)"; }
    var short = shortCaption(full);
    var expanded = false;
    function paint() { if (full) cap.innerHTML = highlight(expanded ? full : short.text, qTerms); }
    paint();
    body.appendChild(cap);
    if (short.truncated) {
      var more = el("button", "more", "Show full caption");
      more.type = "button";
      more.addEventListener("click", function (e) {
        e.stopPropagation(); expanded = !expanded; paint();
        more.textContent = expanded ? "Show less" : "Show full caption";
      });
      body.appendChild(more);
    }
    var y = el("div", "yolo");
    renderYolo(c._yolo, y);
    body.appendChild(y);
    node.appendChild(body);

    node.addEventListener("click", function () { openDetail(c, qTerms); });
    node.addEventListener("keydown", function (e) { if (e.key === "Enter") openDetail(c, qTerms); });
    return node;
  }

  function openDetail(c, qTerms) {
    detailBody.innerHTML = "";
    var v = document.createElement("video");
    v.controls = true; v.muted = true; v.autoplay = true; v.playsInline = true; v.loop = true; v.preload = "auto";
    v.src = c.file;
    var p = v.play && v.play(); if (p && p.catch) p.catch(function () {});
    detailBody.appendChild(v);

    var kv = el("div", "kv");
    var parts = [["Clip", clipName(c, 0)], ["Camera", c.camera_id], ["VSS similarity", fmtNum(c.similarity, 3)],
      ["Window", c.start != null || c.end != null ? (c.start != null ? c.start : "?") + " → " + (c.end != null ? c.end : "?") : null],
      ["Source", c.source]];
    parts.forEach(function (p) {
      if (p[1] == null || p[1] === "") return;
      var s = el("span"); s.innerHTML = esc(p[0]) + ": <b>" + esc(p[1]) + "</b>"; kv.appendChild(s);
    });
    detailBody.appendChild(kv);

    detailBody.appendChild(el("h3", null, "Cosmos Reason caption"));
    detailBody.appendChild(el("div", "attr", "Cosmos Reason (VAST pipeline)"));
    var cap = el("p", "caption");
    cap.innerHTML = c.cosmos_caption ? highlight(String(c.cosmos_caption).trim(), qTerms) : "No Cosmos caption in export.";
    detailBody.appendChild(cap);

    detailBody.appendChild(el("h3", null, "Detections"));
    var y = el("div", "yolo");
    renderYolo(c._yolo, y);
    detailBody.appendChild(y);

    detail.classList.remove("hidden");
    document.getElementById("detail-close").focus();
  }
  function closeDetail() {
    var v = detailBody.querySelector("video");
    if (v) { v.pause(); v.removeAttribute("src"); v.load(); }
    detail.classList.add("hidden");
  }
  document.getElementById("detail-close").addEventListener("click", closeDetail);
  detail.addEventListener("click", function (e) { if (e.target === detail) closeDetail(); });
  document.addEventListener("keydown", function (e) { if (e.key === "Escape" && !detail.classList.contains("hidden")) closeDetail(); });

  // ---------- search wiring ----------
  var t = null;
  q.addEventListener("input", function () { clearTimeout(t); t = setTimeout(render, 80); });
  document.getElementById("clear").addEventListener("click", function () { q.value = ""; render(); q.focus(); });
  document.querySelectorAll(".chip").forEach(function (ch) {
    ch.addEventListener("click", function () { q.value = ch.dataset.q; render(); });
  });

  // ---------- load ----------
  function showEmpty(reason) {
    empty.classList.remove("hidden");
    grid.innerHTML = "";
    meta.textContent = "";
    if (reason) console.info("[vast] " + reason);
  }
  fetch("manifest.json", { cache: "no-cache" })
    .then(function (r) {
      if (!r.ok) throw new Error("manifest.json " + r.status);
      return r.json();
    })
    .then(function (m) {
      var list = (m && Array.isArray(m.clips)) ? m.clips : (Array.isArray(m) ? m : []);
      if (!list.length) return showEmpty("manifest has no clips");
      window.__vastQuery = m && m.query;
      clips = list.filter(function (c) { return c && c.file; }).map(function (c) {
        var tf = {};
        terms(c.cosmos_caption).forEach(function (w) { tf[w] = (tf[w] || 0) + 1; });
        c._tf = tf;
        c._yolo = summarizeDetections(c.yolo_detections);
        return c;
      });
      if (!clips.length) return showEmpty("manifest clips have no files");
      render();
    })
    .catch(function (e) { showEmpty(e && e.message); });
})();
