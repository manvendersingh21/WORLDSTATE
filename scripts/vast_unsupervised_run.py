"""Run WORLDSTATE's unsupervised flow pipeline on the 40 VAST warehouse clips (no labels)."""
from __future__ import annotations
import csv, json, re, sys, time
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from worldstate.engine import WorldModel  # noqa: E402
from worldstate.series import series_from_video_flow  # noqa: E402
from worldstate.learn import _path_agreement  # noqa: E402

VAST = ROOT / "data" / "vast"
OUT = VAST / "results"
OUT.mkdir(parents=True, exist_ok=True)
KEYWORDS = ["person near forklift", "near-miss", "near miss", "moving forklift", "close", "near", "pedestrian", "forklift"]

def caption_flags(c: str) -> dict:
    low = (c or "").lower()
    ev = re.search(r"EVENTS:\s*(.*?)\s*\|", c or "")
    risk = re.search(r"RISK:\s*<?(\w+)>?", c or "")
    return {
        "events": ev.group(1) if ev else "",
        "risk": risk.group(1).lower() if risk else "",
        "has_events": bool(ev and ev.group(1).strip("<> ").upper() not in ("NONE", "")),
        "kw": {k: (k in low) for k in KEYWORDS},
    }

def main():
    t0 = time.time()
    man = json.loads((VAST / "manifest.json").read_text())
    clips = man["clips"]
    eps, series_of = [], {}
    for i, c in enumerate(clips):
        vid = VAST / c["file"]
        eid = Path(c["file"]).stem
        split = "test" if i % 5 == 4 else "train"
        eps.append({"id": eid, "role": "reference" if split == "train" else "heldout", "split": split,
                    "kind": "unknown", "file": vid.name, "video": str(vid), "listed": True,
                    "dataset": "vast-sdg-warehouse", "failure_type": "unknown"})
        series_of[eid] = series_from_video_flow(vid)
    t_flow = time.time() - t0
    model = WorldModel()
    model.fit(eps, series_of, "flow")
    p = model.payload
    agreement = _path_agreement(model)
    seqs = p.get("sequences", {})
    seq_counter = Counter(tuple(s) for s in seqs.values())
    dom_seq, dom_n = seq_counter.most_common(1)[0]
    names = p.get("names", {})
    full = p.get("full_names", {})
    rows = []
    for ep, c in zip(eps, clips):
        d = model.analyze_series(ep["id"], series_of[ep["id"]])
        yd = c.get("yolo_detections") or {}
        counts = yd.get("object_counts", {})
        top = ", ".join(f"{k}:{v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])[:3])
        cf = caption_flags(c.get("cosmos_caption", ""))
        rows.append({
            "clip": ep["id"], "split": ep["split"], "score": d["score"], "status": d["status"],
            "first_divergence_s": d["first_divergence_s"], "threshold": d["threshold"],
            "components": d["components"],
            "observed_to": d["observed"]["to"],
            "states": [e.get("caption") or e.get("state") for e in d.get("events", [])],
            "mean_flow_mag": round(float(np.mean(series_of[ep["id"]]["mag"])), 4),
            "vast_similarity": c.get("similarity"),
            "caption_events": cf["events"], "caption_risk": cf["risk"], "caption_has_events": cf["has_events"],
            "caption_keywords": [k for k, v in cf["kw"].items() if v],
            "caption": c.get("cosmos_caption", ""),
            "yolo_top": top,
        })
    flagged = [r for r in rows if r["status"] != "normal"]
    normal = [r for r in rows if r["status"] == "normal"]
    def rate(group, pred):
        return None if not group else round(sum(1 for r in group if pred(r)) / len(group), 3)
    sanity = {}
    for label, pred in [("caption_has_EVENTS", lambda r: r["caption_has_events"]),
                        ("risk_not_low", lambda r: r["caption_risk"] not in ("", "low"))] + \
                       [(f"kw:{k}", (lambda k: lambda r: k in r["caption_keywords"])(k)) for k in KEYWORDS]:
        sanity[label] = {"flagged": rate(flagged, pred), "normal": rate(normal, pred)}
    scores = np.array([r["score"] for r in rows])
    report = {
        "dataset": {"n_clips": len(rows), "camera": man.get("camera_id"), "query": man.get("query"),
                     "train": sum(e["split"] == "train" for e in eps), "test": sum(e["split"] == "test" for e in eps),
                     "holdout_rule": "every 5th clip (index % 5 == 4)"},
        "feature_kind": "flow", "clusterer": p.get("clusterer"),
        "n_states": len(p.get("nodes", [])),
        "states": [{"id": n["id"], "name": names.get(n["id"]), "full": full.get(n["id"])} for n in p.get("nodes", [])],
        "threshold": p.get("threshold"), "novel_radius": p.get("novel_radius"),
        "path_agreement": round(agreement, 3),
        "dominant_path": [names.get(s, s) for s in dom_seq], "dominant_path_count": f"{dom_n}/{len(seqs)}",
        "n_distinct_train_sequences": len(seq_counter),
        "score_stats": {"min": float(scores.min()), "median": float(np.median(scores)), "max": float(scores.max()),
                         "mean": float(scores.mean())},
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "status_counts_test_only": dict(Counter(r["status"] for r in rows if r["split"] == "test")),
        "caption_sanity_check_WEAK": sanity,
        "runtime_s": {"flow": round(t_flow, 1), "total": round(time.time() - t0, 1)},
        "rows": rows,
    }
    (OUT / "vast_worldstate_report.json").write_text(json.dumps(report, indent=2, default=str))
    with open(OUT / "vast_worldstate_scores.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip", "split", "score", "status", "first_divergence_s", "mean_flow_mag", "caption_events",
                    "caption_risk", "caption_keywords", "yolo_top", "caption"])
        for r in rows:
            w.writerow([r["clip"], r["split"], r["score"], r["status"], r["first_divergence_s"], r["mean_flow_mag"],
                        r["caption_events"], r["caption_risk"], ";".join(r["caption_keywords"]), r["yolo_top"], r["caption"]])
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=1, default=str))
    for r in sorted(rows, key=lambda r: -r["score"]):
        print(r["clip"], r["split"], r["score"], r["status"], r["first_divergence_s"], r["mean_flow_mag"],
              "|", r["caption_events"], "|", r["caption_risk"], "|", r["yolo_top"])

if __name__ == "__main__":
    main()
