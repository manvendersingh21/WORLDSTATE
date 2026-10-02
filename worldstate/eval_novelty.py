"""Held-out novelty detection: failures should score above normals, then become a known class."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from worldstate.config import ROOT
from worldstate.engine import WorldModel
from worldstate.series import series_from_tracks


def _series(episode_id: str):
    path = Path(f"data/model/tracks/{episode_id}.json")
    if not path.exists():
        path = ROOT / "data" / "model" / "tracks" / f"{episode_id}.json"
    return series_from_tracks(json.loads(path.read_text()))


def main() -> None:
    model = WorldModel.load()
    if not model.ready:
        raise SystemExit("no world model; run python -m worldstate.learn")
    rows = []
    for ep in model.payload["episodes"]:
        if ep["split"] == "train":
            continue
        detail = model.analyze_series(ep["id"], _series(ep["id"]))
        normal = ep["kind"] == "normal" or ep.get("role") == "heldout_normal"
        rows.append((ep["id"], normal, detail["score"], detail["status"], detail["observed"]["to"]))
        print(f"{ep['id']:18} {'normal' if normal else 'failure':8} score={detail['score']:.3f} {detail['status']:16} {detail['observed']['to']}")
    normals = [score for _, normal, score, _, _ in rows if normal]
    failures = [score for _, normal, score, _, _ in rows if not normal]
    labels = [0] * len(normals) + [1] * len(failures)
    scores = normals + failures
    auc = float(roc_auc_score(labels, scores)) if len(set(scores)) > 1 else 0.0
    separated = bool(failures) and bool(normals) and min(failures) > max(normals)
    print(f"auc={auc:.3f} normal_max={max(normals):.3f} failure_min={min(failures):.3f} separated={separated}")

    clone = WorldModel(json.loads(json.dumps(model.payload)))
    clone.save = lambda *args, **kwargs: None
    unseen = next(ep["id"] for ep in model.payload["episodes"] if ep["role"] == "unseen_failure")
    analysis = clone.analyze_series(unseen, _series(unseen))
    clone.remember(unseen, _series(unseen), "displaced_after_align", analysis)
    memory_ok = True
    for ep in model.payload["episodes"]:
        if ep["id"] == unseen or ep["split"] == "train":
            continue
        if ep["role"] == "similar_hidden" or ep["kind"].startswith("miss") or ep["role"] == "heldout_normal":
            detail = clone.analyze_series(ep["id"], _series(ep["id"]))
            expect = "normal" if ep["role"] == "heldout_normal" or ep["kind"] == "normal" else "known_failure"
            ok = detail["status"] == expect
            memory_ok = memory_ok and ok
            print(f"after remember {ep['id']:18} got {detail['status']:16} class={detail.get('known_class')} expect {expect} {'ok' if ok else 'FAIL'}")
    report = {
        "auc": auc,
        "separated": separated,
        "normal_max": max(normals),
        "failure_min": min(failures),
        "memory_ok": memory_ok,
        "n_reference": model.payload["n_reference"],
        "dataset": model.payload.get("dataset"),
        "threshold": model.payload.get("threshold"),
    }
    out = ROOT / "data" / "eval_report.json"
    out.write_text(json.dumps(report, indent=2))
    print("memory_ok", memory_ok)
    if not separated or auc < 0.9 or not memory_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
