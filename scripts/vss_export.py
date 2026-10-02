"""Export clips from the team's VSS index (run on the VAST workshop VM).

Logs in with the VM's /config team file, searches the indexed corpus, downloads each
segment with its Cosmos Reason caption and YOLO detections, and zips them:
    python3 vss_export.py [query] [camera_id] [count]
Output: ~/worldstate_vss_export.zip (manifest.json, videos/*.mp4, captions + detections).
"""

from __future__ import annotations

import glob
import json
import os
import sys
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path


def load_config() -> dict:
    env = dict(os.environ)
    for path in glob.glob("/config/*.config"):
        for line in Path(path).read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, _, value = line.partition("=")
                env.setdefault(key.strip(), value.strip())
    return env


def call(base: str, path: str, token: str | None = None, body: dict | None = None, raw: bool = False):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=data, headers=headers, method="POST" if data else "GET")
    with urllib.request.urlopen(req, timeout=120) as resp:
        payload = resp.read()
    return payload if raw else json.loads(payload)


def main() -> None:
    query = sys.argv[1] if len(sys.argv) > 1 else "forklift moving through a warehouse aisle"
    camera = sys.argv[2] if len(sys.argv) > 2 else "sdg_warehouse_cam-2"
    count = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    env = load_config()
    base = env["INGRESS_URL"].rstrip("/")
    token = call(base, "/api/v1/auth/login", body={"username": env["USERNAME"], "password": env["PASSWORD"]})["access_token"]
    filters = {"camera_id": camera} if camera else {}
    hits = call(base, "/api/v1/search", token, {"query": query, "top_k": min(count, 100), "llm_top_n": 0,
                                                 "min_similarity": 0.0, "metadata_filters": filters})
    out = Path.home() / "worldstate_vss_export"
    (out / "videos").mkdir(parents=True, exist_ok=True)
    manifest = {"source": "VAST VSS team index", "query": query, "camera_id": camera, "clips": []}
    for i, hit in enumerate(hits.get("results", [])[:count]):
        source = hit.get("source")
        if not source:
            continue
        name = f"clip_{i:03d}.mp4"
        q = urllib.parse.urlencode({"source": source, "token": token})
        try:
            (out / "videos" / name).write_bytes(call(base, f"/api/v1/videos/stream?{q}", raw=True))
        except Exception as exc:
            print("skip", source, exc)
            continue
        try:
            detections = call(base, "/api/v1/videos/detections?" + urllib.parse.urlencode({"source": source}), token)
        except Exception:
            detections = None
        manifest["clips"].append({
            "file": f"videos/{name}", "source": source, "similarity": hit.get("similarity_score"),
            "cosmos_caption": hit.get("reasoning_content"), "camera_id": camera,
            "start": hit.get("start_time") or hit.get("start_sec"), "end": hit.get("end_time") or hit.get("end_sec"),
            "yolo_detections": detections,
        })
        print(f"{i + 1}/{count} {name}")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    archive = Path.home() / "worldstate_vss_export.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in out.rglob("*"):
            zf.write(path, path.relative_to(out))
    print("wrote", archive, len(manifest["clips"]), "clips")


if __name__ == "__main__":
    main()
