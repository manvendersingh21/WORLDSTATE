"""WORLDSTATE demo API and static UI."""

from __future__ import annotations

import json
import shutil
import threading
import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from worldstate.adapters import CosmosReasonAdapter
from worldstate.config import MODEL_DIR, TRACKS_DIR, UPLOAD_DIR, WEB_DIR
from worldstate.engine import WorldModel
from worldstate.learn import learn_model
from worldstate.memory import MemoryStore
from worldstate.narrative import describe_run, run_facts
from worldstate.perception import perceive, tracks_for
from worldstate.series import series_from_tracks, series_from_video_flow

app = FastAPI(title="WORLDSTATE")
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

_lock = threading.Lock()
_model = WorldModel()
_memory = None
_cache: dict[tuple, dict] = {}


def _boot() -> None:
    global _model, _memory, _cache
    _model = WorldModel.load()
    _memory = MemoryStore()
    _cache = {}


@app.on_event("startup")
def startup() -> None:
    _boot()


def _episode(episode_id: str) -> dict:
    for ep in _model.payload.get("episodes", []):
        if ep["id"] == episode_id:
            return ep
    raise HTTPException(404, "episode not found")


def _series(ep: dict):
    if _model.payload.get("feature_kind") == "flow":
        path = TRACKS_DIR / f"{ep['id']}.flow.npz"
        if path.exists():
            import numpy as np

            blob = np.load(path, allow_pickle=False)
            series = {key: blob[key] for key in blob.files}
            series["feature_kind"] = np.array(["flow"])
            return series
        return series_from_video_flow(Path(ep["video"]))
    return series_from_tracks(tracks_for(Path(ep["video"]), ep["id"], _model.payload.get("perception_mode")))


def _analyze(episode_id: str) -> dict:
    key = (episode_id, _model.version)
    if key in _cache:
        return _cache[key]
    ep = _episode(episode_id)
    detail = _model.analyze_series(episode_id, _series(ep))
    detail["analogous"] = _model.analogous([step["state"] for step in detail["sequence"]])
    _cache[key] = detail
    return detail


_narrations: dict[tuple, dict] = {}
_narrate_lock = threading.Lock()


@app.get("/api/episodes/{episode_id}/narration")
def narration(episode_id: str):
    """Event text for one run: Cosmos Reason when it answers, else the kinematic events.

    Cluster identity and the novelty verdict stay with the graph; Cosmos only writes text.
    """
    if not _model.ready:
        raise HTTPException(400, "learn a process first")
    detail = _analyze(episode_id)
    key = (episode_id, _model.version)
    cache_dir = MODEL_DIR / "narrations"
    cache_dir.mkdir(exist_ok=True)
    cache_path = cache_dir / f"{episode_id}.v{_model.version}.json"
    with _narrate_lock:
        if key in _narrations:
            return _narrations[key]
        if cache_path.exists():
            _narrations[key] = json.loads(cache_path.read_text())
            return _narrations[key]
        ep = _episode(episode_id)
        cosmos = CosmosReasonAdapter()
        video_path = Path(ep["video"])
        facts = run_facts(_series(ep))
        tracked = {
            "cube_shifted_sideways_at_s": facts["shift_t"],
            "cube_was_lifted": facts["lifted"],
            "cube_ended_on_target_pad": facts["ends_on_target"],
        }
        narrative = cosmos.narrate_run(video_path, detail, tracked) if cosmos.available else None
        events = None
        if narrative:
            summary = " ".join(ev["action"] for ev in detail["events"])
            events = cosmos.refine_events(video_path, detail["events"], summary)
        kinematic = " ".join(f"At {ev['time']:.2f}s {ev['action']}." for ev in detail["events"])
        if narrative:
            result = {
                "source": "cosmos-reason",
                "model": cosmos.model,
                "narrative": narrative,
                "events": events or detail["events"],
                "events_source": "cosmos-reason" if events else "kinematic",
            }
        else:
            result = {
                "source": "kinematic",
                "model": None,
                "narrative": kinematic,
                "events": detail["events"],
                "events_source": "kinematic",
                "reason": cosmos.last_status.get("skipped_reason") or cosmos.last_status.get("last_error"),
            }
        _narrations[key] = result
        if result["source"] == "cosmos-reason":
            cache_path.write_text(json.dumps(result))
        return result


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/health")
def health():
    adapters = dict(_model.payload.get("adapters") or {})
    cosmos = _model.payload.get("cosmos") or {}
    memory = _memory.stats() if _memory else {}
    return {
        "ok": _model.ready,
        "version": _model.version,
        "dataset": _model.payload.get("dataset"),
        "fallback": _model.payload.get("fallback"),
        "cosmos": cosmos,
        "adapters": adapters,
        "memory": memory,
    }


@app.get("/api/model")
def model():
    if not _model.ready:
        return {"ready": False}
    public = _model.public()
    public["ready"] = True
    return public


@app.get("/api/episodes")
def episodes():
    return [
        {
            "id": ep["id"],
            "role": ep["role"],
            "kind": ep["kind"],
            "split": ep["split"],
            "listed": ep.get("listed", True),
        }
        for ep in _model.payload.get("episodes", [])
        if ep.get("listed", True)
    ]


@app.get("/api/episodes/{episode_id}")
def episode_detail(episode_id: str):
    ep = _episode(episode_id)
    tracks_path = TRACKS_DIR / f"{ep['id']}.json"
    frames = []
    duration = None
    fps = None
    if tracks_path.exists():
        tracks = json.loads(tracks_path.read_text())
        frames = tracks["frames"]
        duration = tracks.get("duration")
        fps = tracks.get("fps")
    return {**ep, "frames": frames, "duration": duration, "fps": fps}


@app.get("/api/episodes/{episode_id}/analysis")
def analysis(episode_id: str):
    if not _model.ready:
        raise HTTPException(400, "learn a process first")
    return _analyze(episode_id)


@app.get("/api/episodes/{episode_id}/video")
def video(episode_id: str):
    ep = _episode(episode_id)
    path = Path(ep["video"])
    if not path.exists():
        raise HTTPException(404, "video missing")
    return FileResponse(path, media_type="video/mp4")


class RememberBody(BaseModel):
    label: str = "displaced_after_align"


@app.post("/api/episodes/{episode_id}/remember")
def remember(episode_id: str, body: RememberBody):
    if not _model.ready:
        raise HTTPException(400, "learn a process first")
    detail = _analyze(episode_id)
    if detail["status"] == "normal":
        raise HTTPException(400, "this run matches the learned process")
    label = body.label.strip() or "displaced_after_align"
    ep = _episode(episode_id)
    with _lock:
        result = _model.remember(episode_id, _series(ep), label, detail)
        _memory.add_rule(result["version"], result["rule"])
        series = _series(ep)
        _cache.clear()
        after = _analyze(episode_id)
        _memory.add_episode(episode_id, describe_run(episode_id, after, series, label), ep["role"], label, after["events"])
    return {"remembered": result, "analysis": _analyze(episode_id), "model": _model.public()}


@app.get("/api/search")
def search(q: str, k: int = 5):
    return _memory.search(q, k)


@app.get("/api/demo/similar-file")
def similar_file():
    for ep in _model.payload.get("episodes", []):
        if ep["role"] == "similar_hidden":
            path = Path(ep["video"])
            if path.exists():
                return FileResponse(path, media_type="video/mp4", filename="similar_run.mp4")
    raise HTTPException(404, "no similar clip bundled")


def _ingest(path: Path, episode_id: str) -> dict:
    tracks = perceive(path, _model.payload.get("perception_mode"))
    tracks["episode_id"] = episode_id
    (TRACKS_DIR / f"{episode_id}.json").write_text(json.dumps(tracks))
    episode = {
        "id": episode_id,
        "role": "upload",
        "split": "external",
        "kind": "upload",
        "file": path.name,
        "video": str(path),
        "listed": True,
    }
    _model.payload["episodes"] = [ep for ep in _model.payload["episodes"] if ep["id"] != episode_id] + [episode]
    _model.save()
    _cache.clear()
    detail = _analyze(episode_id)
    text = describe_run(episode_id, detail, series_from_tracks(tracks), detail.get("known_class"))
    _memory.add_episode(episode_id, text, "upload", detail["status"], detail["events"])
    return {"episode": episode, "analysis": detail}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(400, "missing filename")
    suffix = Path(file.filename).suffix.lower() or ".mp4"
    if suffix not in {".mp4", ".mov", ".avi"}:
        raise HTTPException(400, "upload an mp4, mov, or avi")
    episode_id = f"upload_{int(time.time())}"
    dest = UPLOAD_DIR / f"{episode_id}{suffix}"
    with dest.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)
    with _lock:
        return _ingest(dest, episode_id)


@app.post("/api/learn")
def learn():
    global _model, _memory
    with _lock:
        _model = learn_model()
        _memory = MemoryStore()
        _cache.clear()
    return _model.public()
