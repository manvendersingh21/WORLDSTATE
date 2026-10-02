"""External service adapters. Every one no-ops cleanly when unconfigured."""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np


class TextEmbedder:
    """Event-text embeddings.

    Uses sentence-transformers when WORLDSTATE_TEXT_ENCODER asks for it and the
    package is installed. Otherwise a deterministic hashing embedder so retrieval
    works fully offline.
    """

    def __init__(self) -> None:
        self.mode = "hashing"
        self._model = None
        self.dim = 256
        preference = os.environ.get("WORLDSTATE_TEXT_ENCODER", "auto")
        model_name = os.environ.get("WORLDSTATE_ST_MODEL", "all-MiniLM-L6-v2")
        if preference != "hashing":
            try:
                from sentence_transformers import SentenceTransformer

                cached = _sentence_transformer_cached(model_name)
                if preference == "sentence-transformers" or cached:
                    self._model = SentenceTransformer(model_name)
                    self.mode = "sentence-transformers"
                    self.dim = int(self._model.get_sentence_embedding_dimension())
            except Exception:
                self._model = None
                self.mode = "hashing"
        if self._model is None:
            from sklearn.feature_extraction.text import HashingVectorizer

            # Character n-grams so "grasps" still meets "grasp" and "moved" meets "move".
            self.dim = 1024
            self._hash = HashingVectorizer(
                n_features=self.dim,
                alternate_sign=False,
                norm="l2",
                analyzer="char_wb",
                ngram_range=(3, 5),
            )

    def embed(self, texts: list[str]) -> np.ndarray:
        cleaned = [text if text and text.strip() else "empty" for text in texts]
        if self._model is not None:
            vectors = self._model.encode(cleaned, normalize_embeddings=True)
            return np.asarray(vectors, dtype=np.float32)
        matrix = self._hash.transform(cleaned).astype(np.float32)
        return np.asarray(matrix.toarray(), dtype=np.float32)


def _sentence_transformer_cached(model_name: str) -> bool:
    slug = "models--" + model_name.replace("/", "--")
    roots = [
        Path.home() / ".cache" / "huggingface" / "hub",
        Path.home() / ".cache" / "torch" / "sentence_transformers",
    ]
    for root in roots:
        if not root.exists():
            continue
        for path in root.iterdir():
            if model_name.replace("/", "_") in path.name or slug in path.name:
                return True
    return False


class VectorIndex:
    """FAISS inner-product index, with a NumPy fallback of the same API."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.ids: list[str] = []
        self.matrix = np.zeros((0, dim), dtype=np.float32)
        self._faiss = None
        self.backend = "numpy"
        try:
            import faiss

            self._faiss_mod = faiss
            self.backend = "faiss"
        except Exception:
            self._faiss_mod = None
            self.backend = "numpy"

    def add(self, id_: str, vector: np.ndarray) -> None:
        vec = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        vec = _l2(vec)
        if id_ in self.ids:
            self.matrix[self.ids.index(id_)] = vec
        else:
            self.ids.append(id_)
            self.matrix = vec if self.matrix.size == 0 else np.vstack([self.matrix, vec])
        self._rebuild()

    def _rebuild(self) -> None:
        if self._faiss_mod is None or self.matrix.size == 0:
            self._faiss = None
            return
        index = self._faiss_mod.IndexFlatIP(self.dim)
        index.add(np.ascontiguousarray(self.matrix))
        self._faiss = index

    def search(self, vector: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
        if not self.ids:
            return []
        query = _l2(np.asarray(vector, dtype=np.float32).reshape(1, -1))
        k = min(k, len(self.ids))
        if self._faiss is not None:
            scores, indices = self._faiss.search(np.ascontiguousarray(query), k)
            out = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0:
                    continue
                out.append((self.ids[int(idx)], float(score)))
            return out
        sims = (self.matrix @ query.T).ravel()
        order = np.argsort(-sims)[:k]
        return [(self.ids[int(i)], float(sims[int(i)])) for i in order]


def _l2(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return matrix / norms


def credential_report() -> dict:
    """Lengths only. Never include secret values."""
    names = (
        "COSMOS_API_KEY",
        "GPU_BEARER_TOKEN",
        "NVIDIA_API_KEY",
        "NGC_API_KEY",
        "NV_API_KEY",
        "NVIDIA_BASE_URL",
        "COSMOS_BASE_URL",
        "NVIDIA_API_BASE",
        "NVIDIA_COSMOS_MODEL",
        "COSMOS_MODEL",
    )
    report = {}
    for name in names:
        raw = os.environ.get(name)
        report[name] = {
            "set": bool(raw and raw.strip()),
            "length": len(raw.strip()) if raw and raw.strip() else 0,
        }
    return report


def resolve_nvidia() -> dict:
    key = ""
    key_source = None
    # COSMOS_API_KEY / GPU_BEARER_TOKEN: bearer for a self-hosted or event Cosmos endpoint.
    for name in ("COSMOS_API_KEY", "GPU_BEARER_TOKEN", "NVIDIA_API_KEY", "NGC_API_KEY", "NV_API_KEY"):
        raw = os.environ.get(name, "").strip()
        if raw:
            key = raw
            key_source = name
            break
    base = "https://integrate.api.nvidia.com/v1"
    base_source = "default"
    for name in ("COSMOS_BASE_URL", "NVIDIA_BASE_URL", "NVIDIA_API_BASE"):
        raw = os.environ.get(name, "").strip()
        if raw:
            base = raw.rstrip("/")
            base_source = name
            break
    custom = base_source != "default"
    if custom and not base.endswith("/v1"):
        base = base + "/v1"
    model = os.environ.get("NVIDIA_COSMOS_MODEL", "").strip() or os.environ.get("COSMOS_MODEL", "").strip()
    if not model:
        model = "nvidia/cosmos3-nano-reasoner"
    return {
        "api_key": key,
        "key_source": key_source,
        "base": base,
        "base_source": base_source,
        "custom_base": custom,
        "model": model,
    }


COSMOS_FALLBACK_MODELS = ["nvidia/cosmos-reason2-8b"]


class CosmosReasonAdapter:
    """Cosmos Reason as the event extractor and cluster namer.

    Hosted default is integrate.api.nvidia.com, model nvidia/cosmos3-nano-reasoner.
    A COSMOS_BASE_URL or NVIDIA_BASE_URL switches to that NIM. Any failure returns
    None so the kinematic extractor remains the offline path.
    """

    def __init__(self) -> None:
        cfg = resolve_nvidia()
        self.api_key = cfg["api_key"]
        self.base = cfg["base"]
        self.model = cfg["model"]
        self.custom_base = cfg["custom_base"]
        self.key_source = cfg["key_source"]
        # Only used when the model was not pinned by NVIDIA_COSMOS_MODEL / COSMOS_MODEL.
        pinned = bool(os.environ.get("NVIDIA_COSMOS_MODEL", "").strip() or os.environ.get("COSMOS_MODEL", "").strip())
        self._fallbacks = [] if pinned or self.custom_base else [m for m in COSMOS_FALLBACK_MODELS if m != self.model]
        self.last_status: dict = {
            "attempted": 0,
            "succeeded": 0,
            "failed": 0,
            "skipped_reason": None if self.api_key else "no COSMOS_API_KEY / NVIDIA_API_KEY in the environment",
        }
        if self.custom_base and not pinned and self.api_key:
            self.model = self._discover_model() or "nvidia/cosmos3-reason"

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _discover_model(self) -> str | None:
        """A NIM / vLLM endpoint names its model at /v1/models (the event serves nvidia/cosmos3-reason)."""
        request = urllib.request.Request(
            f"{self.base}/models", headers={"Authorization": f"Bearer {self.api_key}"}
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                data = json.loads(response.read().decode()).get("data") or []
            ids = [item.get("id") for item in data if item.get("id")]
            cosmos = [i for i in ids if "cosmos" in i.lower()]
            return (cosmos or ids or [None])[0]
        except Exception as exc:
            self.last_status["last_error"] = _redact(f"model discovery failed: {exc}", self.api_key)[:300]
            return None

    def refine_events(self, video_path: Path, events: list[dict], summary: str) -> list[dict] | None:
        if not self.available or not events:
            return None
        prompt = (
            "Answer the question using the following format: <think>your reasoning</think>\n"
            "Then reply with a JSON array only, same length and order as the segments.\n"
            "You are the event extractor for a physical process. The segment times were "
            "discovered from motion before any label list existed. Watch the video and fill "
            "each event. Schema: "
            '{"time","end_time","actors","action","relations","state_change"}.\n'
            "action is a short physical description. Do not invent a segment that is not listed.\n\n"
            f"Summary:\n{summary}\n\nSegments:\n{json.dumps(events)}"
        )
        text = self._complete(prompt, video_path=video_path)
        parsed = _extract_json(text or "")
        if not isinstance(parsed, list) or len(parsed) != len(events):
            return None
        merged = []
        for original, updated in zip(events, parsed):
            if not isinstance(updated, dict) or not updated.get("action"):
                return None
            item = dict(original)
            item["action"] = str(updated["action"])
            if isinstance(updated.get("actors"), list):
                item["actors"] = updated["actors"]
            if isinstance(updated.get("relations"), dict):
                item["relations"] = updated["relations"]
            item["source"] = "cosmos-reason"
            merged.append(item)
        return merged

    def narrate_run(self, video_path: Path, analysis: dict) -> str | None:
        """Plain-language account of one run. The graph already decided normal vs novel."""
        if not self.available:
            return None
        facts = {
            "status": analysis.get("status"),
            "first_divergence_s": analysis.get("first_divergence_s"),
            "expected_transition": analysis.get("expected"),
            "observed_transition": analysis.get("observed"),
            "support": (analysis.get("support") or {}).get("text"),
            "learned_path": [step["name"] for step in analysis.get("expected_path", [])],
        }
        prompt = (
            "Answer the question using the following format: <think>your reasoning</think>\n"
            "Then write 2-3 plain sentences, no JSON.\n"
            "Watch this robot pick-and-place video. A world model learned the process graph from "
            "unlabeled reference runs and produced the facts below; do not change its verdict. "
            "Describe what physically happens in this run, and if it diverges, what the gripper and "
            "the object do at the divergence time.\n"
            f"Facts: {json.dumps(facts)}"
        )
        text = self._complete(prompt, video_path=video_path)
        if not text:
            return None
        if "</think>" in text:
            text = text.split("</think>", 1)[1]
        text = text.strip()
        return text[:900] or None

    def name_clusters(self, video_path: Path, clusters: list[dict]) -> dict[str, str] | None:
        """Name discovered clusters after they exist. Returns {id: short name}."""
        if not self.available or not clusters:
            return None
        brief = [
            {"id": c["id"], "motion": c.get("full_name"), "mean_time_s": c.get("mean_t")}
            for c in clusters
        ]
        prompt = (
            "Answer the question using the following format: <think>your reasoning</think>\n"
            "These clusters were discovered by clustering motion, with no state names given. "
            "Watch the video and name each cluster with a 2-4 word physical-process phrase. "
            "Reply with a JSON object mapping id to name, and nothing else.\n"
            f"{json.dumps(brief)}"
        )
        text = self._complete(prompt, video_path=video_path)
        if not text:
            return None
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end < start:
            return None
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, dict):
            return None
        out = {}
        for cluster in clusters:
            name = parsed.get(cluster["id"])
            if isinstance(name, str) and name.strip():
                out[cluster["id"]] = name.strip()[:48]
        return out or None

    def _complete(self, prompt: str, video_path: Path | None = None) -> str | None:
        self.last_status["attempted"] += 1
        content: list = [{"type": "text", "text": prompt}]
        if video_path is not None and video_path.exists():
            raw = video_path.read_bytes()
            # Keep the request bounded. Longer real clips are sampled down by the NIM fps flag.
            if len(raw) > 8_000_000:
                self.last_status["failed"] += 1
                self.last_status["last_error"] = "video too large for inline video_url"
                return None
            b64 = base64.b64encode(raw).decode()
            content.append(
                {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{b64}"}}
            )
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.2,
            "max_tokens": 1024,
            "media_io_kwargs": {"video": {"fps": 4.0}},
        }
        request = urllib.request.Request(
            f"{self.base}/chat/completions",
            data=json.dumps(body).encode(),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                payload = json.loads(response.read().decode())
            text = payload["choices"][0]["message"]["content"]
            self.last_status["succeeded"] += 1
            self.last_status["last_error"] = None
            self.last_status["model"] = self.model
            return text
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:300]
            if exc.code == 404 and self._fallbacks:
                # The configured name is not served here; try the next catalogued Cosmos Reason model.
                self.model = self._fallbacks.pop(0)
                self.last_status["attempted"] -= 1
                return self._complete(prompt, video_path)
            self.last_status["failed"] += 1
            message = _redact(f"HTTP {exc.code} from {self.model}: {detail}", self.api_key)
            self.last_status["last_error"] = message[:300]
            return None
        except Exception as exc:
            self.last_status["failed"] += 1
            message = _redact(str(exc), self.api_key)
            self.last_status["last_error"] = message[:300]
            return None


class VastMemoryAdapter:
    """Optional remote memory. Local SQLite + FAISS remain the source of truth."""

    def __init__(self) -> None:
        self.url = os.environ.get("VAST_API_URL", "").rstrip("/")
        self.api_key = os.environ.get("VAST_API_KEY", "").strip()

    @property
    def enabled(self) -> bool:
        return bool(self.url)

    def upsert_episode(self, episode_id: str, vector: list[float], metadata: dict) -> dict:
        if not self.enabled:
            return {"backend": "local"}
        body = {
            "id": episode_id,
            "vector": vector,
            "metadata": metadata,
        }
        request = urllib.request.Request(
            f"{self.url}/v1/episodes",
            data=json.dumps(body).encode(),
            headers={
                "Authorization": f"Bearer {self.api_key}" if self.api_key else "",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                detail = response.read().decode()[:500]
            return {"backend": "vast", "status": "synced", "detail": detail}
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            return {"backend": "local", "status": "error", "detail": str(exc)}


class RecoveryVideoAdapter:
    """Cosmos Predict-style video generation stub. Never called for real pixels here."""

    def __init__(self) -> None:
        self.endpoint = os.environ.get("COSMOS_PREDICT_URL", "").strip()

    def plan(self, prompt: str) -> dict:
        return {
            "provider": "nvidia-cosmos-predict",
            "status": "stub",
            "endpoint": self.endpoint or None,
            "prompt": prompt,
            "note": (
                "Recovery video generation is stubbed. With COSMOS_PREDICT_URL set, "
                "this prompt is what would be submitted to synthesize the recovery clip."
            ),
        }


def _sample_jpegs(video_path: Path, count: int) -> list[bytes]:
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        cap.release()
        return []
    indexes = np.linspace(0, total - 1, count).astype(int)
    blobs = []
    for index in indexes:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = cap.read()
        if not ok:
            continue
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if ok:
            blobs.append(buf.tobytes())
    cap.release()
    return blobs


def _redact(text: str, secret: str) -> str:
    if secret and secret in text:
        return text.replace(secret, "[redacted]")
    return text


def _extract_json(text: str):
    start = text.find("[")
    end = text.rfind("]")
    if start < 0 or end < start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
