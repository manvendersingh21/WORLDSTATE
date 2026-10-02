"""Episodic and semantic memory: SQLite plus a FAISS (or NumPy) index."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

import numpy as np

from worldstate.adapters import TextEmbedder, VastMemoryAdapter, VectorIndex
from worldstate.config import MEMORY_DB, VECTOR_PATH


class MemoryStore:
    def __init__(self) -> None:
        self.embedder = TextEmbedder()
        self.index = VectorIndex(self.embedder.dim)
        self.vast = VastMemoryAdapter()
        self.db = sqlite3.connect(MEMORY_DB, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS episodes (
                id TEXT PRIMARY KEY,
                role TEXT,
                kind TEXT,
                transcript TEXT,
                updated_at TEXT
            );
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                episode_id TEXT,
                t REAL,
                action TEXT,
                payload TEXT
            );
            CREATE TABLE IF NOT EXISTS rules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version INTEGER,
                text TEXT
            );
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT
            );
            """
        )
        self._load_index()

    def _load_index(self) -> None:
        if not VECTOR_PATH.exists():
            return
        blob = np.load(VECTOR_PATH, allow_pickle=False)
        matrix = blob["matrix"]
        ids = blob["ids"].tolist()
        if matrix.ndim != 2 or matrix.shape[1] != self.embedder.dim or not len(ids):
            return
        self.index.matrix = matrix.astype(np.float32)
        self.index.ids = [str(i) for i in ids]
        self.index._rebuild()

    def _persist_index(self) -> None:
        np.savez(
            VECTOR_PATH,
            matrix=self.index.matrix.astype(np.float32),
            ids=np.array(self.index.ids),
        )

    def rebuild(self, transcripts: dict[str, str], episodes: list[dict]) -> None:
        self.db.execute("DELETE FROM episodes")
        self.db.execute("DELETE FROM events")
        meta = {ep["id"]: ep for ep in episodes}
        self.index = VectorIndex(self.embedder.dim)
        now = datetime.now(timezone.utc).isoformat()
        ids = list(transcripts)
        if not ids:
            self.db.commit()
            return
        vectors = self.embedder.embed([transcripts[i] for i in ids])
        for episode_id, vector in zip(ids, vectors):
            ep = meta.get(episode_id, {})
            self.index.add(episode_id, vector)
            self.db.execute(
                "INSERT OR REPLACE INTO episodes (id, role, kind, transcript, updated_at) VALUES (?,?,?,?,?)",
                (episode_id, ep.get("role"), ep.get("kind"), transcripts[episode_id], now),
            )
            self.vast.upsert_episode(
                episode_id,
                vector.tolist(),
                {"role": ep.get("role"), "kind": ep.get("kind")},
            )
        self.db.commit()
        self._persist_index()

    def add_episode(self, episode_id: str, transcript: str, role: str, kind: str, events: list[dict]) -> None:
        vector = self.embedder.embed([transcript])[0]
        now = datetime.now(timezone.utc).isoformat()
        self.db.execute(
            "INSERT OR REPLACE INTO episodes (id, role, kind, transcript, updated_at) VALUES (?,?,?,?,?)",
            (episode_id, role, kind, transcript, now),
        )
        self.db.execute("DELETE FROM events WHERE episode_id = ?", (episode_id,))
        for event in events:
            self.db.execute(
                "INSERT INTO events (episode_id, t, action, payload) VALUES (?,?,?,?)",
                (episode_id, event.get("time"), event.get("action"), json.dumps(event)),
            )
        self.db.commit()
        self.index.add(episode_id, vector)
        self._persist_index()
        self.vast.upsert_episode(episode_id, vector.tolist(), {"role": role, "kind": kind})

    def add_rule(self, version: int, text: str) -> None:
        self.db.execute("INSERT INTO rules (version, text) VALUES (?,?)", (version, text))
        self.db.commit()

    def search(self, query: str, k: int = 5) -> list[dict]:
        if not self.index.ids:
            return []
        vector = self.embedder.embed([query])[0]
        hits = []
        for episode_id, score in self.index.search(vector, k):
            row = self.db.execute("SELECT * FROM episodes WHERE id = ?", (episode_id,)).fetchone()
            transcript = row["transcript"] if row else ""
            hits.append(
                {
                    "id": episode_id,
                    "score": round(float(score), 4),
                    "role": row["role"] if row else None,
                    "kind": row["kind"] if row else None,
                    "snippet": transcript[:220],
                }
            )
        return hits

    def stats(self) -> dict:
        episodes = self.db.execute("SELECT COUNT(*) AS n FROM episodes").fetchone()["n"]
        rules = self.db.execute("SELECT COUNT(*) AS n FROM rules").fetchone()["n"]
        return {
            "episodes": episodes,
            "rules": rules,
            "embedder": self.embedder.mode,
            "vector_index": self.index.backend,
            "vast": self.vast.enabled,
        }
