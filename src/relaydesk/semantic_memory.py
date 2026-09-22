"""pgvector-backed retrieval of similar historical support cases."""

from __future__ import annotations

import json
import os
from typing import Protocol

import httpx

from .postgres_store import PostgresRelayDeskStore


class Embedder(Protocol):
    def embed(self, text: str) -> list[float]: ...


class OllamaEmbedder:
    def __init__(self, model: str = "qwen3-embedding:0.6b") -> None:
        self.model = model
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

    def embed(self, text: str) -> list[float]:
        response = httpx.post(
            f"{self.base_url}/api/embed",
            json={"model": self.model, "input": text},
            timeout=30,
        )
        response.raise_for_status()
        vector = response.json()["embeddings"][0]
        if len(vector) != 1024:
            raise ValueError(f"Expected 1024 embedding dimensions, received {len(vector)}")
        return vector


def case_memory_text(packet: dict) -> str:
    return json.dumps(
        {
            "confirmed_facts": packet.get("confirmed_facts", []),
            "completed_actions": packet.get("completed_actions", []),
            "evidence": packet.get("evidence", []),
            "unresolved_tasks": packet.get("unresolved_tasks", []),
            "specialist": packet.get("current_specialist"),
        },
        ensure_ascii=False,
        sort_keys=True,
    )


class PostgresCaseMemory:
    def __init__(self, store: PostgresRelayDeskStore, embedder: Embedder | None = None) -> None:
        self.store = store
        self.embedder = embedder or OllamaEmbedder()

    @staticmethod
    def _literal(vector: list[float]) -> str:
        return "[" + ",".join(f"{value:.8f}" for value in vector) + "]"

    def index_case(self, case_id: str) -> dict:
        packet = self.store.get_case(case_id)
        if packet.get("found") is False:
            return packet
        text = case_memory_text(packet)
        vector = self._literal(self.embedder.embed(text))
        with self.store._connect() as db:
            db.execute(
                """INSERT INTO case_memory (case_id, memory_text, embedding)
                   VALUES (?, ?, ?::vector)
                   ON CONFLICT (case_id) DO UPDATE SET memory_text=EXCLUDED.memory_text,
                   embedding=EXCLUDED.embedding, updated_at=NOW()""",
                (case_id, text, vector),
            )
        return {"indexed": True, "case_id": case_id, "memory_text": text}

    def search(self, query: str, limit: int = 3) -> list[dict]:
        vector = self._literal(self.embedder.embed(query))
        with self.store._connect() as db:
            rows = db.execute(
                """SELECT case_id, memory_text,
                          1 - (embedding <=> ?::vector) AS similarity
                   FROM case_memory ORDER BY embedding <=> ?::vector LIMIT ?""",
                (vector, vector, limit),
            ).fetchall()
        return [
            {
                "case_id": row["case_id"],
                "memory_text": row["memory_text"],
                "similarity": float(row["similarity"]),
            }
            for row in rows
        ]
