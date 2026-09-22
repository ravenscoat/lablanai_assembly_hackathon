"""
RAG engine: Qdrant vector search + Gemini embeddings.
Ported from navai-voice-agent/src/tools/knowledge_base.py
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

from observability.network_topology import network_request_context, register_service_route

logger = logging.getLogger(__name__)


@dataclass
class RAGResult:
    """Result from a knowledge base search."""

    answer: str
    confidence: float
    source: str = ""
    chunks: list[str] | None = None
    # NAV-214: Per-chunk scores aligned with `chunks`, so callers can surface
    # top-k cross-reference material to the LLM (not just the single
    # highest-scoring answer).
    chunk_scores: list[float] | None = None
    # Curated keyword aliases of the TOP-matched entry (from its Qdrant payload).
    # Lets search_kb rescue a terse in-scope query that embeds just below the
    # confidence floor when it exactly matches an operator-curated alias.
    keywords: list[str] | None = None


class RAGEngine:
    """
    Per-tenant RAG engine:
    - Qdrant for vector search
    - Gemini for embeddings (free tier)
    """

    def __init__(
        self,
        qdrant_url: str | None = None,
    ):
        # .strip() guards against trailing whitespace / CR from CRLF .env files —
        # qdrant-client refuses to parse "http://localhost:6333\r".
        self._qdrant_url = (qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6333")).strip()
        self._qdrant_client: Any | None = None
        self._initialized = False
        self._loop = None

        # Skip the Vertex region probe when an AI Studio API key is configured —
        # _get_embedding will route through the API key path instead.
        if os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
            self._vertex_location = "global"
            register_service_route(
                "gemini_embeddings",
                "https://generativelanguage.googleapis.com",
                provider="google_ai_studio",
                notes="google.genai embedding requests for RAG (AI Studio key)",
            )
        else:
            from utils.vertex_region import resolve_location, vertex_endpoint

            self._vertex_location = resolve_location("gemini-embedding-001")
            register_service_route(
                "gemini_embeddings",
                vertex_endpoint(self._vertex_location),
                provider="vertex-ai",
                notes="google.genai embedding requests for RAG (Vertex)",
            )
        register_service_route("qdrant", self._qdrant_url, provider="qdrant")

    async def _ensure_initialized(self) -> None:
        """Lazy init: connect to Qdrant on first use."""
        import asyncio

        current_loop = asyncio.get_running_loop()
        if self._initialized and self._loop is current_loop:
            return

        self._loop = current_loop
        try:
            from qdrant_client import AsyncQdrantClient

            self._qdrant_client = AsyncQdrantClient(url=self._qdrant_url, timeout=5)
            self._initialized = True
            logger.info(f"Connected to Qdrant at {self._qdrant_url}")
        except Exception as e:
            logger.warning(f"Qdrant unavailable at {self._qdrant_url}: {e}")
            self._initialized = True  # Don't retry

    async def search(
        self,
        *,
        collection: str,
        query: str,
        top_k: int = 3,
    ) -> RAGResult | None:
        """
        Search a Qdrant collection for relevant answers.

        Args:
            collection: Qdrant collection name
            query: User's question
            top_k: Number of results to return
        """
        await self._ensure_initialized()

        if not self._qdrant_client:
            return None

        try:
            # Generate embedding
            embedding = await self._get_embedding(query)
            if not embedding:
                return None

            with network_request_context(
                "qdrant",
                "query_points",
                metadata={"collection": collection, "top_k": top_k},
            ):
                results = await self._qdrant_client.query_points(
                    collection_name=collection,
                    query=embedding,
                    limit=top_k,
                )

            points = results.points if hasattr(results, "points") else results
            if not points:
                return RAGResult(answer="", confidence=0.0)

            # Build answer from top results
            top = points[0]
            answer = top.payload.get("answer", top.payload.get("text", ""))
            source = top.payload.get("source", "")
            chunks = [r.payload.get("answer", r.payload.get("text", "")) for r in points]
            chunk_scores = [float(r.score) for r in points]
            keywords = top.payload.get("keywords") or []

            return RAGResult(
                answer=answer,
                confidence=top.score,
                source=source,
                chunks=chunks,
                chunk_scores=chunk_scores,
                keywords=keywords,
            )

        except Exception as e:
            logger.error(f"RAG search error: {e}")
            return None

    async def warmup(self, collection: str) -> bool:
        """Verify collection exists and is accessible."""
        await self._ensure_initialized()
        try:
            if self._qdrant_client:
                with network_request_context(
                    "qdrant",
                    "get_collection",
                    metadata={"collection": collection},
                ):
                    info = await self._qdrant_client.get_collection(collection)
                logger.info(f"KB warmup: collection={collection}, " f"points={info.points_count}")
                return True
        except Exception as e:
            logger.warning(f"KB warmup failed for {collection}: {e}")
        return False

    async def _get_embedding(self, text: str) -> list[float] | None:
        """Generate embedding using Gemini.

        Prefers the Google AI Studio API key path (``GOOGLE_API_KEY`` /
        ``GEMINI_API_KEY``) when set, otherwise falls back to Vertex AI with
        Application Default Credentials.
        """
        try:
            import os

            from google import genai
            from google.genai.types import EmbedContentConfig

            from utils.vertex_region import PROJECT

            api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if api_key:
                client = genai.Client(api_key=api_key)
            else:
                client = genai.Client(
                    vertexai=True,
                    project=PROJECT,
                    location=self._vertex_location,
                )

            with network_request_context(
                "gemini_embeddings",
                "embed_content",
                metadata={
                    "model": "gemini-embedding-001",
                    "text_length": len(text),
                },
            ):
                result = client.models.embed_content(
                    model="gemini-embedding-001",
                    contents=text,
                    config=EmbedContentConfig(output_dimensionality=768),
                )
            return result.embeddings[0].values
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return None
