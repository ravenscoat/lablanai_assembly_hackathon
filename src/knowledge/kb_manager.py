"""
KB Manager: manages per-tenant knowledge base instances.
Handles lazy warmup and search delegation.
"""

from __future__ import annotations

import logging

from .rag_engine import RAGEngine, RAGResult

logger = logging.getLogger(__name__)


class KBManager:
    """Manages RAG engine per tenant. Lazy initialization + warmup."""

    def __init__(self, rag_engine: RAGEngine | None = None):
        self._engine = rag_engine or RAGEngine()
        self._warmed: set[str] = set()

    async def search(
        self,
        tenant_id: str,
        query: str,
        collection: str,
        top_k: int = 3,
    ) -> RAGResult | None:
        """Search tenant's knowledge base."""
        if not collection:
            return None

        # Warmup on first search
        if tenant_id not in self._warmed:
            await self.warmup(tenant_id, collection)

        return await self._engine.search(
            collection=collection,
            query=query,
            top_k=top_k,
        )

    async def warmup(self, tenant_id: str, collection: str) -> bool:
        """Warm up a tenant's KB collection.

        Idempotent per (tenant_id, collection): a second call from the same
        process short-circuits so we don't pay the ~1.3s Qdrant get_collection
        round-trip on every inbound call.
        """
        if not collection:
            return False

        if tenant_id in self._warmed:
            return True

        success = await self._engine.warmup(collection)
        if success:
            self._warmed.add(tenant_id)
            logger.info(f"KB warmed up: tenant={tenant_id}, collection={collection}")
        return success

    def is_warmed(self, tenant_id: str) -> bool:
        """Check if tenant's KB has been warmed up."""
        return tenant_id in self._warmed
