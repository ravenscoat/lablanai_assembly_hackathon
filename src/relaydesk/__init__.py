"""RelayDesk demo domain and persistent case memory."""

import os

from .postgres_store import PostgresRelayDeskStore
from .semantic_memory import OllamaEmbedder, PostgresCaseMemory
from .store import RelayDeskStore


def create_store() -> RelayDeskStore:
    """Use PostgreSQL when configured; retain SQLite for local tests and fallback."""
    dsn = os.getenv("DATABASE_URL", "").strip()
    if dsn:
        return PostgresRelayDeskStore(dsn)
    return RelayDeskStore(os.getenv("RELAYDESK_DB_PATH", "data/relaydesk.db"))


__all__ = [
    "OllamaEmbedder",
    "PostgresCaseMemory",
    "PostgresRelayDeskStore",
    "RelayDeskStore",
    "create_store",
]
