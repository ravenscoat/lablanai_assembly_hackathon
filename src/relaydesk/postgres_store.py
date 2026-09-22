"""PostgreSQL implementation of RelayDesk durable business and case memory."""

from __future__ import annotations

from typing import Any

import psycopg
from psycopg.rows import dict_row

from .store import RelayDeskStore


class _CompatConnection:
    """Translate the small qmark query surface shared with the SQLite repository."""

    def __init__(self, connection: psycopg.Connection) -> None:
        self.connection = connection

    def execute(self, query: str, params: tuple[Any, ...] = ()):
        return self.connection.execute(query.replace("?", "%s"), params)

    def __enter__(self) -> "_CompatConnection":
        self.connection.__enter__()
        return self

    def __exit__(self, *args: object) -> None:
        self.connection.__exit__(*args)


class PostgresRelayDeskStore(RelayDeskStore):
    """Same domain contract as SQLite, backed by PostgreSQL transactions."""

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        super().__init__(path=":postgres:")

    def _connect(self) -> _CompatConnection:
        return _CompatConnection(psycopg.connect(self.dsn, row_factory=dict_row))

    def _initialize(self) -> None:
        with self._connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS customers (
                    id TEXT PRIMARY KEY, email TEXT UNIQUE NOT NULL, name TEXT NOT NULL
                )""")
            db.execute("""CREATE TABLE IF NOT EXISTS charges (
                    id TEXT PRIMARY KEY, customer_id TEXT NOT NULL REFERENCES customers(id),
                    operation_id TEXT NOT NULL, amount_cents INTEGER NOT NULL,
                    status TEXT NOT NULL DEFAULT 'captured'
                )""")
            db.execute("""CREATE TABLE IF NOT EXISTS memberships (
                    customer_id TEXT NOT NULL REFERENCES customers(id), project_id TEXT NOT NULL,
                    role TEXT NOT NULL, active BOOLEAN NOT NULL DEFAULT TRUE,
                    PRIMARY KEY(customer_id, project_id)
                )""")
            db.execute("""CREATE TABLE IF NOT EXISTS cases (
                    id TEXT PRIMARY KEY, customer_id TEXT, specialist TEXT NOT NULL,
                    confirmed_facts JSONB NOT NULL, completed_actions JSONB NOT NULL,
                    evidence JSONB NOT NULL, unresolved_tasks JSONB NOT NULL
                )""")
            db.execute("""CREATE TABLE IF NOT EXISTS action_keys (
                    action_key TEXT PRIMARY KEY, result JSONB NOT NULL
                )""")
            db.execute("""CREATE TABLE IF NOT EXISTS case_memory (
                    case_id TEXT PRIMARY KEY REFERENCES cases(id) ON DELETE CASCADE,
                    memory_text TEXT NOT NULL,
                    embedding vector(1024) NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )""")
            db.execute("""CREATE TABLE IF NOT EXISTS runtime_events (
                    id TEXT PRIMARY KEY, session_id TEXT NOT NULL, case_id TEXT,
                    event_type TEXT NOT NULL, actor TEXT NOT NULL,
                    payload JSONB NOT NULL, created_at TIMESTAMPTZ NOT NULL
                )""")
            db.execute(
                """INSERT INTO customers VALUES (?, ?, ?)
                   ON CONFLICT (id) DO NOTHING""",
                ("cust_demo", "alex@relaydesk.demo", "Alex Morgan"),
            )
            for values in (
                ("charge_1", "cust_demo", "purchase_demo_100", 2500),
                ("charge_2", "cust_demo", "purchase_demo_100", 2500),
            ):
                db.execute(
                    """INSERT INTO charges (id, customer_id, operation_id, amount_cents)
                       VALUES (?, ?, ?, ?) ON CONFLICT (id) DO NOTHING""",
                    values,
                )
            db.execute(
                """INSERT INTO memberships VALUES (?, ?, ?, ?)
                   ON CONFLICT (customer_id, project_id) DO NOTHING""",
                ("cust_demo", "project_atlas", "viewer", False),
            )

    def get_case(self, case_id: str) -> dict[str, Any]:
        with self._connect() as db:
            row = db.execute("SELECT * FROM cases WHERE id=?", (case_id,)).fetchone()
        if not row:
            return {"found": False, "case_id": case_id}
        return {
            "case_id": row["id"],
            "customer_id": row["customer_id"],
            "current_specialist": row["specialist"],
            "confirmed_facts": self._decode_json(row["confirmed_facts"]),
            "completed_actions": self._decode_json(row["completed_actions"]),
            "evidence": self._decode_json(row["evidence"]),
            "unresolved_tasks": self._decode_json(row["unresolved_tasks"]),
        }
