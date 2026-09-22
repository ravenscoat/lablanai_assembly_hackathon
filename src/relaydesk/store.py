"""Persistent, deterministic business store for the RelayDesk demo."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class RelayDeskStore:
    """Small SQLite repository with idempotent, independently verified actions."""

    def __init__(self, path: str | Path = "data/relaydesk.db") -> None:
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @staticmethod
    def _decode_json(value: Any) -> Any:
        return value if isinstance(value, (list, dict)) else json.loads(value)

    @staticmethod
    def _normalize_project_id(project_id: str) -> str:
        normalized = project_id.strip().lower().replace(" ", "_").replace("-", "_")
        return "project_atlas" if normalized in {"atlas", "project_atlas"} else normalized

    def _initialize(self) -> None:
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS customers (
                    id TEXT PRIMARY KEY, email TEXT UNIQUE NOT NULL, name TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS charges (
                    id TEXT PRIMARY KEY, customer_id TEXT NOT NULL, operation_id TEXT NOT NULL,
                    amount_cents INTEGER NOT NULL, status TEXT NOT NULL DEFAULT 'captured',
                    FOREIGN KEY(customer_id) REFERENCES customers(id)
                );
                CREATE TABLE IF NOT EXISTS memberships (
                    customer_id TEXT NOT NULL, project_id TEXT NOT NULL, role TEXT NOT NULL,
                    active INTEGER NOT NULL DEFAULT 1,
                    PRIMARY KEY(customer_id, project_id),
                    FOREIGN KEY(customer_id) REFERENCES customers(id)
                );
                CREATE TABLE IF NOT EXISTS cases (
                    id TEXT PRIMARY KEY, customer_id TEXT, specialist TEXT NOT NULL,
                    confirmed_facts TEXT NOT NULL, completed_actions TEXT NOT NULL,
                    evidence TEXT NOT NULL, unresolved_tasks TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS action_keys (
                    action_key TEXT PRIMARY KEY, result TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS runtime_events (
                    id TEXT PRIMARY KEY, session_id TEXT NOT NULL, case_id TEXT,
                    event_type TEXT NOT NULL, actor TEXT NOT NULL,
                    payload TEXT NOT NULL, created_at TEXT NOT NULL
                );
                """)
            db.execute(
                "INSERT OR IGNORE INTO customers VALUES (?, ?, ?)",
                ("cust_demo", "alex@relaydesk.demo", "Alex Morgan"),
            )
            db.executemany(
                "INSERT OR IGNORE INTO charges VALUES (?, ?, ?, ?, 'captured')",
                [
                    ("charge_1", "cust_demo", "purchase_demo_100", 2500),
                    ("charge_2", "cust_demo", "purchase_demo_100", 2500),
                ],
            )
            db.execute(
                "INSERT OR IGNORE INTO memberships VALUES (?, ?, ?, ?)",
                ("cust_demo", "project_atlas", "viewer", 0),
            )

    def identify_customer(self, email: str) -> dict[str, Any]:
        with self._connect() as db:
            row = db.execute(
                "SELECT id, email, name FROM customers WHERE lower(email)=lower(?)", (email,)
            ).fetchone()
        return dict(row) if row else {"found": False, "email": email}

    def inspect_billing(self, customer_id: str) -> dict[str, Any]:
        with self._connect() as db:
            rows = db.execute(
                """SELECT operation_id, COUNT(*) AS charge_count,
                          SUM(amount_cents) AS total_cents
                   FROM charges WHERE customer_id=? AND status='captured'
                   GROUP BY operation_id ORDER BY operation_id""",
                (customer_id,),
            ).fetchall()
        operations = [dict(row) for row in rows]
        duplicates = [row for row in operations if row["charge_count"] > 1]
        return {
            "customer_id": customer_id,
            "operations": operations,
            "duplicate_operations": duplicates,
            "verified": True,
        }

    def refund_duplicate(self, customer_id: str, operation_id: str) -> dict[str, Any]:
        action_key = f"refund:{customer_id}:{operation_id}"
        with self._lock, self._connect() as db:
            prior = db.execute(
                "SELECT result FROM action_keys WHERE action_key=?", (action_key,)
            ).fetchone()
            if prior:
                result = self._decode_json(prior["result"])
                result["idempotent_replay"] = True
                return result

            charges = db.execute(
                """SELECT id, amount_cents FROM charges
                   WHERE customer_id=? AND operation_id=? AND status='captured'
                   ORDER BY id""",
                (customer_id, operation_id),
            ).fetchall()
            if len(charges) < 2:
                return {"ok": False, "reason": "No verified duplicate charge exists"}

            duplicate = charges[-1]
            db.execute("UPDATE charges SET status='refunded' WHERE id=?", (duplicate["id"],))
            remaining = db.execute(
                """SELECT COUNT(*) AS remaining FROM charges
                   WHERE customer_id=? AND operation_id=? AND status='captured'""",
                (customer_id, operation_id),
            ).fetchone()["remaining"]
            result = {
                "ok": remaining == 1,
                "refunded_charge_id": duplicate["id"],
                "refund_cents": duplicate["amount_cents"],
                "captured_charges_after": remaining,
                "verified": remaining == 1,
                "idempotent_replay": False,
            }
            db.execute("INSERT INTO action_keys VALUES (?, ?)", (action_key, json.dumps(result)))
            return result

    def inspect_permissions(self, customer_id: str, project_id: str) -> dict[str, Any]:
        project_id = self._normalize_project_id(project_id)
        with self._connect() as db:
            row = db.execute(
                """SELECT customer_id, project_id, role, active FROM memberships
                   WHERE customer_id=? AND project_id=?""",
                (customer_id, project_id),
            ).fetchone()
        if not row:
            return {"found": False, "customer_id": customer_id, "project_id": project_id}
        result = dict(row)
        result["active"] = bool(result["active"])
        result["verified"] = True
        return result

    def restore_access(self, customer_id: str, project_id: str) -> dict[str, Any]:
        project_id = self._normalize_project_id(project_id)
        action_key = f"restore:{customer_id}:{project_id}"
        with self._lock, self._connect() as db:
            prior = db.execute(
                "SELECT result FROM action_keys WHERE action_key=?", (action_key,)
            ).fetchone()
            if prior:
                result = self._decode_json(prior["result"])
                result["idempotent_replay"] = True
                return result
            changed = db.execute(
                "UPDATE memberships SET active=TRUE WHERE customer_id=? AND project_id=?",
                (customer_id, project_id),
            ).rowcount
            active = db.execute(
                """SELECT active FROM memberships
                   WHERE customer_id=? AND project_id=?""",
                (customer_id, project_id),
            ).fetchone()
            result = {
                "ok": bool(changed and active and active["active"]),
                "customer_id": customer_id,
                "project_id": project_id,
                "active": bool(active["active"]) if active else False,
                "verified": bool(active and active["active"]),
                "idempotent_replay": False,
            }
            db.execute("INSERT INTO action_keys VALUES (?, ?)", (action_key, json.dumps(result)))
            return result

    def create_case(self, customer_id: str | None, issue: str) -> dict[str, Any]:
        case_id = f"case_{uuid.uuid4().hex[:10]}"
        with self._connect() as db:
            db.execute(
                "INSERT INTO cases VALUES (?, ?, 'front_desk', ?, '[]', '[]', ?)",
                (case_id, customer_id, json.dumps([issue]), json.dumps([issue])),
            )
        return self.get_case(case_id)

    def route_case(self, case_id: str, specialist: str, confirmed_fact: str = "") -> dict[str, Any]:
        allowed = {"front_desk", "billing", "subscriptions", "permissions"}
        if specialist not in allowed:
            return {"ok": False, "reason": f"Unknown specialist: {specialist}"}
        with self._lock, self._connect() as db:
            row = db.execute("SELECT * FROM cases WHERE id=?", (case_id,)).fetchone()
            if not row:
                return {"ok": False, "reason": "Case not found"}
            facts = self._decode_json(row["confirmed_facts"])
            if confirmed_fact and confirmed_fact not in facts:
                facts.append(confirmed_fact)
            db.execute(
                "UPDATE cases SET specialist=?, confirmed_facts=? WHERE id=?",
                (specialist, json.dumps(facts), case_id),
            )
        packet = self.get_case(case_id)
        packet["handoff_message"] = "Context transferred; do not ask the caller to repeat it."
        return packet

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

    def record_case_event(
        self, case_id: str, action: str, evidence: dict[str, Any]
    ) -> dict[str, Any]:
        """Persist a compact audit event after a tool has read or changed business state."""
        with self._lock, self._connect() as db:
            row = db.execute("SELECT * FROM cases WHERE id=?", (case_id,)).fetchone()
            if not row:
                return {"ok": False, "reason": "Case not found"}
            actions = self._decode_json(row["completed_actions"])
            evidence_log = self._decode_json(row["evidence"])
            event = {"action": action, "result": evidence}
            if event not in evidence_log:
                evidence_log.append(event)
            if evidence.get("ok") is True or evidence.get("verified") is True:
                if action not in actions:
                    actions.append(action)
            db.execute(
                "UPDATE cases SET completed_actions=?, evidence=? WHERE id=?",
                (json.dumps(actions), json.dumps(evidence_log), case_id),
            )
        return self.get_case(case_id)

    def record_runtime_event(
        self,
        session_id: str,
        event_type: str,
        actor: str,
        payload: dict[str, Any],
        case_id: str | None = None,
    ) -> dict[str, Any]:
        event = {
            "id": f"evt_{uuid.uuid4().hex}",
            "session_id": session_id,
            "case_id": case_id,
            "event_type": event_type,
            "actor": actor,
            "payload": payload,
            "created_at": datetime.now(UTC).isoformat(),
        }
        with self._connect() as db:
            db.execute(
                "INSERT INTO runtime_events VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    event["id"],
                    session_id,
                    case_id,
                    event_type,
                    actor,
                    json.dumps(payload),
                    event["created_at"],
                ),
            )
        return event

    def list_runtime_events(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM runtime_events ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [{**dict(row), "payload": self._decode_json(row["payload"])} for row in rows]

    def list_cases(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute("SELECT id FROM cases ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [self.get_case(row["id"]) for row in rows]
