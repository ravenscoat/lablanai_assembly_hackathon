"""Response dataclasses for Platform API."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CallerHistory:
    has_history: bool = False
    total_calls: int = 0
    last_topic: str = ""
    last_summary: str = ""
    last_call_status: str = ""
    last_language: str = ""  # ISO 639-1 code; empty if not on file / backend does not echo


@dataclass
class OperatorStatus:
    status: str = "unknown"  # in_queue, ringing, transferred, unknown
    operator_id: str | None = None
    operator_name: str | None = None


@dataclass
class MurojaatResult:
    success: bool = False
    murojaat_id: str | None = None
    error: str | None = None
