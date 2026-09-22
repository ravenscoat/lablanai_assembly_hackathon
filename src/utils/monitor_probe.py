"""Monitor probe detection helpers."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any


def is_monitor_probe(ctx: Any) -> bool:
    """Return true when a LiveKit job/room was created by the agent monitor."""
    room = getattr(ctx, "room", None)
    room_name = getattr(room, "name", "") if room else ""
    if isinstance(room_name, str) and room_name.startswith("navai-monitor-"):
        return True

    for metadata in _iter_metadata(ctx):
        if _metadata_marks_monitor(metadata):
            return True

    return False


def _iter_metadata(ctx: Any) -> Iterable[Any]:
    job = getattr(ctx, "job", None)
    if job:
        for attr in ("metadata", "dispatch_metadata", "agent_metadata"):
            value = getattr(job, attr, None)
            if value:
                yield value

    room = getattr(ctx, "room", None)
    if room:
        value = getattr(room, "metadata", None)
        if value:
            yield value


def _metadata_marks_monitor(metadata: Any) -> bool:
    parsed = _parse_metadata(metadata)
    if not isinstance(parsed, dict):
        return False

    nested = parsed.get("metadata")
    if isinstance(nested, dict) and _metadata_marks_monitor(nested):
        return True

    return parsed.get("source") == "agent_monitor" or parsed.get("skip_call_tracking") is True


def _parse_metadata(metadata: Any) -> Any:
    if isinstance(metadata, str):
        try:
            return json.loads(metadata)
        except json.JSONDecodeError:
            return None
    return metadata
