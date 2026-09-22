from __future__ import annotations

import json
from types import SimpleNamespace

from utils.monitor_probe import is_monitor_probe


def _ctx(*, room_name: str = "regular-room", room_metadata=None, job_metadata=None):
    return SimpleNamespace(
        job=SimpleNamespace(metadata=job_metadata),
        room=SimpleNamespace(name=room_name, metadata=room_metadata),
    )


def test_monitor_probe_detects_monitor_room_prefix() -> None:
    assert is_monitor_probe(_ctx(room_name="navai-monitor-1779221490-1d7ceb"))


def test_monitor_probe_detects_room_metadata_source() -> None:
    metadata = json.dumps({"source": "agent_monitor"})

    assert is_monitor_probe(_ctx(room_metadata=metadata))


def test_monitor_probe_detects_dispatch_skip_flag() -> None:
    metadata = json.dumps({"skip_call_tracking": True})

    assert is_monitor_probe(_ctx(job_metadata=metadata))


def test_monitor_probe_detects_nested_dispatch_metadata() -> None:
    metadata = json.dumps({"metadata": {"source": "agent_monitor"}})

    assert is_monitor_probe(_ctx(job_metadata=metadata))


def test_monitor_probe_ignores_regular_calls_and_bad_metadata() -> None:
    assert not is_monitor_probe(_ctx(job_metadata="{not-json"))
    assert not is_monitor_probe(_ctx(room_metadata=json.dumps({"source": "voice_agent"})))
