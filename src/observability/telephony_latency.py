"""
Telephony latency tracker: measures timing across the full voice loop.

Captures timing data from multiple sources:

1. SIP participant lifecycle (room events):
   - Call setup, track subscribe, first audio frame

2. Direct e2e measurement (session state changes):
   - user_stopped_speaking -> agent_started_speaking = measured_e2e_ms
   - This works regardless of SDK metrics availability

3. Per-turn pipeline metrics (ChatMessage.metrics, if available):
   - e2e_latency, transcription_delay, llm_node_ttft, tts_node_ttfb

4. Transport overhead = e2e - (STT + LLM + TTS processing)
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# Prometheus metrics (lazy import to avoid circular deps at module scope)
_prom = None


def _get_prom():
    global _prom
    if _prom is None:
        try:
            from observability import prometheus_metrics as pm

            _prom = pm
        except Exception:
            pass
    return _prom


TELEPHONY_PERSIST_PATH = os.getenv("TELEPHONY_PERSIST_PATH", "/tmp/navai-telephony-latency.json")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class SIPTimeline:
    """Timestamps for the SIP participant lifecycle within a single call."""

    agent_entrypoint_at: float | None = None
    session_started_at: float | None = None
    participant_connected_at: float | None = None
    track_subscribed_at: float | None = None
    first_audio_at: float | None = None
    call_status_transitions: list[dict[str, Any]] = field(default_factory=list)

    def record_status(self, status: str) -> None:
        self.call_status_transitions.append(
            {
                "status": status,
                "timestamp": _utc_now(),
                "time": time.monotonic(),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}

        if self.agent_entrypoint_at and self.participant_connected_at:
            result["call_setup_ms"] = round(
                (self.participant_connected_at - self.agent_entrypoint_at) * 1000, 2
            )

        if self.participant_connected_at and self.track_subscribed_at:
            result["track_subscribe_ms"] = round(
                (self.track_subscribed_at - self.participant_connected_at) * 1000, 2
            )

        if self.track_subscribed_at and self.first_audio_at:
            result["first_audio_ms"] = round(
                (self.first_audio_at - self.track_subscribed_at) * 1000, 2
            )

        if self.session_started_at and self.first_audio_at:
            result["session_to_first_audio_ms"] = round(
                (self.first_audio_at - self.session_started_at) * 1000, 2
            )

        if self.call_status_transitions:
            result["sip_status_transitions"] = [
                {"status": t["status"], "timestamp": t["timestamp"]}
                for t in self.call_status_transitions
            ]

        return result


class TelephonyLatencyTracker:
    """
    Tracks telephony-level latency for a single call session.

    Uses two independent timing methods:
    1. Direct state-change timing (always works)
    2. ChatMessage.metrics from SDK (bonus, if available)
    """

    def __init__(self, *, call_id: str = "", tenant: str = ""):
        self.call_id = call_id
        self.tenant = tenant
        self.sip_timeline = SIPTimeline()
        self._turn_metrics: deque[dict[str, Any]] = deque(maxlen=200)
        self._turn_count = 0

        # Direct e2e timing via state changes
        self._user_stopped_at: float | None = None
        self._agent_started_at: float | None = None
        self._last_stt_ms: float = 0
        self._last_llm_ms: float = 0
        self._last_tts_ms: float = 0

    # ---- SIP participant lifecycle ----

    def mark_entrypoint(self) -> None:
        self.sip_timeline.agent_entrypoint_at = time.monotonic()

    def mark_session_started(self) -> None:
        self.sip_timeline.session_started_at = time.monotonic()

    def mark_participant_connected(self, participant_identity: str = "") -> None:
        self.sip_timeline.participant_connected_at = time.monotonic()
        logger.info(
            "[TELEPHONY] SIP participant connected: %s (call_id=%s)",
            participant_identity,
            self.call_id,
        )

    def mark_track_subscribed(self) -> None:
        self.sip_timeline.track_subscribed_at = time.monotonic()
        logger.info("[TELEPHONY] Audio track subscribed (call_id=%s)", self.call_id)

    def mark_first_audio(self) -> None:
        if self.sip_timeline.first_audio_at is not None:
            return
        self.sip_timeline.first_audio_at = time.monotonic()
        timeline = self.sip_timeline.to_dict()
        logger.info(
            "[TELEPHONY] First audio frame received (call_id=%s): "
            "call_setup=%sms, track_subscribe=%sms, first_audio=%sms",
            self.call_id,
            timeline.get("call_setup_ms", "?"),
            timeline.get("track_subscribe_ms", "?"),
            timeline.get("first_audio_ms", "?"),
        )
        # Export SIP timeline to Prometheus
        pm = _get_prom()
        if pm and timeline:
            t = self.tenant or "unknown"
            if "call_setup_ms" in timeline:
                pm.navai_sip_call_setup_seconds.labels(tenant=t).observe(
                    timeline["call_setup_ms"] / 1000
                )
            if "track_subscribe_ms" in timeline:
                pm.navai_sip_track_subscribe_seconds.labels(tenant=t).observe(
                    timeline["track_subscribe_ms"] / 1000
                )
            if "first_audio_ms" in timeline:
                pm.navai_sip_first_audio_seconds.labels(tenant=t).observe(
                    timeline["first_audio_ms"] / 1000
                )

        self.persist()

    def record_sip_status(self, status: str) -> None:
        self.sip_timeline.record_status(status)
        logger.info(
            "[TELEPHONY] SIP status -> %s (call_id=%s)",
            status,
            self.call_id,
        )

    # ---- Direct e2e timing from session state changes ----

    def mark_user_stopped_speaking(self) -> None:
        """Called when user_state_changed -> listening/idle."""
        self._user_stopped_at = time.monotonic()
        self._last_stt_ms = 0
        self._last_llm_ms = 0
        self._last_tts_ms = 0

    def mark_agent_started_speaking(self) -> None:
        """Called when agent_state_changed -> speaking."""
        self._agent_started_at = time.monotonic()

        if self._user_stopped_at is None:
            return

        e2e_ms = (self._agent_started_at - self._user_stopped_at) * 1000
        stt_ms = self._last_stt_ms
        tts_ms = self._last_tts_ms
        llm_ms = self._last_llm_ms

        # If we don't have direct LLM measurement, estimate from network topology
        if llm_ms == 0:
            llm_ms = self._get_last_llm_duration()

        processing_ms = stt_ms + llm_ms + tts_ms
        transport_ms = max(0, e2e_ms - processing_ms) if processing_ms > 0 else 0

        self._turn_count += 1
        entry: dict[str, Any] = {
            "turn": self._turn_count,
            "role": "assistant",
            "timestamp": _utc_now(),
            "measured_e2e_ms": round(e2e_ms, 2),
            "stt_ms": round(stt_ms, 2),
            "llm_ms": round(llm_ms, 2),
            "tts_ms": round(tts_ms, 2),
            "processing_ms": round(processing_ms, 2),
            "transport_overhead_ms": round(transport_ms, 2),
        }
        self._turn_metrics.append(entry)

        logger.info(
            "[TELEPHONY] Turn %d: e2e=%dms (stt=%dms + llm=%dms + tts=%dms + transport=%dms)",
            self._turn_count,
            int(e2e_ms),
            int(stt_ms),
            int(llm_ms),
            int(tts_ms),
            int(transport_ms),
        )

        # Export to Prometheus
        pm = _get_prom()
        if pm:
            t = self.tenant or "unknown"
            pm.navai_turn_e2e_latency_seconds.labels(tenant=t).observe(e2e_ms / 1000)
            if stt_ms > 0:
                pm.navai_stt_latency_seconds.labels(tenant=t, provider="yandex").observe(
                    stt_ms / 1000
                )
            if llm_ms > 0:
                pm.navai_llm_latency_seconds.labels(tenant=t, provider="gemini").observe(
                    llm_ms / 1000
                )
            if tts_ms > 0:
                pm.navai_tts_latency_seconds.labels(tenant=t, provider="yandex").observe(
                    tts_ms / 1000
                )
            if transport_ms > 0:
                pm.navai_transport_overhead_seconds.labels(tenant=t).observe(transport_ms / 1000)

        self._user_stopped_at = None
        self.persist()

    def _get_last_llm_duration(self) -> float:
        """Pull last LLM request duration from network topology stats."""
        try:
            from observability.network_topology import get_network_topology_snapshot

            snap = get_network_topology_snapshot(limit=1)
            for svc in snap.get("services", []):
                if svc.get("service") in ("gemini_llm", "openai_llm"):
                    last = svc.get("stats", {}).get("last_request")
                    if last and last.get("ok"):
                        return float(last.get("duration_ms", 0))
        except Exception:
            pass
        return 0

    def record_stt_duration(self, duration_ms: float) -> None:
        self._last_stt_ms = duration_ms

    def record_llm_duration(self, duration_ms: float) -> None:
        self._last_llm_ms = duration_ms

    def record_tts_duration(self, duration_ms: float) -> None:
        self._last_tts_ms = duration_ms

    # ---- SDK metrics (bonus, if ChatMessage.metrics is populated) ----

    def record_sdk_metrics(self, *, role: str, sdk_metrics: dict[str, Any]) -> None:
        """Record per-turn latency from ChatMessage.metrics (if available)."""
        entry: dict[str, Any] = {
            "turn": self._turn_count or "sdk",
            "role": role,
            "source": "sdk",
            "timestamp": _utc_now(),
        }

        has_data = False
        for key in (
            "e2e_latency",
            "transcription_delay",
            "end_of_turn_delay",
            "on_user_turn_completed_delay",
            "llm_node_ttft",
            "tts_node_ttfb",
        ):
            val = sdk_metrics.get(key)
            if val is not None:
                entry[f"{key}_ms"] = round(float(val) * 1000, 2)
                has_data = True

        if has_data:
            self._turn_metrics.append(entry)
            sdk_e2e = entry.get("e2e_latency_ms", "?")
            sdk_ttft = entry.get("llm_node_ttft_ms", "?")
            sdk_ttfb = entry.get("tts_node_ttfb_ms", "?")
            logger.info(
                "[TELEPHONY] SDK metrics: e2e=%sms, llm_ttft=%sms, tts_ttfb=%sms",
                sdk_e2e,
                sdk_ttft,
                sdk_ttfb,
            )

            # Export SDK metrics to Prometheus
            pm = _get_prom()
            if pm:
                t = self.tenant or "unknown"
                if "e2e_latency_ms" in entry:
                    pm.navai_sdk_e2e_latency_seconds.labels(tenant=t).observe(
                        entry["e2e_latency_ms"] / 1000
                    )
                if "llm_node_ttft_ms" in entry:
                    pm.navai_sdk_llm_ttft_seconds.labels(tenant=t).observe(
                        entry["llm_node_ttft_ms"] / 1000
                    )
                if "tts_node_ttfb_ms" in entry:
                    pm.navai_sdk_tts_ttfb_seconds.labels(tenant=t).observe(
                        entry["tts_node_ttfb_ms"] / 1000
                    )

            self.persist()  # Write SDK metrics to JSON for bridge

    # ---- Snapshot + persistence ----

    def snapshot(self) -> dict[str, Any]:
        turns = list(self._turn_metrics)

        # Compute averages from direct measurements (role=assistant, not source=sdk)
        direct_turns = [
            t for t in turns if t.get("role") == "assistant" and t.get("source") != "sdk"
        ]
        avg: dict[str, Any] = {}
        if direct_turns:
            for key in (
                "measured_e2e_ms",
                "stt_ms",
                "llm_ms",
                "tts_ms",
                "processing_ms",
                "transport_overhead_ms",
            ):
                values = [t[key] for t in direct_turns if key in t and t[key] > 0]
                if values:
                    avg[key] = round(sum(values) / len(values), 2)

        # Also grab SDK averages if available
        sdk_turns = [t for t in turns if t.get("source") == "sdk"]
        sdk_avg: dict[str, Any] = {}
        if sdk_turns:
            for key in (
                "e2e_latency_ms",
                "transcription_delay_ms",
                "llm_node_ttft_ms",
                "tts_node_ttfb_ms",
            ):
                values = [t[key] for t in sdk_turns if key in t]
                if values:
                    sdk_avg[key] = round(sum(values) / len(values), 2)

        return {
            "call_id": self.call_id,
            "tenant": self.tenant,
            "generated_at": _utc_now(),
            "sip_timeline": self.sip_timeline.to_dict(),
            "turn_count": self._turn_count,
            "averages": avg,
            "sdk_averages": sdk_avg,
            "recent_turns": turns[-20:],
        }

    def persist(self) -> None:
        """Write snapshot to disk so the health server (main process) can read it."""
        try:
            data = self.snapshot()
            payload = json.dumps(data, ensure_ascii=False)
            tmp = TELEPHONY_PERSIST_PATH + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(payload)
            os.replace(tmp, TELEPHONY_PERSIST_PATH)
        except Exception as e:
            logger.debug("Failed to persist telephony data: %s", e)


def get_telephony_snapshot() -> dict[str, Any]:
    """Read persisted telephony data (called from health server in main process)."""
    try:
        if not os.path.exists(TELEPHONY_PERSIST_PATH):
            return {"active_calls": 0, "calls": []}
        with open(TELEPHONY_PERSIST_PATH, encoding="utf-8") as f:
            data = json.loads(f.read())
        return {"active_calls": 1, "calls": [data]}
    except Exception:
        return {"active_calls": 0, "calls": []}
