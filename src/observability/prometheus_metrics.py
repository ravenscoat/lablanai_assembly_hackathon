"""
Prometheus metrics for the voice agent.

Defines all navai_* metrics that the Grafana dashboard queries.
These metrics are populated by instrumentation in:
  - telephony_latency.py (per-turn STT/LLM/TTS/E2E latency)
  - session_events.py (SDK metrics)
  - lifecycle/shutdown.py (call duration)
  - lifecycle/call_tracker.py (call counts)
  - knowledge/rag_engine.py (KB search)
  - tools/ (tool call tracking)

Uses prometheus_client with multiprocess support so metrics from
child job processes are visible to the main-process health server.
"""

from __future__ import annotations

import logging
import os

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
)

logger = logging.getLogger(__name__)

# =============================================================================
# Histogram buckets tuned for voice pipeline latencies
# =============================================================================

# STT/TTS: typically 100ms–2s
VOICE_BUCKETS = (0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0)

# LLM: typically 200ms–5s
LLM_BUCKETS = (0.1, 0.2, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0)

# E2E turn: sum of pipeline stages, typically 500ms–5s
E2E_BUCKETS = (0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0, 15.0)

# Call duration: seconds to minutes
CALL_DURATION_BUCKETS = (10, 30, 60, 120, 180, 300, 600, 900, 1800, 3600)

# SIP setup: typically <1s
SIP_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0)

# KB / tool latency
KB_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0)


# =============================================================================
# Call metrics
# =============================================================================

navai_calls_active = Gauge(
    "navai_calls_active",
    "Number of currently active calls",
    ["tenant"],
)

navai_calls_total = Counter(
    "navai_calls_total",
    "Total number of calls",
    ["tenant", "status"],
)

navai_call_duration_seconds = Histogram(
    "navai_call_duration_seconds",
    "Call duration in seconds",
    ["tenant"],
    buckets=CALL_DURATION_BUCKETS,
)


# =============================================================================
# AI Pipeline latency (from TelephonyLatencyTracker direct measurement)
# =============================================================================

navai_turn_e2e_latency_seconds = Histogram(
    "navai_turn_e2e_latency_seconds",
    "End-to-end turn latency (user stopped speaking → agent started speaking)",
    ["tenant"],
    buckets=E2E_BUCKETS,
)

navai_stt_latency_seconds = Histogram(
    "navai_stt_latency_seconds",
    "Speech-to-text processing latency",
    ["tenant", "provider"],
    buckets=VOICE_BUCKETS,
)

navai_llm_latency_seconds = Histogram(
    "navai_llm_latency_seconds",
    "LLM response latency",
    ["tenant", "provider"],
    buckets=LLM_BUCKETS,
)

navai_tts_latency_seconds = Histogram(
    "navai_tts_latency_seconds",
    "Text-to-speech synthesis latency",
    ["tenant", "provider"],
    buckets=VOICE_BUCKETS,
)

navai_transport_overhead_seconds = Histogram(
    "navai_transport_overhead_seconds",
    "Transport overhead (E2E minus processing time)",
    ["tenant"],
    buckets=VOICE_BUCKETS,
)


# =============================================================================
# LiveKit SDK metrics (bonus, from ChatMessage.metrics)
# =============================================================================

navai_sdk_e2e_latency_seconds = Histogram(
    "navai_sdk_e2e_latency_seconds",
    "SDK-reported end-to-end latency",
    ["tenant"],
    buckets=E2E_BUCKETS,
)

navai_sdk_llm_ttft_seconds = Histogram(
    "navai_sdk_llm_ttft_seconds",
    "SDK-reported LLM time-to-first-token",
    ["tenant"],
    buckets=LLM_BUCKETS,
)

navai_sdk_tts_ttfb_seconds = Histogram(
    "navai_sdk_tts_ttfb_seconds",
    "SDK-reported TTS time-to-first-byte",
    ["tenant"],
    buckets=VOICE_BUCKETS,
)


# =============================================================================
# SIP / Telephony setup
# =============================================================================

navai_sip_call_setup_seconds = Histogram(
    "navai_sip_call_setup_seconds",
    "SIP call setup time (entrypoint → participant connected)",
    ["tenant"],
    buckets=SIP_BUCKETS,
)

navai_sip_track_subscribe_seconds = Histogram(
    "navai_sip_track_subscribe_seconds",
    "Audio track subscribe time",
    ["tenant"],
    buckets=SIP_BUCKETS,
)

navai_sip_first_audio_seconds = Histogram(
    "navai_sip_first_audio_seconds",
    "Time from track subscribe to first audio frame",
    ["tenant"],
    buckets=SIP_BUCKETS,
)


# =============================================================================
# Knowledge Base / RAG
# =============================================================================

navai_kb_search_total = Counter(
    "navai_kb_search_total",
    "Total knowledge base searches",
    ["tenant", "result"],
)

navai_kb_search_latency_seconds = Histogram(
    "navai_kb_search_latency_seconds",
    "KB search total latency (embedding + vector search)",
    ["tenant"],
    buckets=KB_BUCKETS,
)

navai_kb_embedding_latency_seconds = Histogram(
    "navai_kb_embedding_latency_seconds",
    "KB embedding generation latency",
    ["tenant"],
    buckets=KB_BUCKETS,
)

navai_kb_qdrant_latency_seconds = Histogram(
    "navai_kb_qdrant_latency_seconds",
    "KB Qdrant vector search latency",
    ["tenant"],
    buckets=KB_BUCKETS,
)


# =============================================================================
# Tool calls
# =============================================================================

navai_tool_calls_total = Counter(
    "navai_tool_calls_total",
    "Total tool invocations",
    ["tenant", "tool_name"],
)

navai_tool_call_latency_seconds = Histogram(
    "navai_tool_call_latency_seconds",
    "Tool call execution latency",
    ["tenant", "tool_name"],
    buckets=KB_BUCKETS,
)


# =============================================================================
# Errors
# =============================================================================

navai_errors_total = Counter(
    "navai_errors_total",
    "Total errors by type",
    ["tenant", "error_type"],
)


# =============================================================================
# Agent info (static labels)
# =============================================================================

navai_agent_info = Info(
    "navai_agent",
    "Voice agent metadata",
)


# =============================================================================
# Metrics export helper (handles multiprocess mode)
# =============================================================================


def get_metrics_output() -> tuple[bytes, str]:
    """
    Generate Prometheus metrics output.
    Returns (body_bytes, content_type).
    Bridges persisted telephony data into prometheus metrics before export.
    """
    _bridge_telephony_data()
    return generate_latest(), CONTENT_TYPE_LATEST


# Dedupe per call: turn numbers restart from 1 in each call's persisted
# snapshot, so a global monotonic counter would silently skip every
# subsequent call after the first. Key the high-watermark by call_id and
# reset when the call changes.
_last_observed_call_id: str | None = None
_last_observed_turn = 0
_sip_bridged_call_ids: set[str] = set()


def _bridge_telephony_data() -> None:
    """Read persisted telephony JSON and push new turns into prometheus metrics."""
    global _last_observed_call_id, _last_observed_turn
    try:
        import json

        path = os.getenv("TELEPHONY_PERSIST_PATH", "/tmp/navai-telephony-latency.json")
        if not os.path.exists(path):
            return
        with open(path) as f:
            data = json.loads(f.read())

        tenant = data.get("tenant", "unknown")
        call_id = data.get("call_id") or ""

        # New call → reset turn watermark.
        if call_id != _last_observed_call_id:
            _last_observed_call_id = call_id
            _last_observed_turn = 0

        # Bridge SIP timeline once per call (not once per process lifetime).
        sip = data.get("sip_timeline", {})
        if sip and call_id and call_id not in _sip_bridged_call_ids:
            _sip_bridged_call_ids.add(call_id)
            if "call_setup_ms" in sip:
                navai_sip_call_setup_seconds.labels(tenant=tenant).observe(
                    sip["call_setup_ms"] / 1000
                )
            if "track_subscribe_ms" in sip:
                navai_sip_track_subscribe_seconds.labels(tenant=tenant).observe(
                    sip["track_subscribe_ms"] / 1000
                )
            if "first_audio_ms" in sip:
                navai_sip_first_audio_seconds.labels(tenant=tenant).observe(
                    sip["first_audio_ms"] / 1000
                )

        # Bridge turn metrics (only new turns)
        for turn in data.get("recent_turns", []):
            turn_num = turn.get("turn", 0)
            if isinstance(turn_num, int) and turn_num <= _last_observed_turn:
                continue

            if turn.get("source") == "sdk":
                if "e2e_latency_ms" in turn:
                    navai_sdk_e2e_latency_seconds.labels(tenant=tenant).observe(
                        turn["e2e_latency_ms"] / 1000
                    )
                if "llm_node_ttft_ms" in turn:
                    navai_sdk_llm_ttft_seconds.labels(tenant=tenant).observe(
                        turn["llm_node_ttft_ms"] / 1000
                    )
                if "tts_node_ttfb_ms" in turn:
                    navai_sdk_tts_ttfb_seconds.labels(tenant=tenant).observe(
                        turn["tts_node_ttfb_ms"] / 1000
                    )
            else:
                if "measured_e2e_ms" in turn:
                    navai_turn_e2e_latency_seconds.labels(tenant=tenant).observe(
                        turn["measured_e2e_ms"] / 1000
                    )
                if turn.get("stt_ms", 0) > 0:
                    navai_stt_latency_seconds.labels(tenant=tenant, provider="yandex").observe(
                        turn["stt_ms"] / 1000
                    )
                if turn.get("llm_ms", 0) > 0:
                    navai_llm_latency_seconds.labels(tenant=tenant, provider="gemini").observe(
                        turn["llm_ms"] / 1000
                    )
                if turn.get("tts_ms", 0) > 0:
                    navai_tts_latency_seconds.labels(tenant=tenant, provider="yandex").observe(
                        turn["tts_ms"] / 1000
                    )
                if turn.get("transport_overhead_ms", 0) > 0:
                    navai_transport_overhead_seconds.labels(tenant=tenant).observe(
                        turn["transport_overhead_ms"] / 1000
                    )

            if isinstance(turn_num, int) and turn_num > _last_observed_turn:
                _last_observed_turn = turn_num

    except Exception as e:
        logger.debug("Telephony bridge error: %s", e)


def init_agent_info(*, environment: str = "", version: str = "0.1.0") -> None:
    """Set static agent info labels (call once at startup)."""
    try:
        navai_agent_info.info(
            {
                "environment": environment or os.getenv("AGENT_ENV", "dev"),
                "version": version,
                "deploy_mode": os.getenv("DEPLOY_MODE", "unknown"),
            }
        )
    except Exception as e:
        logger.debug("Failed to set agent info: %s", e)
