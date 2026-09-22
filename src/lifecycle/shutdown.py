# src/lifecycle/shutdown.py
"""
Shutdown callbacks: cleanup on call disconnect.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

from utils.monitor_probe import is_monitor_probe

logger = logging.getLogger(__name__)


def _iso_z(dt: datetime) -> str:
    """Format datetime as ISO 8601 with millisecond precision and Z suffix.

    Backend rejects '+00:00' offset, so we always emit the Z form.
    """
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def _build_latency_metrics(telephony_tracker: Any) -> dict[str, Any]:
    snapshot = telephony_tracker.snapshot()
    avgs = snapshot.get("averages", {})
    sdk_avgs = snapshot.get("sdk_averages", {})

    mapping = {
        "stt_latency_avg_ms": ("stt_ms", "transcription_delay_ms"),
        "llm_latency_avg_ms": ("llm_ms", "llm_node_ttft_ms"),
        "tts_latency_avg_ms": ("tts_ms", "tts_node_ttfb_ms"),
        "total_latency_avg_ms": ("measured_e2e_ms", "e2e_latency_ms"),
    }
    metrics: dict[str, Any] = {}
    for output_key, (direct_key, sdk_key) in mapping.items():
        value = avgs.get(direct_key) or sdk_avgs.get(sdk_key)
        if isinstance(value, (int, float)) and value > 0:
            metrics[output_key] = round(float(value), 2)
    turn_latencies = _build_turn_latencies(snapshot)
    if turn_latencies:
        metrics["turn_latencies"] = turn_latencies
    return metrics


def _positive_ms(value: Any) -> float | None:
    if isinstance(value, (int, float)) and value > 0:
        return round(float(value), 2)
    return None


def _normalize_metric_timestamp(value: Any) -> str | None:
    if not value:
        return None
    text = str(value)
    if text.endswith("+00:00"):
        return text[:-6] + "Z"
    return text


def _build_turn_latencies(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    turns = snapshot.get("recent_turns", [])
    if not isinstance(turns, list):
        return []

    direct_turns = [
        turn
        for turn in turns
        if isinstance(turn, dict)
        and turn.get("role") == "assistant"
        and turn.get("source") != "sdk"
        and isinstance(turn.get("turn"), int)
    ]

    if direct_turns:
        normalized: list[dict[str, Any]] = []
        for index, turn in enumerate(direct_turns[-20:], start=1):
            item: dict[str, Any] = {
                "turn_index": index,
                "source": "agent_direct",
            }
            timestamp = _normalize_metric_timestamp(turn.get("timestamp"))
            if timestamp:
                item["timestamp"] = timestamp
            values = {
                "total_latency_ms": _positive_ms(turn.get("measured_e2e_ms")),
                "stt_latency_ms": _positive_ms(turn.get("stt_ms")),
                "llm_latency_ms": _positive_ms(turn.get("llm_ms")),
                "tts_latency_ms": _positive_ms(turn.get("tts_ms")),
                "transport_overhead_ms": _positive_ms(turn.get("transport_overhead_ms")),
            }
            item.update({key: value for key, value in values.items() if value is not None})
            normalized.append(item)
        return normalized

    sdk_turns = [turn for turn in turns if isinstance(turn, dict) and turn.get("source") == "sdk"]
    normalized = []
    for index, turn in enumerate(sdk_turns[-20:], start=1):
        item = {
            "turn_index": index,
            "source": "livekit_sdk",
        }
        timestamp = _normalize_metric_timestamp(turn.get("timestamp"))
        if timestamp:
            item["timestamp"] = timestamp
        values = {
            "total_latency_ms": _positive_ms(turn.get("e2e_latency_ms")),
            "stt_latency_ms": _positive_ms(turn.get("transcription_delay_ms")),
            "llm_latency_ms": _positive_ms(turn.get("llm_node_ttft_ms")),
            "tts_latency_ms": _positive_ms(turn.get("tts_node_ttfb_ms")),
            "eou_latency_ms": _positive_ms(turn.get("end_of_turn_delay_ms")),
        }
        item.update({key: value for key, value in values.items() if value is not None})
        normalized.append(item)
    return normalized


def register_shutdown_callbacks(
    ctx: Any,
    agent: Any,
    config: Any,
) -> None:
    """
    Register cleanup callbacks for when the call ends.

    Sends to backend via PATCH /internal/calls/:id:
    - status, duration_seconds, transcript, transcript_text
    - ai_summary, transfer_reason, ended_at
    - metadata with room_name, agent_identity, handoff_mode
    """
    from lifecycle.call_tracker import update_call

    if not hasattr(ctx, "add_shutdown_callback"):
        logger.debug("No shutdown callback support (console mode)")
        return

    @ctx.add_shutdown_callback
    async def on_shutdown():
        logger.info(f"Call shutdown: tenant={config.tenant.slug}")
        if hasattr(agent, "_stop_silence_monitor"):
            agent._stop_silence_monitor()

        # Calculate duration
        duration = 0
        if hasattr(agent, "call_start_time") and agent.call_start_time:
            duration = int(time.time() - agent.call_start_time)

        # Build transcript in backend format (role, text, timestamp).
        # Backend requires ISO 8601 with Z suffix; epoch floats are not accepted.
        raw_history = getattr(agent, "conversation_history", None)
        transcript = None
        transcript_text = ""
        if raw_history:
            transcript = []
            lines = []
            for entry in raw_history:
                role = entry.get("role", "unknown")
                # Backend expects "agent" not "assistant"
                if role == "assistant":
                    role = "agent"
                content = entry.get("content", "")
                ts_epoch = entry.get("timestamp")
                if isinstance(ts_epoch, (int, float)):
                    ts_iso = _iso_z(datetime.fromtimestamp(ts_epoch, tz=timezone.utc))
                else:
                    ts_iso = ""
                transcript.append({"role": role, "text": content, "timestamp": ts_iso})
                lines.append(f"{role}: {content}")
            transcript_text = "\n".join(lines)

        # Determine transfer state
        transfer_offered = bool(
            getattr(agent, "transferred", False)
            or getattr(getattr(agent, "_session_state", None), "_transfer_offered", False)
        )
        handoff_completed = bool(getattr(agent, "handoff_completed", False))
        transferred = transfer_offered or handoff_completed
        status = "transferred" if transferred else "completed"
        transfer_reason = getattr(agent, "transfer_reason", "")

        # Build metadata with room info
        room_name = ""
        agent_identity = ""
        if hasattr(ctx, "room") and ctx.room:
            room_name = getattr(ctx.room, "name", "")
        if hasattr(agent, "agent_identity"):
            agent_identity = agent.agent_identity

        metadata = {
            "tenant": config.tenant.slug,
            "transferred": transferred,
            "handoff_completed": handoff_completed,
            "room_name": room_name,
            "agent_identity": agent_identity,
            "language": getattr(
                agent,
                "language",
                config.languages.default if hasattr(config, "languages") else "",
            ),
        }
        if transferred:
            transfer_mode = str(
                getattr(getattr(config, "transfer", None), "transfer_mode", "web") or "web"
            ).lower()
            metadata["handoff_mode"] = "sip" if transfer_mode == "sip" else "web_livekit_only"

        # Per-transfer telemetry (NAV-150). Populated by the escalate_to_human
        # tool when a transfer fires. Absent when no transfer happened.
        pending_transfer = getattr(agent, "_pending_transfer_metadata", None)
        if pending_transfer:
            metadata["transfer"] = dict(pending_transfer)

        # CSAT rating (NAV-156). Populated by the collect_csat tool when the
        # caller answers the 1-5 rating prompt before end_call. Absent when
        # the caller refused, hung up first, or never reached end_call (e.g.
        # transferred to a human — caller isn't on the line to rate).
        pending_csat = getattr(agent, "_pending_csat", None)
        if isinstance(pending_csat, int) and 1 <= pending_csat <= 5:
            metadata["csat"] = pending_csat

        telephony_tracker = getattr(agent, "_telephony_tracker", None)
        latency_metrics = _build_latency_metrics(telephony_tracker) if telephony_tracker else {}

        # Update call record
        if hasattr(agent, "call_db_id") and agent.call_db_id:
            try:
                if handoff_completed:
                    # AI intentionally left after operator handoff. Do not mark
                    # call as ended here; PBX/webhook lifecycle should finalize it.
                    await update_call(
                        call_db_id=agent.call_db_id,
                        tenant_id=config.tenant.id,
                        tenant_slug=config.tenant.slug,
                        status="transferred",
                        duration_seconds=duration,
                        ai_summary=getattr(agent, "ai_summary", ""),
                        transfer_reason=transfer_reason,
                        metrics=latency_metrics or None,
                        metadata={
                            **metadata,
                            "operator_handoff_completed_at": _iso_z(datetime.now(timezone.utc)),
                        },
                        murojat_id=getattr(agent, "murojaat_id", None),
                    )
                else:
                    await update_call(
                        call_db_id=agent.call_db_id,
                        tenant_id=config.tenant.id,
                        tenant_slug=config.tenant.slug,
                        status=status,
                        duration_seconds=duration,
                        transcript=transcript,
                        transcript_text=transcript_text,
                        ai_summary=getattr(agent, "ai_summary", ""),
                        transfer_reason=transfer_reason,
                        ended_at=_iso_z(datetime.now(timezone.utc)),
                        metrics=latency_metrics or None,
                        metadata=metadata,
                        murojat_id=getattr(agent, "murojaat_id", None),
                    )
            except Exception as e:
                logger.error(f"Failed to update call record on shutdown: {e}")

        # Log final telephony metrics and unregister tracker
        if telephony_tracker and not is_monitor_probe(ctx):
            try:
                snapshot = telephony_tracker.snapshot()
                avgs = snapshot.get("averages", {})
                logger.info(
                    f"Telephony summary: e2e_avg={avgs.get('e2e_latency_ms', '?')}ms, "
                    f"transport_overhead_avg={avgs.get('transport_overhead_ms', '?')}ms, "
                    f"transcription_delay_avg={avgs.get('transcription_delay_ms', '?')}ms, "
                    f"turns={snapshot.get('turn_count', 0)}"
                )
            except Exception:
                pass

            try:
                telephony_tracker.persist()
            except Exception:
                pass

        if handoff_completed:
            logger.info(
                f"Agent session ended after operator handoff: tenant={config.tenant.slug}, "
                f"duration={duration}s, transferred={transferred}"
            )
        else:
            logger.info(
                f"Call completed: tenant={config.tenant.slug}, "
                f"duration={duration}s, transferred={transferred}"
            )
