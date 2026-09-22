#!/usr/bin/env python3
"""Hourly production monitor for the LiveKit voice agent.

The monitor is intentionally separate from the agent runtime. It performs a
cheap health check plus a text-mode LiveKit probe that sends a short greeting
to a temporary room and waits for the agent's text transcription response.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import socket
import sys
import time
import traceback
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib import error, parse, request

DEFAULT_HEALTH_URL = "http://agent:8082/health"
DEFAULT_INTERVAL_SECONDS = 3600
DEFAULT_INITIAL_DELAY_SECONDS = 30
DEFAULT_STATE_PATH = "/state/agent-monitor-state.json"
DEFAULT_NOTIFY_MODE = "first_24h_all"
DEFAULT_FIRST_SUCCESS_HOURS = 24
DEFAULT_PROBE_TEXT = "hello"


@dataclass(frozen=True)
class MonitorConfig:
    health_url: str
    interval_seconds: int
    initial_delay_seconds: int
    state_path: Path
    notify_mode: str
    first_success_hours: float
    telegram_bot_token: str
    telegram_chat_id: str
    telegram_thread_id: str
    livekit_url: str
    livekit_api_key: str
    livekit_api_secret: str
    agent_name: str
    called_phone: str
    probe_text: str
    health_timeout_seconds: float
    agent_join_timeout_seconds: float
    response_timeout_seconds: float
    connect_grace_seconds: float


@dataclass
class ProbeResult:
    ok: bool
    status: str
    started_at: float
    finished_at: float
    health_ok: bool = False
    room_name: str = ""
    agent_identity: str = ""
    response_type: str = ""
    response_preview: str = ""
    error_type: str = ""
    error_message: str = ""
    details: str = ""
    # Enrichment for human-readable Telegram alerts. Defaults keep the dataclass
    # backward-compatible with callers and JSON state written by older versions.
    stage: str = ""
    category: str = ""
    summary: str = ""
    remediation: tuple[str, ...] = ()
    livekit_project: str = ""

    @property
    def duration_seconds(self) -> float:
        return max(0.0, self.finished_at - self.started_at)


@dataclass(frozen=True)
class FailureClassification:
    """Human-readable mapping of a raw probe failure to an actionable summary."""

    category: str
    summary: str
    remediation: tuple[str, ...] = ()


class _StageTracker:
    """Mutable tracker so an outer ``except`` can report which stage was last reached.

    Also carries the room name once it has been generated so failure alerts can
    include it in the ``Where it failed`` section — without that, the
    ``agent_not_joined`` remediation pointed readers at a room name that the
    alert body never contained (review feedback on PR #110).
    """

    __slots__ = ("current", "room_name")

    def __init__(self, initial: str = "init") -> None:
        self.current = initial
        self.room_name = ""

    def set(self, name: str) -> None:
        self.current = name

    def set_room_name(self, room_name: str) -> None:
        self.room_name = room_name


_QUOTA_PATTERN = re.compile(r"minutes?\s+limit\s+exceeded", re.IGNORECASE)
_LIVEKIT_PROJECT_PATTERN = re.compile(
    r"^(?:wss?|https?)://([^./]+)\.livekit\.cloud(?:[:/]|$)",
    re.IGNORECASE,
)
_NETWORK_HINTS = (
    "name or service not known",
    "temporary failure in name resolution",
    "no route to host",
    "connection refused",
    "connection reset",
    "network is unreachable",
)


def _extract_livekit_project(url: str) -> str:
    """Return the LiveKit Cloud project slug from a websocket URL.

    ``wss://debt-collection-agent-tzxnyg91.livekit.cloud`` -> ``debt-collection-agent-tzxnyg91``.
    Self-hosted or unknown URLs return an empty string.
    """
    if not url:
        return ""
    match = _LIVEKIT_PROJECT_PATTERN.match(url.strip())
    return match.group(1) if match else ""


def _human_delta(seconds: float) -> str:
    """Short human-friendly duration, e.g. ``3h 1m``, ``12m 4s``, ``45s``."""
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {secs}s"
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h {minutes}m"
    days, hours = divmod(hours, 24)
    return f"{days}d {hours}h"


def classify_failure(stage: str, error_type: str, error_message: str) -> FailureClassification:
    """Translate a raw probe failure into an actionable category + remediation list.

    Stage is the pipeline step where the exception bubbled (e.g. ``agent_join_wait``).
    The classifier inspects both the stage and the message so it works for both
    well-typed exceptions (``TimeoutError``) and opaque vendor errors (``ConnectError``
    whose only signal is the HTTP body embedded in ``str(exc)``).

    The ``stage == "health_check"`` branch is intentionally evaluated FIRST: the
    standard-library ``urlopen`` raises ``HTTPError`` whose ``str()`` includes
    ``HTTP Error 503: ...``, which would otherwise be swept up by the generic
    HTTP-status branches and falsely point on-call at LiveKit instead of the
    agent itself (review feedback on PR #110).
    """
    msg = (error_message or "").strip()
    lower = msg.lower()

    # 0) Health-check failures — must precede any HTTP-status-code branch.
    if stage == "health_check":
        if (
            "did not return JSON" in msg
            or "non-healthy" in msg
            or "returned HTTP" in msg
            or re.search(r"HTTP\s*Error\s*\d{3}", msg, re.IGNORECASE) is not None
        ):
            return FailureClassification(
                category="health_check_unhealthy",
                summary="Agent /health endpoint returned a non-healthy response.",
                remediation=(
                    "Inspect agent logs: docker logs --tail 200 voice-agent",
                    "Check agent dependencies (Qdrant, network egress).",
                ),
            )
        return FailureClassification(
            category="health_check_unreachable",
            summary="Could not reach the agent /health endpoint.",
            remediation=(
                "Container may be down: docker ps | grep voice-agent",
                "Verify MONITOR_HEALTH_URL points to the right host:port.",
            ),
        )

    # 1) LiveKit Cloud monthly minutes cap — very specific, must precede the
    #    generic 429 branch so the alert names the real root cause.
    if _QUOTA_PATTERN.search(lower):
        return FailureClassification(
            category="livekit_quota_exhausted",
            summary="LiveKit Cloud monthly connection-minutes quota exhausted.",
            remediation=(
                "Top up the LiveKit Cloud plan or buy more minutes at https://cloud.livekit.io",
                "Real calls share this project and will also return 429 until the quota resets.",
                "Consider raising MONITOR_INTERVAL_SECONDS or moving the monitor to a separate LiveKit project.",
            ),
        )

    # 2) Plain HTTP 429 (rate-limit, NOT monthly quota)
    if re.search(r"\b429\b", msg) or "too many requests" in lower:
        return FailureClassification(
            category="livekit_rate_limited",
            summary="LiveKit Cloud rate-limited the probe (HTTP 429).",
            remediation=(
                "Back off probe frequency by raising MONITOR_INTERVAL_SECONDS.",
                "Verify only one monitor instance is running against this LiveKit project.",
            ),
        )

    # 3) Auth/permission failures
    if (
        re.search(r"\b40[13]\b", msg)
        or "unauthorized" in lower
        or "invalid api key" in lower
        or "permission denied" in lower
        or "forbidden" in lower
    ):
        return FailureClassification(
            category="livekit_auth_failed",
            summary="LiveKit credentials are invalid or missing required permissions.",
            remediation=(
                "Verify LIVEKIT_API_KEY and LIVEKIT_API_SECRET match the active LiveKit Cloud project.",
                "Ensure the API key has room.create + agent_dispatch permissions.",
            ),
        )

    # 4) LiveKit Cloud server-side errors
    if re.search(r"\b5\d\d\b", msg):
        return FailureClassification(
            category="livekit_server_error",
            summary="LiveKit Cloud returned a server error.",
            remediation=(
                "Check https://status.livekit.io for ongoing incidents.",
                "The next probe interval will retry automatically.",
            ),
        )

    # 5) Agent dispatched but worker never picked it up
    if stage == "agent_join_wait" or "did not join" in lower:
        return FailureClassification(
            category="agent_not_joined",
            summary="Agent worker did not pick up the monitor dispatch within the join timeout.",
            remediation=(
                "Check the voice-agent container: docker logs --tail 100 voice-agent",
                "Verify MONITOR_AGENT_NAME matches the AGENT_NAME the worker registered with.",
                "Look up the room name from this alert in agent logs to see the dispatch trail.",
            ),
        )

    # 6) Agent joined but never produced a transcription
    if stage == "response_wait" or "did not respond" in lower:
        return FailureClassification(
            category="agent_not_responded",
            summary="Agent joined the probe room but did not transcribe a response in time.",
            remediation=(
                "Inspect agent logs around the room name for STT/LLM/TTS errors.",
                "Verify upstream services (Gemini, Yandex, Custom STT) are reachable from the agent container.",
                "If responses are merely slow, raise MONITOR_RESPONSE_TIMEOUT_SECONDS.",
            ),
        )

    # 7) Generic network / DNS errors
    if any(hint in lower for hint in _NETWORK_HINTS):
        return FailureClassification(
            category="network_error",
            summary=f"Network/DNS error reaching upstream during '{stage or 'probe'}'.",
            remediation=(
                "Check container network connectivity and DNS resolution.",
                "If only LiveKit fails, verify LIVEKIT_URL is correct and reachable.",
            ),
        )

    # 8) Missing required configuration
    if "missing LiveKit monitor configuration" in msg or msg.endswith(" is not set"):
        return FailureClassification(
            category="config_missing",
            summary="Monitor is missing required configuration.",
            remediation=(
                "Set the env-vars named in the raw error on the voice-agent-monitor container.",
            ),
        )

    # 9) Catch-all
    return FailureClassification(
        category="unknown_error",
        summary=f"Unclassified failure ({error_type or 'error'}) during '{stage or 'probe'}'.",
        remediation=(
            "Check container logs: docker logs --tail 200 voice-agent-monitor",
            "See the raw error below for clues.",
        ),
    )


def _now() -> float:
    return time.time()


def _iso(ts: float | None = None) -> str:
    return datetime.fromtimestamp(ts or _now(), UTC).isoformat(timespec="seconds")


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def load_config() -> MonitorConfig:
    return MonitorConfig(
        health_url=os.getenv("MONITOR_HEALTH_URL", DEFAULT_HEALTH_URL),
        interval_seconds=_env_int("MONITOR_INTERVAL_SECONDS", DEFAULT_INTERVAL_SECONDS),
        initial_delay_seconds=_env_int(
            "MONITOR_INITIAL_DELAY_SECONDS", DEFAULT_INITIAL_DELAY_SECONDS
        ),
        state_path=Path(os.getenv("MONITOR_STATE_PATH", DEFAULT_STATE_PATH)),
        notify_mode=os.getenv("MONITOR_NOTIFY_MODE", DEFAULT_NOTIFY_MODE),
        first_success_hours=_env_float("MONITOR_FIRST_SUCCESS_HOURS", DEFAULT_FIRST_SUCCESS_HOURS),
        telegram_bot_token=os.getenv("MONITOR_TELEGRAM_BOT_TOKEN")
        or os.getenv("TELEGRAM_BOT_TOKEN", ""),
        telegram_chat_id=os.getenv("MONITOR_TELEGRAM_CHAT_ID") or os.getenv("TELEGRAM_CHAT_ID", ""),
        telegram_thread_id=os.getenv("MONITOR_TELEGRAM_THREAD_ID")
        or os.getenv("TELEGRAM_THREAD_ID", ""),
        livekit_url=os.getenv("LIVEKIT_URL", ""),
        livekit_api_key=os.getenv("LIVEKIT_API_KEY", ""),
        livekit_api_secret=os.getenv("LIVEKIT_API_SECRET", ""),
        agent_name=os.getenv("MONITOR_AGENT_NAME") or os.getenv("AGENT_NAME", "voice-agent"),
        called_phone=os.getenv("MONITOR_CALLED_PHONE") or os.getenv("AGENT_PHONE_NUMBER", ""),
        probe_text=os.getenv("MONITOR_PROBE_TEXT", DEFAULT_PROBE_TEXT),
        health_timeout_seconds=_env_float("MONITOR_HEALTH_TIMEOUT_SECONDS", 5.0),
        agent_join_timeout_seconds=_env_float("MONITOR_AGENT_JOIN_TIMEOUT_SECONDS", 60.0),
        response_timeout_seconds=_env_float("MONITOR_RESPONSE_TIMEOUT_SECONDS", 60.0),
        connect_grace_seconds=_env_float("MONITOR_CONNECT_GRACE_SECONDS", 5.0),
    )


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"monitor state at {path} must be a JSON object")
    return data


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2, sort_keys=True)
        fh.write("\n")
    tmp_path.replace(path)


def ensure_state_started(state: dict[str, Any], now: float) -> None:
    state.setdefault("started_at", now)


def within_first_success_window(
    state: dict[str, Any], now: float, first_success_hours: float
) -> bool:
    started_at = float(state.get("started_at", now))
    return now - started_at <= first_success_hours * 3600


def should_notify(result: ProbeResult, state: dict[str, Any], config: MonitorConfig) -> bool:
    mode = config.notify_mode.strip().lower()
    if mode == "always":
        return True
    if mode == "failures_only":
        return not result.ok
    if mode == "failures_and_recovery":
        return (not result.ok) or state.get("last_status") == "failure"
    if mode == "first_24h_all":
        if within_first_success_window(state, result.finished_at, config.first_success_hours):
            return True
        return (not result.ok) or state.get("last_status") == "failure"
    raise ValueError(
        "MONITOR_NOTIFY_MODE must be one of: always, failures_only, "
        "failures_and_recovery, first_24h_all"
    )


def update_state(state: dict[str, Any], result: ProbeResult, notified: bool) -> None:
    state["last_status"] = "ok" if result.ok else "failure"
    state["last_checked_at"] = result.finished_at
    state["last_duration_seconds"] = round(result.duration_seconds, 3)
    state["last_error_type"] = result.error_type
    state["last_error_message"] = result.error_message
    state["last_response_preview"] = result.response_preview
    state["last_stage"] = result.stage
    state["last_category"] = result.category
    if result.ok:
        state["last_success_at"] = result.finished_at
        state["last_success_room"] = result.room_name
    if notified:
        state["last_notified_at"] = result.finished_at
        state["last_notified_status"] = state["last_status"]


def run_health_check(config: MonitorConfig) -> None:
    with request.urlopen(config.health_url, timeout=config.health_timeout_seconds) as resp:
        if resp.status != 200:
            raise RuntimeError(f"health check returned HTTP {resp.status}")
        body = resp.read(4096).decode("utf-8", errors="replace")
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"health check did not return JSON: {body[:120]!r}") from exc
    if payload.get("status") != "healthy":
        raise RuntimeError(f"health check returned non-healthy payload: {payload!r}")


def _validate_livekit_config(config: MonitorConfig) -> None:
    missing = [
        name
        for name, value in (
            ("LIVEKIT_URL", config.livekit_url),
            ("LIVEKIT_API_KEY", config.livekit_api_key),
            ("LIVEKIT_API_SECRET", config.livekit_api_secret),
            ("MONITOR_AGENT_NAME/AGENT_NAME", config.agent_name),
            ("MONITOR_CALLED_PHONE/AGENT_PHONE_NUMBER", config.called_phone),
        )
        if not value
    ]
    if missing:
        raise RuntimeError("missing LiveKit monitor configuration: " + ", ".join(missing))


async def run_livekit_text_probe(
    config: MonitorConfig, stage: _StageTracker | None = None
) -> dict[str, str]:
    if stage is None:
        stage = _StageTracker("livekit_probe")

    stage.set("livekit_validate_config")
    _validate_livekit_config(config)

    from livekit import api, rtc
    from livekit.agents.types import ATTRIBUTE_SIMULATOR, TOPIC_CHAT, TOPIC_TRANSCRIPTION

    room_name = f"voice-agent-monitor-{int(_now())}-{uuid.uuid4().hex[:6]}"
    stage.set_room_name(room_name)
    identity = f"monitor-{uuid.uuid4().hex[:8]}"
    agent_identity: dict[str, str] = {"value": ""}
    probe_sent: dict[str, bool] = {"value": False}
    events: list[str] = []
    response_fut: asyncio.Future[dict[str, str]] = asyncio.get_running_loop().create_future()
    room = rtc.Room()

    stage.set("livekit_api_init")
    async with api.LiveKitAPI(
        url=config.livekit_url,
        api_key=config.livekit_api_key,
        api_secret=config.livekit_api_secret,
    ) as lk:
        try:
            stage.set("livekit_room_create")
            await lk.room.create_room(
                api.CreateRoomRequest(
                    name=room_name,
                    empty_timeout=60,
                    departure_timeout=15,
                    metadata=json.dumps(
                        {
                            "called_phone": config.called_phone,
                            "source": "agent_monitor",
                            "skip_call_tracking": True,
                        }
                    ),
                )
            )

            def on_participant_connected(participant: rtc.RemoteParticipant) -> None:
                events.append(
                    f"participant_connected:{participant.identity}:kind={participant.kind}"
                )
                if participant.identity != identity and not agent_identity["value"]:
                    agent_identity["value"] = participant.identity

            def on_track_subscribed(
                track: rtc.Track,
                publication: rtc.RemoteTrackPublication,
                participant: rtc.RemoteParticipant,
            ) -> None:
                del publication
                events.append(f"track_subscribed:{participant.identity}:kind={track.kind}")

            def on_transcription(reader: rtc.TextStreamReader, participant_identity: str) -> None:
                async def read_text() -> None:
                    try:
                        text = await reader.read_all()
                        preview = text[:120].replace("\n", " ")
                        events.append(f"transcription:{participant_identity}:{preview}")
                        if (
                            probe_sent["value"]
                            and participant_identity != identity
                            and text.strip().lower() != config.probe_text.lower()
                            and not response_fut.done()
                        ):
                            response_fut.set_result(
                                {
                                    "type": "transcription",
                                    "from": participant_identity,
                                    "text": text,
                                }
                            )
                    except Exception as exc:  # pragma: no cover - defensive event callback
                        events.append(f"transcription_error:{type(exc).__name__}:{exc}")

                asyncio.create_task(read_text())

            room.on("participant_connected", on_participant_connected)
            room.on("track_subscribed", on_track_subscribed)
            room.register_text_stream_handler(TOPIC_TRANSCRIPTION, on_transcription)

            stage.set("livekit_token_mint")
            token = (
                api.AccessToken(config.livekit_api_key, config.livekit_api_secret)
                .with_identity(identity)
                .with_name("Agent Monitor")
                .with_attributes({ATTRIBUTE_SIMULATOR: "true"})
                .with_grants(
                    api.VideoGrants(
                        room_join=True,
                        room=room_name,
                        can_publish=True,
                        can_subscribe=True,
                        can_publish_data=True,
                    )
                )
                .to_jwt()
            )
            stage.set("livekit_room_connect")
            await room.connect(config.livekit_url, token)
            events.append("monitor_connected")

            stage.set("livekit_dispatch")
            dispatch = await lk.agent_dispatch.create_dispatch(
                api.CreateAgentDispatchRequest(
                    agent_name=config.agent_name,
                    room=room_name,
                    metadata=json.dumps(
                        {
                            "called_phone": config.called_phone,
                            "source": "agent_monitor",
                            "skip_call_tracking": True,
                            "probe": config.probe_text,
                        }
                    ),
                )
            )
            events.append(f"dispatch_created:{getattr(dispatch, 'id', '')}")

            stage.set("agent_join_wait")
            deadline = _now() + config.agent_join_timeout_seconds
            while _now() < deadline and not agent_identity["value"]:
                await asyncio.sleep(1)
            if not agent_identity["value"]:
                raise TimeoutError("agent participant did not join monitor room")

            stage.set("probe_send")
            await asyncio.sleep(config.connect_grace_seconds)
            probe_sent["value"] = True
            await room.local_participant.send_text(config.probe_text, topic=TOPIC_CHAT)
            events.append(f"sent_text:{config.probe_text}")

            stage.set("response_wait")
            result = await asyncio.wait_for(response_fut, timeout=config.response_timeout_seconds)
            stage.set("success")
            return {
                "room_name": room_name,
                "agent_identity": agent_identity["value"],
                "response_type": result["type"],
                "response_preview": result.get("text", "")[:240].replace("\n", " "),
                "events": json.dumps(events[-12:], ensure_ascii=False),
            }
        finally:
            try:
                await room.disconnect()
            except Exception as exc:  # pragma: no cover - cleanup only
                print(f"monitor cleanup warning: room disconnect failed: {exc}", file=sys.stderr)
            try:
                await lk.room.delete_room(api.DeleteRoomRequest(room=room_name))
            except Exception as exc:  # pragma: no cover - cleanup only
                print(f"monitor cleanup warning: room delete failed: {exc}", file=sys.stderr)


async def run_probe(config: MonitorConfig) -> ProbeResult:
    started_at = _now()
    stage = _StageTracker("health_check")
    livekit_project = _extract_livekit_project(config.livekit_url)
    try:
        run_health_check(config)
        stage.set("livekit_probe")
        livekit_result = await run_livekit_text_probe(config, stage)
        return ProbeResult(
            ok=True,
            status="ok",
            started_at=started_at,
            finished_at=_now(),
            health_ok=True,
            room_name=livekit_result["room_name"],
            agent_identity=livekit_result["agent_identity"],
            response_type=livekit_result["response_type"],
            response_preview=livekit_result["response_preview"],
            details=livekit_result["events"],
            stage="success",
            livekit_project=livekit_project,
        )
    except Exception as exc:
        classification = classify_failure(stage.current, type(exc).__name__, str(exc))
        return ProbeResult(
            ok=False,
            status="failure",
            started_at=started_at,
            finished_at=_now(),
            error_type=type(exc).__name__,
            error_message=str(exc),
            details=traceback.format_exc(limit=8),
            stage=stage.current,
            category=classification.category,
            summary=classification.summary,
            remediation=classification.remediation,
            livekit_project=livekit_project,
            room_name=stage.room_name,
        )


def build_telegram_text(result: ProbeResult, state: dict[str, Any] | None = None) -> str:
    """Compose the Telegram alert body.

    The first line is kept as ``Agent monitor: <STATUS>`` so any existing
    filters/forwards keep working; the body underneath is the new richer format.
    """
    state = state or {}
    if result.ok:
        return _build_ok_text(result)
    return _build_failure_text(result, state)


def _build_ok_text(result: ProbeResult) -> str:
    lines = [
        "✅ Agent monitor: OK",
        "",
        f"Host: {socket.gethostname()}",
        f"Time: {_iso(result.finished_at)}",
        f"Duration: {result.duration_seconds:.2f}s",
    ]
    if result.livekit_project:
        lines.append(f"LiveKit project: {result.livekit_project}")
    if result.room_name:
        lines.append(f"Room: {result.room_name}")
    if result.agent_identity:
        lines.append(f"Agent: {result.agent_identity}")
    if result.response_type:
        lines.append(f"Response type: {result.response_type}")
    if result.response_preview:
        lines.append(f"Response: {result.response_preview}")
    return "\n".join(lines)


def _build_failure_text(result: ProbeResult, state: dict[str, Any]) -> str:
    raw_msg = (result.error_message or "").strip().replace("\n", " ")
    if len(raw_msg) > 300:
        raw_msg = raw_msg[:297] + "..."

    lines = [
        "🚨 Agent monitor: FAILURE",
        "",
        "▸ What's wrong:",
        f"  {result.summary or 'Unclassified failure.'}",
        "",
        "▸ Where it failed:",
        f"  Stage: {result.stage or 'unknown'}",
    ]
    if result.category:
        lines.append(f"  Category: {result.category}")
    if result.livekit_project:
        lines.append(f"  LiveKit project: {result.livekit_project}")
    if result.room_name:
        # Included so the agent_not_joined / agent_not_responded remediation
        # steps (which tell the reader to grep agent logs for this room) are
        # actually actionable.
        lines.append(f"  Room: {result.room_name}")
    lines.append(f"  Raw error: {result.error_type or 'Error'} — {raw_msg or '(no message)'}")

    if result.remediation:
        lines.append("")
        lines.append("▸ What to do:")
        for i, step in enumerate(result.remediation, 1):
            lines.append(f"  {i}. {step}")

    lines.append("")
    lines.append("▸ Context:")
    lines.append(f"  Host: {socket.gethostname()}")
    lines.append(f"  Time: {_iso(result.finished_at)}")
    lines.append(f"  Probe duration: {result.duration_seconds:.2f}s")

    last_success_at = state.get("last_success_at")
    if isinstance(last_success_at, (int, float)) and last_success_at > 0:
        ago = _human_delta(result.finished_at - last_success_at)
        lines.append(f"  Last successful probe: {_iso(last_success_at)} ({ago} ago)")
    else:
        lines.append("  Last successful probe: (none recorded)")

    return "\n".join(lines)


def build_telegram_payload(config: MonitorConfig, text: str) -> tuple[str, bytes]:
    if not config.telegram_bot_token:
        raise RuntimeError("MONITOR_TELEGRAM_BOT_TOKEN/TELEGRAM_BOT_TOKEN is not set")
    if not config.telegram_chat_id:
        raise RuntimeError("MONITOR_TELEGRAM_CHAT_ID/TELEGRAM_CHAT_ID is not set")

    url = f"https://api.telegram.org/bot{config.telegram_bot_token}/sendMessage"
    payload: dict[str, str] = {
        "chat_id": config.telegram_chat_id,
        "text": text,
        "disable_web_page_preview": "true",
    }
    if config.telegram_thread_id:
        payload["message_thread_id"] = config.telegram_thread_id
    return url, parse.urlencode(payload).encode("utf-8")


def redact_secret(text: str, secret: str) -> str:
    if not secret:
        return text
    return text.replace(secret, "<redacted>")


def send_telegram(
    config: MonitorConfig,
    result: ProbeResult,
    state: dict[str, Any] | None = None,
) -> None:
    url, payload = build_telegram_payload(config, build_telegram_text(result, state))
    try:
        with request.urlopen(url, data=payload, timeout=20) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        safe_body = redact_secret(body[:500], config.telegram_bot_token)
        raise RuntimeError(f"Telegram returned HTTP {exc.code}: {safe_body}") from exc
    except error.URLError as exc:
        safe_error = redact_secret(str(exc), config.telegram_bot_token)
        raise RuntimeError(f"Telegram request failed: {safe_error}") from exc
    data = json.loads(body)
    if not data.get("ok"):
        safe_data = redact_secret(repr(data), config.telegram_bot_token)
        raise RuntimeError(f"Telegram send failed: {safe_data}")


async def run_once(config: MonitorConfig) -> int:
    state = load_state(config.state_path)
    ensure_state_started(state, _now())
    result = await run_probe(config)
    notified = False

    if should_notify(result, state, config):
        # Send before update_state so the "Last successful probe" line reflects
        # the PREVIOUS success, not the current run.
        send_telegram(config, result, state)
        notified = True

    update_state(state, result, notified)
    save_state(config.state_path, state)

    summary = {
        "ok": result.ok,
        "status": result.status,
        "stage": result.stage,
        "category": result.category,
        "duration_seconds": round(result.duration_seconds, 3),
        "notified": notified,
        "response_preview": result.response_preview,
        "error_type": result.error_type,
        "error_message": result.error_message,
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if result.ok else 1


async def run_loop(config: MonitorConfig) -> int:
    if config.initial_delay_seconds > 0:
        await asyncio.sleep(config.initial_delay_seconds)

    while True:
        try:
            await run_once(config)
        except Exception as exc:
            print(f"monitor loop error: {type(exc).__name__}: {exc}", file=sys.stderr)
            traceback.print_exc(limit=8)
        await asyncio.sleep(config.interval_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor the LiveKit voice agent")
    parser.add_argument(
        "mode",
        nargs="?",
        choices=("once", "loop"),
        default="once",
        help="run one probe or run forever on MONITOR_INTERVAL_SECONDS",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config()
    if args.mode == "loop":
        return asyncio.run(run_loop(config))
    return asyncio.run(run_once(config))


if __name__ == "__main__":
    raise SystemExit(main())
