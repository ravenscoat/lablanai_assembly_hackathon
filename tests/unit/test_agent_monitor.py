from __future__ import annotations

import json
from pathlib import Path
from urllib import parse

from scripts.agent_monitor import (
    FailureClassification,
    MonitorConfig,
    ProbeResult,
    _extract_livekit_project,
    _human_delta,
    build_telegram_payload,
    build_telegram_text,
    classify_failure,
    redact_secret,
    should_notify,
    update_state,
    within_first_success_window,
)


def make_config(tmp_path: Path, *, notify_mode: str = "first_24h_all") -> MonitorConfig:
    return MonitorConfig(
        health_url="http://agent:8082/health",
        interval_seconds=3600,
        initial_delay_seconds=30,
        state_path=tmp_path / "state.json",
        notify_mode=notify_mode,
        first_success_hours=24,
        telegram_bot_token="token",
        telegram_chat_id="-1003482894819",
        telegram_thread_id="5547",
        livekit_url="wss://example.livekit.cloud",
        livekit_api_key="key",
        livekit_api_secret="secret",
        agent_name="yoshlar_voice_agent_prod",
        called_phone="+923781225396",
        probe_text="salom",
        health_timeout_seconds=5,
        agent_join_timeout_seconds=60,
        response_timeout_seconds=60,
        connect_grace_seconds=5,
    )


def make_result(
    *,
    ok: bool,
    finished_at: float = 1_000.0,
    stage: str = "",
    category: str = "",
    summary: str = "",
    remediation: tuple[str, ...] = (),
    livekit_project: str = "",
    error_type: str | None = None,
    error_message: str | None = None,
) -> ProbeResult:
    return ProbeResult(
        ok=ok,
        status="ok" if ok else "failure",
        started_at=finished_at - 3,
        finished_at=finished_at,
        response_preview="Salom. Sizga qanday yordam bera olaman?" if ok else "",
        error_type=("" if ok else "TimeoutError") if error_type is None else error_type,
        error_message=(
            ("" if ok else "agent did not respond") if error_message is None else error_message
        ),
        stage=stage,
        category=category,
        summary=summary,
        remediation=remediation,
        livekit_project=livekit_project,
    )


def test_first_success_window_uses_state_start_time(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    state = {"started_at": 100.0}

    assert within_first_success_window(state, 100.0 + 23 * 3600, config.first_success_hours)
    assert not within_first_success_window(state, 100.0 + 25 * 3600, config.first_success_hours)


def test_notifies_success_during_first_24_hours(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    state = {"started_at": 100.0, "last_status": "ok"}
    result = make_result(ok=True, finished_at=100.0 + 3600)

    assert should_notify(result, state, config)


def test_suppresses_success_after_window_when_already_ok(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    state = {"started_at": 100.0, "last_status": "ok"}
    result = make_result(ok=True, finished_at=100.0 + 25 * 3600)

    assert not should_notify(result, state, config)


def test_notifies_failure_after_window(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    state = {"started_at": 100.0, "last_status": "ok"}
    result = make_result(ok=False, finished_at=100.0 + 25 * 3600)

    assert should_notify(result, state, config)


def test_failures_only_mode_does_not_notify_recovery(tmp_path: Path) -> None:
    config = make_config(tmp_path, notify_mode="failures_only")
    state = {"started_at": 100.0, "last_status": "failure"}
    result = make_result(ok=True, finished_at=100.0 + 25 * 3600)

    assert not should_notify(result, state, config)


def test_telegram_payload_includes_status_alert_thread(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    url, body = build_telegram_payload(config, "hello")

    assert url == "https://api.telegram.org/bottoken/sendMessage"
    payload = dict(parse.parse_qsl(body.decode("utf-8")))
    assert payload["chat_id"] == "-1003482894819"
    assert payload["message_thread_id"] == "5547"
    assert payload["text"] == "hello"


def test_redact_secret_removes_token_from_error_text() -> None:
    token = "123456:SECRET"
    text = f"https://api.telegram.org/bot{token}/sendMessage failed"

    assert redact_secret(text, token) == "https://api.telegram.org/bot<redacted>/sendMessage failed"


def test_update_state_records_result(tmp_path: Path) -> None:
    del tmp_path
    state: dict[str, object] = {}
    result = make_result(ok=False, finished_at=1234.0)

    update_state(state, result, notified=True)

    assert state["last_status"] == "failure"
    assert state["last_checked_at"] == 1234.0
    assert state["last_notified_status"] == "failure"
    assert state["last_error_type"] == "TimeoutError"


def test_classify_failure_health_check_503_is_unhealthy_not_livekit() -> None:
    """Regression test for the precedence bug raised in PR #110 review.

    ``urlopen`` raises ``HTTPError('HTTP Error 503: Service Unavailable')`` when
    the agent's own /health endpoint returns 5xx. The classifier must report
    that as an agent health-check problem, NOT a LiveKit server outage.
    """
    cls = classify_failure("health_check", "HTTPError", "HTTP Error 503: Service Unavailable")

    assert cls.category == "health_check_unhealthy"
    assert all("livekit" not in step.lower() for step in cls.remediation)
    assert all("status.livekit.io" not in step for step in cls.remediation)


def test_classify_failure_health_check_429_is_unhealthy_not_livekit() -> None:
    cls = classify_failure("health_check", "HTTPError", "HTTP Error 429: Too Many Requests")

    assert cls.category == "health_check_unhealthy"


def test_classify_failure_health_check_403_is_unhealthy_not_livekit() -> None:
    cls = classify_failure("health_check", "HTTPError", "HTTP Error 403: Forbidden")

    assert cls.category == "health_check_unhealthy"


def test_classify_failure_health_check_connection_refused_is_unreachable() -> None:
    """``connection refused`` matches the network-hint list, but the
    ``health_check`` stage must take precedence and point at the agent."""
    cls = classify_failure(
        "health_check", "URLError", "<urlopen error [Errno 111] Connection refused>"
    )

    assert cls.category == "health_check_unreachable"


def test_build_telegram_text_failure_includes_room_name_when_present() -> None:
    """Address PR #110 review issue 2: the remediation tells readers to grep
    agent logs for the room name — the alert body must therefore include it
    when it's known."""
    finished_at = 1_700_000_000.0
    result = make_result(
        ok=False,
        finished_at=finished_at,
        stage="agent_join_wait",
        category="agent_not_joined",
        summary="Agent worker did not pick up the monitor dispatch within the join timeout.",
        remediation=(
            "Look up the room name from this alert in agent logs to see the dispatch trail.",
        ),
        error_type="TimeoutError",
        error_message="agent participant did not join monitor room",
    )
    result.room_name = "navai-monitor-1781025772-7ca750"

    text = build_telegram_text(result)

    assert "Room: navai-monitor-1781025772-7ca750" in text


def test_state_payload_is_json_serializable(tmp_path: Path) -> None:
    del tmp_path
    state: dict[str, object] = {}
    update_state(state, make_result(ok=True), notified=False)

    json.dumps(state)


def test_update_state_records_last_success_at_only_on_ok() -> None:
    state: dict[str, object] = {}

    update_state(state, make_result(ok=True, finished_at=1000.0), notified=False)
    assert state["last_success_at"] == 1000.0

    update_state(state, make_result(ok=False, finished_at=1500.0), notified=True)
    # last_success_at must NOT be overwritten by a failure — it's the anchor
    # the alert uses to tell readers how long the agent has been down.
    assert state["last_success_at"] == 1000.0
    assert state["last_status"] == "failure"
    assert state["last_stage"] == ""  # default for the legacy make_result(ok=False)


def test_update_state_records_stage_and_category() -> None:
    state: dict[str, object] = {}
    result = make_result(
        ok=False,
        finished_at=2000.0,
        stage="livekit_room_connect",
        category="livekit_quota_exhausted",
    )
    update_state(state, result, notified=True)

    assert state["last_stage"] == "livekit_room_connect"
    assert state["last_category"] == "livekit_quota_exhausted"


# ---------------------------------------------------------------------------
# classify_failure
# ---------------------------------------------------------------------------


def test_classify_failure_livekit_quota_exhausted() -> None:
    cls = classify_failure(
        "livekit_room_connect",
        "ConnectError",
        "engine: signal failure: client error: 429 Too Many Requests - "
        "connection minutes limit exceeded. please contact the project owner.",
    )

    assert isinstance(cls, FailureClassification)
    assert cls.category == "livekit_quota_exhausted"
    assert "quota" in cls.summary.lower()
    assert any("cloud.livekit.io" in step for step in cls.remediation)


def test_classify_failure_plain_429_is_rate_limited_not_quota() -> None:
    cls = classify_failure(
        "livekit_dispatch", "Exception", "got HTTP 429 Too Many Requests (rate-limited)"
    )

    assert cls.category == "livekit_rate_limited"
    assert any("MONITOR_INTERVAL_SECONDS" in step for step in cls.remediation)


def test_classify_failure_auth_401() -> None:
    cls = classify_failure(
        "livekit_room_create", "TwirpError", '{"code":"unauthorized","msg":"invalid api key"}'
    )

    assert cls.category == "livekit_auth_failed"


def test_classify_failure_server_5xx() -> None:
    cls = classify_failure(
        "livekit_dispatch", "RuntimeError", "request failed with status 503 service unavailable"
    )

    assert cls.category == "livekit_server_error"
    assert any("status.livekit.io" in step for step in cls.remediation)


def test_classify_failure_agent_not_joined_by_stage() -> None:
    cls = classify_failure(
        "agent_join_wait", "TimeoutError", "agent participant did not join monitor room"
    )

    assert cls.category == "agent_not_joined"


def test_classify_failure_agent_not_responded_by_stage() -> None:
    cls = classify_failure("response_wait", "TimeoutError", "")

    assert cls.category == "agent_not_responded"


def test_classify_failure_health_unhealthy() -> None:
    cls = classify_failure(
        "health_check",
        "RuntimeError",
        "health check returned non-healthy payload: {'status': 'degraded'}",
    )

    assert cls.category == "health_check_unhealthy"


def test_classify_failure_health_unreachable() -> None:
    cls = classify_failure(
        "health_check", "URLError", "<urlopen error [Errno 111] Connection refused>"
    )

    # "connection refused" appears in the network-hints list; health_check stage
    # is more specific so it takes precedence and points at the agent container.
    assert cls.category == "health_check_unreachable"


def test_classify_failure_network_error_outside_health_check() -> None:
    cls = classify_failure("livekit_api_init", "OSError", "[Errno -2] Name or service not known")

    assert cls.category == "network_error"
    assert "livekit_api_init" in cls.summary


def test_classify_failure_config_missing() -> None:
    cls = classify_failure(
        "livekit_validate_config",
        "RuntimeError",
        "missing LiveKit monitor configuration: LIVEKIT_API_KEY, LIVEKIT_API_SECRET",
    )

    assert cls.category == "config_missing"


def test_classify_failure_unknown_fallback() -> None:
    cls = classify_failure("livekit_room_connect", "Exception", "totally unknown weirdness")

    assert cls.category == "unknown_error"
    assert "livekit_room_connect" in cls.summary
    assert cls.remediation  # remediation is never empty


# ---------------------------------------------------------------------------
# _extract_livekit_project
# ---------------------------------------------------------------------------


def test_extract_livekit_project_wss() -> None:
    assert (
        _extract_livekit_project("wss://debt-collection-agent-tzxnyg91.livekit.cloud")
        == "debt-collection-agent-tzxnyg91"
    )


def test_extract_livekit_project_https() -> None:
    assert _extract_livekit_project("https://my-project.livekit.cloud/something") == "my-project"


def test_extract_livekit_project_with_port() -> None:
    assert _extract_livekit_project("wss://abc-xyz.livekit.cloud:443") == "abc-xyz"


def test_extract_livekit_project_returns_empty_for_self_hosted() -> None:
    assert _extract_livekit_project("wss://livekit.internal:7880") == ""
    assert _extract_livekit_project("") == ""
    assert _extract_livekit_project("not-a-url") == ""


# ---------------------------------------------------------------------------
# _human_delta
# ---------------------------------------------------------------------------


def test_human_delta_under_minute() -> None:
    assert _human_delta(0) == "0s"
    assert _human_delta(45) == "45s"
    assert _human_delta(-5) == "0s"  # never negative


def test_human_delta_minutes() -> None:
    assert _human_delta(60) == "1m 0s"
    assert _human_delta(125) == "2m 5s"


def test_human_delta_hours() -> None:
    assert _human_delta(3 * 3600 + 60) == "3h 1m"


def test_human_delta_days() -> None:
    assert _human_delta(2 * 86400 + 5 * 3600) == "2d 5h"


# ---------------------------------------------------------------------------
# build_telegram_text — new richer format
# ---------------------------------------------------------------------------


def test_build_telegram_text_success_keeps_legacy_first_line() -> None:
    result = make_result(ok=True)

    text = build_telegram_text(result)
    first_line = text.splitlines()[0]
    # First line must keep the "Agent monitor: OK" tail so any alert
    # filters / forwards based on that string keep working.
    assert first_line.endswith("Agent monitor: OK")


def test_build_telegram_text_failure_includes_classification_and_remediation() -> None:
    finished_at = 1_700_000_000.0
    result = make_result(
        ok=False,
        finished_at=finished_at,
        stage="livekit_room_connect",
        category="livekit_quota_exhausted",
        summary="LiveKit Cloud monthly connection-minutes quota exhausted.",
        remediation=(
            "Top up the LiveKit Cloud plan or buy more minutes at https://cloud.livekit.io",
            "Real calls share this project and will also return 429 until the quota resets.",
        ),
        livekit_project="debt-collection-agent-tzxnyg91",
        error_type="ConnectError",
        error_message="429 Too Many Requests - connection minutes limit exceeded.",
    )
    state = {"last_success_at": finished_at - (3 * 3600 + 60)}

    text = build_telegram_text(result, state)

    # Stable header for filters.
    assert "Agent monitor: FAILURE" in text.splitlines()[0]
    # Sectioning.
    assert "What's wrong" in text
    assert "Where it failed" in text
    assert "What to do" in text
    assert "Context" in text
    # Concrete content.
    assert "LiveKit Cloud monthly connection-minutes quota exhausted." in text
    assert "Stage: livekit_room_connect" in text
    assert "Category: livekit_quota_exhausted" in text
    assert "LiveKit project: debt-collection-agent-tzxnyg91" in text
    assert "ConnectError" in text
    assert "https://cloud.livekit.io" in text
    # Time-since-last-success.
    assert "Last successful probe" in text
    assert "3h 1m ago" in text


def test_build_telegram_text_failure_without_state_shows_no_previous_success() -> None:
    result = make_result(
        ok=False,
        stage="health_check",
        category="health_check_unreachable",
        summary="Could not reach the agent /health endpoint.",
        remediation=("Container may be down.",),
    )

    text = build_telegram_text(result)

    assert "Last successful probe: (none recorded)" in text


def test_build_telegram_text_failure_truncates_long_raw_error() -> None:
    long_msg = "x" * 1000
    result = make_result(
        ok=False,
        stage="livekit_room_connect",
        category="unknown_error",
        summary="Unclassified.",
        remediation=("look",),
        error_type="Exception",
        error_message=long_msg,
    )

    text = build_telegram_text(result)

    assert "..." in text
    assert "x" * 1000 not in text  # full untruncated message is NOT included


def test_build_telegram_text_message_stays_under_telegram_limit() -> None:
    # Worst-case: 5 remediation steps + long raw error + everything filled.
    result = make_result(
        ok=False,
        stage="livekit_room_connect",
        category="livekit_quota_exhausted",
        summary="LiveKit Cloud monthly connection-minutes quota exhausted on a project with an unusually long name.",
        remediation=tuple(f"step {i}: " + "x" * 200 for i in range(5)),
        livekit_project="some-very-long-project-slug-that-someone-came-up-with",
        error_type="ConnectError",
        error_message="a" * 500,
    )
    state = {"last_success_at": 0}

    text = build_telegram_text(result, state)

    # Telegram sendMessage hard limit is 4096 chars.
    assert len(text) < 4096
