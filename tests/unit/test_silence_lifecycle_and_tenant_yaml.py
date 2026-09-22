from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from agents.tenant_agent import TenantAgent
from config.schema import TenantConfig, TenantIdentity
from lifecycle.shutdown import register_shutdown_callbacks
from resilience.error_recovery import ErrorRecoveryHandler


@pytest.mark.asyncio
async def test_start_silence_monitor_uses_configured_interval():
    cfg = TenantConfig.model_validate(
        {
            "tenant": {"id": "t1", "slug": "test"},
            "silence": {"prompt_timeout": 15.0, "goodbye_timeout": 30.0, "monitor_interval": 1.25},
            "responses": {"silence_goodbye": "Xayr!"},
        }
    )
    agent = TenantAgent.__new__(TenantAgent)
    agent._pending_post_handoff_prompt = None
    agent.config = cfg
    agent._session = object()
    type(agent).session = property(lambda self: self._session)
    agent._error_handler = MagicMock()
    agent._silence_monitor = None

    with patch("resilience.silence_monitor.SilenceMonitor") as monitor_cls:
        monitor_inst = MagicMock()
        monitor_cls.return_value = monitor_inst
        agent._start_silence_monitor()

    assert monitor_cls.call_args.kwargs["check_interval"] == 1.25
    assert monitor_cls.call_args.kwargs["goodbye_message"] == "Xayr!"
    monitor_inst.start.assert_called_once()
    agent._error_handler.reset_silence_timer.assert_not_called()


@pytest.mark.asyncio
async def test_start_silence_monitor_uses_language_specific_goodbye():
    cfg = TenantConfig.model_validate(
        {
            "tenant": {"id": "t1", "slug": "test"},
            "responses": {
                "silence_goodbye": "fallback",
                "silence_goodbye_by_language": {"uz": "Xayr uz", "ru": "Poka ru"},
            },
        }
    )
    agent = TenantAgent.__new__(TenantAgent)
    agent._pending_post_handoff_prompt = None
    agent.config = cfg
    agent.language = "ru"
    agent._session = object()
    type(agent).session = property(lambda self: self._session)
    agent._error_handler = MagicMock()
    agent._silence_monitor = None

    with patch("resilience.silence_monitor.SilenceMonitor") as monitor_cls:
        monitor_inst = MagicMock()
        monitor_cls.return_value = monitor_inst
        agent._start_silence_monitor()

    assert monitor_cls.call_args.kwargs["goodbye_message"] == "Poka ru"


@pytest.mark.asyncio
async def test_on_enter_arms_silence_after_fallback_greeting():
    cfg = TenantConfig.model_validate({"tenant": {"id": "t1", "slug": "test"}})
    agent = TenantAgent.__new__(TenantAgent)
    agent._pending_post_handoff_prompt = None
    agent.config = cfg
    agent.call_id = "c1"
    agent.language = "uz"
    agent._session_state = SimpleNamespace(_call_start=0)
    agent._is_post_handoff = False
    agent.conversation_history = []
    agent._caller_history = None
    agent._greet_monolingual = AsyncMock()
    agent._arm_silence_timer = MagicMock()
    agent._have_all_greeting_audio = MagicMock(return_value=False)

    await agent.on_enter()

    agent._greet_monolingual.assert_awaited_once()
    agent._arm_silence_timer.assert_called_once()


@pytest.mark.asyncio
async def test_on_enter_starts_silence_monitor_for_post_handoff_agent():
    cfg = TenantConfig.model_validate({"tenant": {"id": "t1", "slug": "test"}})
    agent = TenantAgent.__new__(TenantAgent)
    agent._pending_post_handoff_prompt = None
    agent.config = cfg
    agent.call_id = "c1"
    agent.language = "uz"
    agent._session_state = SimpleNamespace(_call_start=0)
    agent._is_post_handoff = True
    agent.conversation_history = []
    agent._start_silence_monitor = MagicMock()
    agent._arm_silence_timer = MagicMock()

    await agent.on_enter()

    agent._start_silence_monitor.assert_called_once()
    agent._arm_silence_timer.assert_called_once()


@pytest.mark.asyncio
async def test_on_enter_arms_silence_after_prerendered_greeting():
    cfg = TenantConfig.model_validate(
        {
            "tenant": {"id": "t1", "slug": "test"},
            "languages": {"default": "uz", "available": ["uz", "ru"]},
        }
    )
    agent = TenantAgent.__new__(TenantAgent)
    agent._pending_post_handoff_prompt = None
    agent.config = cfg
    agent.call_id = "c1"
    agent.language = "uz"
    agent._session_state = SimpleNamespace(_call_start=0)
    agent._is_post_handoff = False
    agent.conversation_history = []
    agent._caller_history = None
    agent._play_prerendered_greeting = AsyncMock()
    agent._arm_silence_timer = MagicMock()
    agent._have_all_greeting_audio = MagicMock(return_value=True)

    await agent.on_enter()

    agent._play_prerendered_greeting.assert_awaited_once()
    agent._arm_silence_timer.assert_called_once()


def test_silence_prompt_restarts_goodbye_window():
    cfg = TenantConfig.model_validate(
        {
            "tenant": {"id": "t1", "slug": "test"},
            "silence": {"prompt_timeout": 5.0, "goodbye_timeout": 10.0},
            "responses": {"silence_prompt": "Ping?"},
        }
    )
    handler = ErrorRecoveryHandler(cfg)

    with patch("resilience.error_recovery.time.time", return_value=100.0):
        handler._state.last_user_speech_time = 94.0
        assert handler.check_silence_timeout() == "Ping?"

    # Silence prompt is agent speech; timer should restart only after speech ends.
    with patch("resilience.error_recovery.time.time", return_value=101.0):
        handler.set_agent_speaking(True)
        handler.set_agent_speaking(False)

    with patch("resilience.error_recovery.time.time", return_value=103.0):
        assert handler.check_silence_timeout() is None

    with patch("resilience.error_recovery.time.time", return_value=111.0):
        assert handler.check_silence_timeout() == "GOODBYE"


def test_silence_prompt_uses_active_language():
    cfg = TenantConfig.model_validate(
        {
            "tenant": {"id": "t1", "slug": "test"},
            "silence": {"prompt_timeout": 5.0, "goodbye_timeout": 10.0},
            "responses": {
                "silence_prompt": "fallback",
                "silence_prompt_by_language": {"uz": "Uz prompt", "ru": "Ru prompt"},
            },
        }
    )
    handler = ErrorRecoveryHandler(cfg)
    handler.set_language("ru")

    with patch("resilience.error_recovery.time.time", return_value=100.0):
        handler._state.last_user_speech_time = 94.0
        assert handler.check_silence_timeout() == "Ru prompt"


def test_silence_timer_uses_latest_of_user_or_agent_activity():
    cfg = TenantConfig.model_validate(
        {
            "tenant": {"id": "t1", "slug": "test"},
            "silence": {"prompt_timeout": 5.0, "goodbye_timeout": 10.0},
            "responses": {"silence_prompt": "Ping?"},
        }
    )
    handler = ErrorRecoveryHandler(cfg)

    # User spoke recently; stale older agent-finished markers must not shorten
    # the silence window.
    with patch("resilience.error_recovery.time.time", return_value=100.0):
        handler._state.last_user_speech_time = 98.0
        handler._state.agent_finished_speaking_time = 92.0
        assert handler.check_silence_timeout() is None

    with patch("resilience.error_recovery.time.time", return_value=103.0):
        assert handler.check_silence_timeout() == "Ping?"


def test_silence_timer_arms_from_agent_finish_transition():
    cfg = TenantConfig.model_validate(
        {
            "tenant": {"id": "t1", "slug": "test"},
            "silence": {"prompt_timeout": 5.0, "goodbye_timeout": 10.0},
            "responses": {"silence_prompt": "Ping?"},
        }
    )
    handler = ErrorRecoveryHandler(cfg)

    with patch("resilience.error_recovery.time.time", return_value=100.0):
        handler.set_agent_speaking(True)
        handler.set_agent_speaking(False)

    with patch("resilience.error_recovery.time.time", return_value=103.0):
        assert handler.check_silence_timeout() is None

    with patch("resilience.error_recovery.time.time", return_value=105.0):
        assert handler.check_silence_timeout() == "Ping?"


def test_silence_timer_uses_user_timestamp_without_agent_finish():
    cfg = TenantConfig.model_validate(
        {
            "tenant": {"id": "t1", "slug": "test"},
            "silence": {"prompt_timeout": 5.0, "goodbye_timeout": 10.0},
            "responses": {"silence_prompt": "Ping?"},
        }
    )
    handler = ErrorRecoveryHandler(cfg)

    with patch("resilience.error_recovery.time.time", return_value=100.0):
        handler._state.last_user_speech_time = 95.0
        handler._state.agent_finished_speaking_time = None
        assert handler.check_silence_timeout() == "Ping?"


def test_silence_timer_does_not_fire_while_user_is_speaking():
    cfg = TenantConfig.model_validate(
        {
            "tenant": {"id": "t1", "slug": "test"},
            "silence": {"prompt_timeout": 5.0, "goodbye_timeout": 10.0},
            "responses": {"silence_prompt": "Ping?"},
        }
    )
    handler = ErrorRecoveryHandler(cfg)

    with patch("resilience.error_recovery.time.time", return_value=100.0):
        handler._state.last_user_speech_time = 80.0
        handler.set_user_speaking(True)
        assert handler.check_silence_timeout() is None

    with patch("resilience.error_recovery.time.time", return_value=101.0):
        handler.set_user_speaking(False)
        assert handler.check_silence_timeout() == "GOODBYE"


def test_arm_silence_timer_clears_stale_agent_speaking():
    cfg = TenantConfig.model_validate({"tenant": {"id": "t1", "slug": "test"}})
    agent = TenantAgent.__new__(TenantAgent)
    agent._pending_post_handoff_prompt = None
    agent.config = cfg
    agent._error_handler = MagicMock()

    agent._arm_silence_timer()

    agent._error_handler.set_agent_speaking.assert_called_once_with(False)
    agent._error_handler.reset_silence_timer.assert_called_once()


@pytest.mark.asyncio
async def test_forward_to_operator_stops_silence_monitor():
    cfg = TenantConfig.model_validate(
        {
            "tenant": {"id": "t1", "slug": "test"},
            "transfer": {"enabled": True, "transfer_number": "sip:711@host"},
        }
    )
    agent = TenantAgent.__new__(TenantAgent)
    agent._pending_post_handoff_prompt = None
    agent.config = cfg
    agent._stop_silence_monitor = MagicMock()
    agent._operator_schedule_context = lambda: {
        "available_now": False,
        "day_uzbek": "Dushanba",
        "current_time": "10:00",
        "next_shift_time": "Seshanba soat 09:00",
    }

    await agent.do_forward_to_operator()
    agent._stop_silence_monitor.assert_called_once()


@pytest.mark.asyncio
async def test_shutdown_callback_stops_silence_monitor():
    callback_holder: dict[str, object] = {}

    class _Ctx:
        room = None

        def add_shutdown_callback(self, fn):
            callback_holder["fn"] = fn
            return fn

    cfg = TenantConfig(tenant=TenantIdentity(id="t1", slug="test"))
    agent = SimpleNamespace(
        _stop_silence_monitor=MagicMock(),
        _sf_instance=None,
        _sf_session_id=None,
        call_start_time=0,
        conversation_history=[],
        transferred=False,
        call_db_id=None,
        _telephony_tracker=None,
    )

    register_shutdown_callbacks(_Ctx(), agent, cfg)
    cb = callback_holder["fn"]
    assert cb is not None
    await cb()  # type: ignore[misc]

    agent._stop_silence_monitor.assert_called_once()


def test_example_tenant_yaml_has_explicit_silence_settings():
    repo_root = Path(__file__).resolve().parents[2]
    tenants_dir = repo_root / "configs" / "tenants"

    example = yaml.safe_load((tenants_dir / "example-tenant.yaml").read_text(encoding="utf-8"))

    # Tenants should declare explicit silence timeouts rather than relying on
    # implicit defaults (so the silence monitor behavior is reviewable per tenant).
    assert "silence" in example
    assert set(("prompt_timeout", "goodbye_timeout", "monitor_interval")).issubset(
        example["silence"].keys()
    )
