from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from lifecycle.session_events import setup_session_events


def _event(state: str) -> SimpleNamespace:
    """LiveKit AgentStateChangedEvent / UserStateChangedEvent shape:
    handlers read `.new_state` off the event object."""
    return SimpleNamespace(new_state=state)


class _FakeSession:
    def __init__(self):
        self.handlers = {}
        self.current_agent = None

    def on(self, event_name):
        def _decorator(fn):
            self.handlers[event_name] = fn
            return fn

        return _decorator


def test_session_events_use_current_agent_after_handoff():
    session = _FakeSession()

    initial_agent = MagicMock()
    initial_agent._error_handler = MagicMock()
    initial_agent._start_silence_monitor = MagicMock()
    active_agent = MagicMock()
    active_agent._error_handler = MagicMock()
    active_agent._start_silence_monitor = MagicMock()

    session.current_agent = initial_agent
    setup_session_events(session=session, agent=initial_agent, config=MagicMock())

    # Simulate handoff: event handlers must target current_agent, not captured initial agent.
    session.current_agent = active_agent
    session.handlers["agent_state_changed"](_event("speaking"))
    session.handlers["user_state_changed"](_event("speaking"))

    assert active_agent._error_handler.set_agent_speaking.call_args_list[0].args == (True,)
    assert active_agent._error_handler.set_user_speaking.call_args_list[0].args == (True,)
    active_agent._error_handler.reset_silence_timer.assert_called_once()
    active_agent._start_silence_monitor.assert_called()
    initial_agent._error_handler.set_agent_speaking.assert_not_called()
    initial_agent._error_handler.set_user_speaking.assert_not_called()
    initial_agent._error_handler.reset_silence_timer.assert_not_called()


def test_agent_state_non_speaking_clears_agent_speaking_flag():
    session = _FakeSession()
    agent = MagicMock()
    agent._error_handler = MagicMock()
    agent._start_silence_monitor = MagicMock()
    session.current_agent = agent
    setup_session_events(session=session, agent=agent, config=MagicMock())

    session.handlers["agent_state_changed"](_event("speaking"))
    session.handlers["agent_state_changed"](_event("thinking"))

    assert agent._error_handler.set_agent_speaking.call_args_list[-1].args == (True,)


@pytest.mark.asyncio
async def test_agent_state_listening_clears_agent_speaking_flag():
    session = _FakeSession()
    # No `wait_for_inactive` → the deferred `_mark_idle_after_playout`
    # task takes the fallback branch and clears the flag after a tiny sleep.
    agent = MagicMock()
    agent._error_handler = MagicMock()
    agent._start_silence_monitor = MagicMock()
    session.current_agent = agent
    setup_session_events(session=session, agent=agent, config=MagicMock())

    session.handlers["agent_state_changed"](_event("speaking"))
    session.handlers["agent_state_changed"](_event("listening"))
    # The listening → idle flip is scheduled via asyncio.create_task and
    # completes after the small fallback sleep; wait for it.
    await asyncio.sleep(0.3)

    assert agent._error_handler.set_agent_speaking.call_args_list[-1].args == (False,)


def test_user_state_non_speaking_clears_user_speaking_flag():
    session = _FakeSession()
    agent = MagicMock()
    agent._error_handler = MagicMock()
    agent._start_silence_monitor = MagicMock()
    session.current_agent = agent
    setup_session_events(session=session, agent=agent, config=MagicMock())

    session.handlers["user_state_changed"](_event("speaking"))
    session.handlers["user_state_changed"](_event("listening"))

    assert agent._error_handler.set_user_speaking.call_args_list[0].args == (True,)
    assert agent._error_handler.set_user_speaking.call_args_list[-1].args == (False,)
