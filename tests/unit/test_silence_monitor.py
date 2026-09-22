from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from resilience.silence_monitor import SILENCE_GOODBYE, SilenceMonitor


def _make_session_with_say():
    """LiveKit session.say() is sync and returns a SpeechHandle with
    async wait_for_playout. Build a mock that matches that contract."""
    session = type("Session", (), {})()
    handle = MagicMock()
    handle.wait_for_playout = AsyncMock()
    session.say = MagicMock(return_value=handle)
    return session, handle


@pytest.mark.asyncio
async def test_silence_monitor_speaks_prompt():
    class _Handler:
        def __init__(self):
            self._called = 0
            self.agent_speaking_calls = []

        def check_silence_timeout(self):
            self._called += 1
            if self._called == 1:
                monitor._stop.set()
                return "Eshitiapsizmi?"
            return None

        def set_agent_speaking(self, speaking: bool):
            self.agent_speaking_calls.append(speaking)

    session, _handle = _make_session_with_say()
    monitor = SilenceMonitor(
        error_handler=_Handler(),
        session=session,
        check_interval=0.0,
        goodbye_message="x",
    )

    await monitor._monitor_loop()

    session.say.assert_called_once_with("Eshitiapsizmi?", allow_interruptions=True)
    assert monitor._handler.agent_speaking_calls == [True, False]


@pytest.mark.asyncio
async def test_silence_monitor_goodbye_invokes_callback():
    class _Handler:
        def check_silence_timeout(self):
            return SILENCE_GOODBYE

    session, _handle = _make_session_with_say()
    on_goodbye = AsyncMock()

    monitor = SilenceMonitor(
        error_handler=_Handler(),
        session=session,
        check_interval=0.0,
        goodbye_message="Xayr!",
        on_goodbye=on_goodbye,
    )

    await monitor._monitor_loop()

    session.say.assert_called_once_with("Xayr!", allow_interruptions=True)
    on_goodbye.assert_awaited_once()


@pytest.mark.asyncio
async def test_silence_monitor_skips_prompt_if_idle_window_closed():
    class _Handler:
        def __init__(self):
            self._called = 0

        def check_silence_timeout(self):
            self._called += 1
            if self._called == 1:
                monitor._stop.set()
                return "Eshitiapsizmi?"
            return None

        def is_silence_window_open(self):
            return False

    session, _handle = _make_session_with_say()
    monitor = SilenceMonitor(
        error_handler=_Handler(),
        session=session,
        check_interval=0.0,
        goodbye_message="x",
    )

    await monitor._monitor_loop()
    session.say.assert_not_called()
