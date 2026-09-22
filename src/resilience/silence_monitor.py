"""
Silence monitor: background task that watches for user silence.
Triggers prompts and eventual call termination.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from resilience.error_recovery import ErrorRecoveryHandler

logger = logging.getLogger(__name__)

SILENCE_GOODBYE = "GOODBYE"


class SilenceMonitor:
    """Background task monitoring user silence with config-driven timeouts."""

    def __init__(
        self,
        error_handler: ErrorRecoveryHandler,
        session: Any,
        check_interval: float = 2.0,
        goodbye_message: str = "Aloqa uzildi shekilli. Xayr!",
        on_goodbye: Any | None = None,
    ):
        self._handler = error_handler
        self._session = session
        self._check_interval = check_interval
        self._goodbye_message = goodbye_message
        self._on_goodbye = on_goodbye
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        # Tracks whether the session was recently active (user speaking,
        # agent processing/speaking, tool work, etc.). We re-arm silence
        # timing only after it transitions back to inactive.
        self._session_was_active = False

    def start(self) -> None:
        """Start the silence monitoring background task."""
        if self._task is None or self._task.done():
            self._stop.clear()
            self._task = asyncio.create_task(self._monitor_loop())
            logger.debug("Silence monitor started")

    def stop(self) -> None:
        """Stop the silence monitor."""
        self._stop.set()
        if self._task and not self._task.done():
            self._task.cancel()
        logger.debug("Silence monitor stopped")

    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while not self._stop.is_set():
            try:
                await asyncio.sleep(self._check_interval)
            except asyncio.CancelledError:
                break

            if self._stop.is_set():
                break

            # Guard: only evaluate silence when the full session is inactive.
            # This prevents prompts from firing mid-turn due to stale baseline
            # timestamps while STT/LLM/TTS work is still in flight.
            session_active = False
            try:
                if self._session and hasattr(self._session, "wait_for_inactive"):
                    await asyncio.wait_for(self._session.wait_for_inactive(), timeout=0.05)
                else:
                    session_active = False
            except asyncio.TimeoutError:
                session_active = True
            except Exception:
                # If session activity probing fails, preserve prior behavior.
                session_active = False

            if session_active:
                self._session_was_active = True
                continue

            if self._session_was_active:
                # Session just became inactive; start a fresh silence window
                # from "response finished" rather than from an older user-time.
                if self._handler and hasattr(self._handler, "reset_silence_timer"):
                    self._handler.reset_silence_timer()
                self._session_was_active = False

            response = self._handler.check_silence_timeout()

            if response == SILENCE_GOODBYE:
                logger.info("Silence goodbye timeout reached")
                try:
                    if self._session:
                        if (
                            self._handler
                            and hasattr(self._handler, "is_silence_window_open")
                            and not self._handler.is_silence_window_open()
                        ):
                            continue
                        if self._handler and hasattr(self._handler, "set_agent_speaking"):
                            self._handler.set_agent_speaking(True)
                        handle = self._session.say(
                            self._goodbye_message,
                            allow_interruptions=True,
                        )
                        # Keep "agent speaking" state until playout fully ends.
                        if hasattr(handle, "wait_for_playout"):
                            await handle.wait_for_playout()
                        else:
                            await asyncio.sleep(3.0)
                except Exception as e:
                    logger.debug(f"Goodbye speech error: {e}")
                finally:
                    if self._handler and hasattr(self._handler, "set_agent_speaking"):
                        self._handler.set_agent_speaking(False)
                try:
                    if self._on_goodbye is not None:
                        await self._on_goodbye()
                except Exception as e:
                    logger.debug(f"Goodbye callback error: {e}")
                break

            elif response:
                logger.info(f"Silence prompt: {response}")
                try:
                    if self._session:
                        # Re-check right before speaking to avoid race conditions
                        # where the user starts talking between monitor ticks.
                        if (
                            self._handler
                            and hasattr(self._handler, "is_silence_window_open")
                            and not self._handler.is_silence_window_open()
                        ):
                            continue
                        if self._handler and hasattr(self._handler, "set_agent_speaking"):
                            self._handler.set_agent_speaking(True)
                        handle = self._session.say(
                            response,
                            allow_interruptions=True,
                        )
                        # Keep "agent speaking" true until prompt playout is done.
                        if hasattr(handle, "wait_for_playout"):
                            await handle.wait_for_playout()
                except Exception as e:
                    logger.debug(f"Silence prompt error: {e}")
                finally:
                    if self._handler and hasattr(self._handler, "set_agent_speaking"):
                        self._handler.set_agent_speaking(False)
