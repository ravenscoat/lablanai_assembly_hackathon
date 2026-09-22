"""
Error recovery handler: config-driven silence detection.
Ported from navai-voice-agent/src/agents/youth/error_recovery.py; the
confidence/repetition/field-retry code paths were never wired in this repo
and were removed in NAV-140 along with the ErrorType enum.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from config.schema import TenantConfig

logger = logging.getLogger(__name__)


@dataclass
class ErrorState:
    """Silence-tracking state for the active conversation."""

    last_user_speech_time: float | None = None
    silence_prompt_sent: bool = False
    user_speaking: bool = False
    agent_speaking: bool = False
    agent_finished_speaking_time: float | None = None


class ErrorRecoveryHandler:
    """
    Silence-timeout handler with config-driven thresholds.
    The broader error-recovery surface (confidence tracking, repetition
    detection, field retries, API-failure counts) was never wired and was
    removed in NAV-140; reintroduce the relevant piece with tests if the
    behavior is needed again.
    """

    def __init__(self, config: TenantConfig):
        self._config = config
        self._state = ErrorState()
        self._language = config.languages.default

        # Silence thresholds from config
        self.silence_prompt_timeout = config.silence.prompt_timeout
        self.silence_goodbye_timeout = config.silence.goodbye_timeout

    def reset_silence_timer(self) -> None:
        """Reset silence timer (call when user speaks)."""
        self._state.last_user_speech_time = time.time()
        self._state.silence_prompt_sent = False

    def set_agent_speaking(self, speaking: bool) -> None:
        """Track agent speaking state to avoid false silence detection."""
        was_speaking = self._state.agent_speaking
        self._state.agent_speaking = speaking
        # Mark "agent finished speaking" only on a real speaking->not-speaking transition.
        if was_speaking and not speaking:
            finished_at = time.time()
            self._state.agent_finished_speaking_time = finished_at
            # Arm silence window when agent finishes speaking (user can respond now),
            # but don't move the baseline backwards if a newer reset already exists.
            baseline = self._state.last_user_speech_time
            if baseline is None or finished_at > baseline:
                self._state.last_user_speech_time = finished_at

    def set_user_speaking(self, speaking: bool) -> None:
        """Track user speaking state to avoid silence prompts during utterances."""
        self._state.user_speaking = speaking

    def set_language(self, language: str) -> None:
        """Update active conversation language for localized responses."""
        self._language = (language or "").strip() or self._config.languages.default

    def is_silence_window_open(self) -> bool:
        """True only when both caller and agent are idle."""
        return not self._state.agent_speaking and not self._state.user_speaking

    def check_silence_timeout(self) -> str | None:
        """
        Check if user has been silent too long.

        Returns:
            None: No timeout
            str: Prompt message or "GOODBYE" for call end
        """
        if not self.is_silence_window_open():
            return None

        # Silence checks use a resettable baseline timestamp that is updated on
        # user speech and on agent-finished-speaking transitions.
        ref_time = self._state.last_user_speech_time
        if ref_time is None:
            self._state.last_user_speech_time = time.time()
            return None

        elapsed = time.time() - ref_time

        if elapsed >= self.silence_goodbye_timeout:
            return "GOODBYE"

        if elapsed >= self.silence_prompt_timeout and not self._state.silence_prompt_sent:
            self._state.silence_prompt_sent = True
            return self._config.responses.silence_prompt_for(self._language)

        return None
