"""
Latency tracker: per-turn latency measurement.
"""

from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)


class LatencyTracker:
    """Tracks per-turn latency: user_stopped → STT → LLM → TTS."""

    def __init__(self, tenant: str = ""):
        self.tenant = tenant
        self._turn_start: float | None = None
        self._stt_end: float | None = None
        self._llm_end: float | None = None
        self._tts_end: float | None = None

    def start_turn(self) -> None:
        """Mark the start of a new turn (user stopped speaking)."""
        self._turn_start = time.time()
        self._stt_end = None
        self._llm_end = None
        self._tts_end = None

    def mark_stt_complete(self) -> None:
        self._stt_end = time.time()

    def mark_llm_complete(self) -> None:
        self._llm_end = time.time()

    def mark_tts_complete(self) -> None:
        self._tts_end = time.time()

    def get_metrics(self) -> dict[str, float]:
        """Get latency metrics for this turn."""
        if not self._turn_start:
            return {}

        metrics = {}
        if self._stt_end:
            metrics["stt_ms"] = (self._stt_end - self._turn_start) * 1000
        if self._llm_end and self._stt_end:
            metrics["llm_ms"] = (self._llm_end - self._stt_end) * 1000
        if self._tts_end and self._llm_end:
            metrics["tts_ms"] = (self._tts_end - self._llm_end) * 1000
        if self._tts_end:
            metrics["total_ms"] = (self._tts_end - self._turn_start) * 1000

        return metrics
