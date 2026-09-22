"""
Gemini Live STT provider used for Paynet's first language-choice turn.

This adapter uses google-genai Live API patterns similar to ai_studio_code.py:
- sends raw audio PCM to a live session
- requests TEXT response only
- returns transcript as a LiveKit SpeechEvent

It can optionally fall back to another STT provider if Gemini Live fails.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Optional, Union

import numpy as np
from google import genai
from google.genai import types
from livekit import rtc
from livekit.agents import stt

logger = logging.getLogger(__name__)


class GeminiLiveSTT(stt.STT):
    """Direct Gemini Live transcription adapter."""

    def __init__(
        self,
        *,
        model: str | None = None,
        language_codes: list[str] | None = None,
        sample_rate: int = 16000,
        fallback_stt: stt.STT | None = None,
        one_shot: bool = True,
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=False,
                interim_results=False,
            )
        )
        self._model = model or os.getenv(
            "GEMINI_LIVE_STT_MODEL", "models/gemini-3.1-flash-live-preview"
        )
        # Optional comma-separated backups, e.g.
        # GEMINI_LIVE_STT_BACKUP_MODELS=models/gemini-3.1-flash-live-preview,models/gemini-live-2.5-flash
        raw_backups = os.getenv("GEMINI_LIVE_STT_BACKUP_MODELS", "")
        self._backup_models = [m.strip() for m in raw_backups.split(",") if m.strip()]
        self._retry_attempts = max(1, int(os.getenv("GEMINI_LIVE_STT_RETRY_ATTEMPTS", "2")))
        self._retry_backoff_ms = max(0, int(os.getenv("GEMINI_LIVE_STT_RETRY_BACKOFF_MS", "120")))
        self._sample_rate = sample_rate
        self._fallback = fallback_stt
        # Greeting override should apply only to the first post-greeting user turn.
        self._one_shot = one_shot
        self._gemini_used = False
        self._language_codes = language_codes or ["ru", "uz"]
        from utils.vertex_region import PROJECT, resolve_location

        self._client = genai.Client(
            vertexai=True,
            project=PROJECT,
            location=resolve_location(self._model.removeprefix("models/")),
            http_options={"api_version": "v1beta"},
        )
        self._first_audio_signaled = False
        self._on_first_audio = None
        self._on_stt_duration = None

        allowed = ", ".join(sorted(set(self._language_codes)))
        self._config = types.LiveConnectConfig(
            response_modalities=["TEXT"],
            system_instruction=(
                "You are a speech transcription engine. "
                "Return only the recognized user speech text, with no commentary, "
                "no translation, and no extra punctuation beyond what is spoken. "
                f"Expected languages are: {allowed}. "
                "If speech is short or mixed, still return best-effort verbatim text."
            ),
        )

    async def close(self) -> None:
        fallback = self._fallback
        if fallback and hasattr(fallback, "close"):
            await fallback.close()

    async def _recognize_impl(
        self,
        buffer: Union[rtc.AudioFrame, bytes],
        *,
        language: Optional[str] = None,
        conn_options: Optional[dict] = None,
    ) -> stt.SpeechEvent:
        start = time.perf_counter()
        if self._one_shot and self._gemini_used:
            return await self._fallback_result(buffer, language)

        if self._one_shot:
            self._gemini_used = True

        if not self._first_audio_signaled:
            self._first_audio_signaled = True
            cb = getattr(self, "_on_first_audio", None)
            if cb:
                try:
                    cb()
                except Exception:
                    pass

        try:
            pcm = self._to_pcm_16k_mono(buffer)
            if not pcm:
                return self._empty_result(language or "")
            transcript = await self._transcribe_pcm_with_retry(pcm)
            total_ms = (time.perf_counter() - start) * 1000
            cb = getattr(self, "_on_stt_duration", None)
            if cb:
                try:
                    cb(total_ms)
                except Exception:
                    pass
            if transcript:
                return stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    alternatives=[
                        stt.SpeechData(
                            text=transcript,
                            confidence=1.0,
                            language=language or "",
                        )
                    ],
                )
            logger.warning("[greeting_stt_override] Gemini returned empty transcript; falling back")
            return await self._fallback_result(buffer, language)
        except Exception as e:
            logger.warning(
                "[greeting_stt_override] Gemini transcription failed; fallback activated: %s",
                e,
            )
            return await self._fallback_result(buffer, language)

    async def _transcribe_pcm(self, pcm_bytes: bytes, *, model: str) -> str:
        async with self._client.aio.live.connect(model=model, config=self._config) as session:
            await session.send(
                input={"data": pcm_bytes, "mime_type": "audio/pcm"},
                end_of_turn=True,
            )
            text_parts: list[str] = []
            turn = session.receive()
            async for response in turn:
                txt = getattr(response, "text", None)
                if txt:
                    text_parts.append(txt.strip())
            return " ".join(part for part in text_parts if part).strip()

    async def _transcribe_pcm_with_retry(self, pcm_bytes: bytes) -> str:
        models = [self._model] + [m for m in self._backup_models if m != self._model]
        last_error: Exception | None = None
        for model in models:
            for attempt in range(1, self._retry_attempts + 1):
                try:
                    transcript = await self._transcribe_pcm(pcm_bytes, model=model)
                    if transcript:
                        if attempt > 1 or model != self._model:
                            logger.info(
                                "[greeting_stt_override] Gemini live recovered: model=%s attempt=%s",
                                model,
                                attempt,
                            )
                        return transcript
                except Exception as e:
                    last_error = e
                    logger.warning(
                        "[greeting_stt_override] Gemini live attempt failed: model=%s attempt=%s/%s error=%s",
                        model,
                        attempt,
                        self._retry_attempts,
                        e,
                    )
                    if attempt < self._retry_attempts and self._retry_backoff_ms > 0:
                        await asyncio.sleep(self._retry_backoff_ms / 1000.0)
        if last_error is not None:
            raise last_error
        return ""

    async def _fallback_result(
        self,
        buffer: Union[rtc.AudioFrame, bytes],
        language: Optional[str] = None,
    ) -> stt.SpeechEvent:
        if self._fallback is None:
            return self._empty_result(language or "")
        try:
            fn = getattr(self._fallback, "_recognize_impl", None)
            if callable(fn):
                return await fn(buffer, language=language, conn_options=None)
        except Exception as e:
            logger.warning("[greeting_stt_override] fallback STT failed: %s", e)
        return self._empty_result(language or "")

    def _to_pcm_16k_mono(self, buffer: Union[rtc.AudioFrame, bytes]) -> bytes:
        if isinstance(buffer, bytes):
            return buffer
        if not isinstance(buffer, rtc.AudioFrame):
            return b""

        audio_data = np.frombuffer(buffer.data, dtype=np.int16)
        if buffer.num_channels > 1:
            audio_data = audio_data.reshape(-1, buffer.num_channels)
            audio_data = audio_data.mean(axis=1).astype(np.int16)

        if buffer.sample_rate != self._sample_rate:
            ratio = self._sample_rate / buffer.sample_rate
            new_length = int(len(audio_data) * ratio)
            if new_length <= 0:
                return b""
            indices = np.linspace(0, len(audio_data) - 1, new_length)
            audio_data = np.interp(indices, np.arange(len(audio_data)), audio_data).astype(np.int16)
        return audio_data.tobytes()

    @staticmethod
    def _empty_result(language: str) -> stt.SpeechEvent:
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(text="", confidence=0.0, language=language)],
        )
