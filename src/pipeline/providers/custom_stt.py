"""
Custom STT (Speech-to-Text) provider.
Uses local GPU server for Uzbek speech recognition.
Endpoint: POST /transcribe/transcribe-batch
"""

from __future__ import annotations

import io
import logging
import os
import time
import wave
from typing import Optional, Union

import aiohttp
import numpy as np
from livekit import rtc
from livekit.agents import stt

from observability.network_topology import network_request_context, register_service_route

logger = logging.getLogger(__name__)


class CustomSTT(stt.STT):
    """Custom STT using local GPU server for Uzbek language."""

    def __init__(
        self,
        *,
        base_url: str = None,
        language: str = "uz",
        sample_rate: int = 16000,
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=False,
                interim_results=False,
            )
        )

        self._base_url = (base_url or os.getenv("CUSTOM_STT_URL", "http://localhost:8080")).rstrip(
            "/"
        )
        self._endpoint = f"{self._base_url}/transcribe/transcribe-batch"
        self._language = language
        self._sample_rate = sample_rate
        self._session: Optional[aiohttp.ClientSession] = None

        self._first_audio_signaled = False
        self._on_first_audio = None  # callback set by main.py for telephony timing
        self._on_stt_duration = None  # callback to report STT duration

        register_service_route(
            "custom_stt",
            self._base_url,
            provider="custom_stt",
            metadata={"language": self._language},
        )
        logger.info(f"Custom STT initialized: url={self._base_url}, lang={self._language}")

    def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            # connect=1.0 keeps the per-request penalty bounded when the upstream
            # GPU host is TCP-unreachable (was 3.0s, observed as a flat 3001ms
            # timeout per turn in production when the host is down).
            timeout = aiohttp.ClientTimeout(total=15, connect=1.0, sock_read=12)
            connector = aiohttp.TCPConnector(
                limit=10,
                limit_per_host=5,
                force_close=False,
                enable_cleanup_closed=True,
            )
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
            )
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def _recognize_impl(
        self,
        buffer: Union[rtc.AudioFrame, bytes],
        *,
        language: Optional[str] = None,
        conn_options: Optional[dict] = None,
    ) -> stt.SpeechEvent:
        overall_start = time.perf_counter()

        # Signal first audio arrival to telephony tracker
        if not self._first_audio_signaled:
            self._first_audio_signaled = True
            cb = getattr(self, "_on_first_audio", None)
            if cb:
                try:
                    cb()
                except Exception:
                    pass

        try:
            if isinstance(buffer, rtc.AudioFrame):
                audio_duration = buffer.samples_per_channel / buffer.sample_rate
                if audio_duration < 0.1:
                    return self._empty_result(language or self._language)

            wav_data = self._prepare_wav(buffer)
            if len(wav_data) == 0:
                return self._empty_result(language or self._language)

            transcript = await self._transcribe(wav_data, language)
            total_ms = (time.perf_counter() - overall_start) * 1000

            # Report STT duration to telephony tracker
            cb = getattr(self, "_on_stt_duration", None)
            if cb:
                try:
                    cb(total_ms)
                except Exception:
                    pass

            if isinstance(buffer, rtc.AudioFrame):
                audio_duration = buffer.samples_per_channel / buffer.sample_rate
            else:
                audio_duration = len(wav_data) / (self._sample_rate * 2)

            if transcript and transcript.strip():
                logger.info(
                    f"Custom STT [{audio_duration:.2f}s -> {total_ms:.0f}ms]: "
                    f"'{transcript[:50]}'"
                )
                return stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    alternatives=[
                        stt.SpeechData(
                            text=transcript.strip(),
                            confidence=1.0,
                            language=language or self._language,
                        )
                    ],
                )
            else:
                return self._empty_result(language or self._language)

        except Exception as e:
            logger.error(f"Custom STT error: {e}")
            return self._empty_result(language or self._language)

    async def _transcribe(self, wav_data: bytes, language: Optional[str] = None) -> str:
        """Send audio to custom STT API.

        The server auto-detects the language from the audio; the legacy
        `languages` form field is intentionally omitted (it was redundant
        and rejected anything other than short ISO codes like 'uz').
        """
        session = self._ensure_session()
        lang = language or self._language

        try:
            form = aiohttp.FormData()
            form.add_field(
                "audio_files",
                wav_data,
                filename="audio.wav",
                content_type="audio/wav",
            )

            with network_request_context(
                "custom_stt",
                "transcribe",
                metadata={
                    "language": lang,
                    "audio_bytes": len(wav_data),
                },
            ):
                async with session.post(self._endpoint, data=form) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Custom STT API error {response.status}: {error_text[:200]}")
                        return ""

                    result = await response.json()

                # Batch response: [{"transcription": "...", "language": "uz", "segments": [...]}]
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get("transcription", "").strip()
                elif isinstance(result, dict):
                    return result.get("transcription", result.get("text", "")).strip()

                return ""

        except aiohttp.ClientError as e:
            logger.error(f"Custom STT network error: {e}")
            return ""
        except Exception as e:
            logger.error(f"Custom STT error: {e}")
            return ""

    def _prepare_wav(self, buffer: Union[rtc.AudioFrame, bytes]) -> bytes:
        """Convert AudioFrame to WAV bytes."""
        try:
            if isinstance(buffer, rtc.AudioFrame):
                audio_data = np.frombuffer(buffer.data, dtype=np.int16)

                # Mono conversion
                if buffer.num_channels > 1:
                    audio_data = audio_data.reshape(-1, buffer.num_channels)
                    audio_data = audio_data.mean(axis=1).astype(np.int16)

                # Resample to target rate
                if buffer.sample_rate != self._sample_rate:
                    ratio = self._sample_rate / buffer.sample_rate
                    new_length = int(len(audio_data) * ratio)
                    if new_length > 0 and ratio != 1.0:
                        indices = np.linspace(0, len(audio_data) - 1, new_length)
                        audio_data = np.interp(
                            indices, np.arange(len(audio_data)), audio_data
                        ).astype(np.int16)

                wav_buffer = io.BytesIO()
                with wave.open(wav_buffer, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(self._sample_rate)
                    wf.writeframes(audio_data.tobytes())
                return wav_buffer.getvalue()

            return buffer if isinstance(buffer, bytes) else b""
        except Exception as e:
            logger.error(f"Custom STT audio prep error: {e}")
            return b""

    def _empty_result(self, language: str) -> stt.SpeechEvent:
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(text="", confidence=0.0, language=language)],
        )
