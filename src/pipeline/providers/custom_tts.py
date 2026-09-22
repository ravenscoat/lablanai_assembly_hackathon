"""
Custom TTS (Text-to-Speech) provider.
Uses local GPU server for Uzbek speech synthesis.
Endpoint: POST /synthesize/local
"""

from __future__ import annotations

import io
import logging
import os
import re
import time
import wave
from typing import List, Optional

import aiohttp
import numpy as np
from livekit import rtc
from livekit.agents import tts

from observability.network_topology import network_request_context, register_service_route

logger = logging.getLogger(__name__)


def _normalize_tts_text(text: str) -> str:
    """Normalize text before synthesis (collapse whitespace).

    TODO(urdu): add tenant/language-specific spoken-form expansions here
    (e.g. expand domain names, acronyms, "%" → the spoken word). The original
    NavAI implementation hard-coded Uzbek government abbreviations; those were
    removed for the generic blueprint. Keep such rules out of this shared
    provider — prefer per-tenant/per-language config.
    """
    if not text:
        return text
    return re.sub(r"\s+", " ", text).strip()


class CustomTTS(tts.TTS):
    """Custom TTS using local GPU server for Uzbek language."""

    def __init__(
        self,
        *,
        base_url: str = None,
        voice: str = "default",
        language: str = "uz",
        sample_rate: int = 16000,
        num_channels: int = 1,
        speed: float = 1.0,
        output_format: str = "wav",
        max_chunk_length: int = 250,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=sample_rate,
            num_channels=num_channels,
        )

        self._base_url = (base_url or os.getenv("CUSTOM_TTS_URL", "http://localhost:8002")).rstrip(
            "/"
        )
        self._endpoint = f"{self._base_url}/synthesize/local"
        self._voice = voice or os.getenv("CUSTOM_VOICE_ID", "default")
        self._language = language
        self._sample_rate = sample_rate
        self._num_channels = num_channels
        self._speed = speed
        self._output_format = output_format
        self._max_chunk_length = max_chunk_length
        self._session: Optional[aiohttp.ClientSession] = None

        self._on_tts_duration = None  # callback to report TTS duration

        register_service_route(
            "custom_tts",
            self._base_url,
            provider="custom_tts",
            metadata={"voice": self._voice, "speed": self._speed},
        )
        logger.info(
            f"Custom TTS initialized: url={self._base_url}, "
            f"voice={self._voice}, speed={self._speed}"
        )

    def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=15, connect=3, sock_read=12)
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

    def _split_text(self, text: str) -> List[str]:
        """Split text into chunks for processing."""
        if len(text) <= self._max_chunk_length:
            return [text]

        chunks = []
        sentences = re.split(r"(?<=[.!?])\s+", text)
        current = ""

        for sentence in sentences:
            if len(current + " " + sentence) <= self._max_chunk_length:
                current += (" " if current else "") + sentence
            else:
                if current.strip():
                    chunks.append(current.strip())
                current = sentence

        if current.strip():
            chunks.append(current.strip())

        return [c for c in chunks if c.strip()]

    def synthesize(
        self,
        text: str,
        *,
        voice: Optional[str] = None,
        language: Optional[str] = None,
        conn_options: Optional[dict] = None,
    ) -> "CustomTTSStream":
        normalized = _normalize_tts_text(text)
        return CustomTTSStream(
            tts_instance=self,
            text=normalized,
            voice=voice or self._voice,
        )

    async def _fetch_audio(self, text: str, voice: str) -> bytes:
        """Fetch audio from custom TTS API."""
        if not text.strip():
            return b""

        try:
            session = self._ensure_session()
            request_start = time.perf_counter()

            form = aiohttp.FormData()
            form.add_field("target_text", text.strip())
            form.add_field("output_format", self._output_format)

            with network_request_context(
                "custom_tts",
                "synthesize",
                metadata={
                    "voice": voice,
                    "text_length": len(text.strip()),
                    "output_format": self._output_format,
                },
            ):
                async with session.post(self._endpoint, data=form) as response:
                    request_ms = (time.perf_counter() - request_start) * 1000
                    logger.debug(f"Custom TTS network: {request_ms:.0f}ms")

                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Custom TTS API error {response.status}: {error_text[:200]}")
                        return b""

                    audio_data = await response.read()
                    if len(audio_data) == 0:
                        logger.error("Empty audio from Custom TTS")
                    return audio_data

        except Exception as e:
            logger.error(f"Custom TTS API error: {e}")
            return b""

    def _wav_to_audio_frame(self, wav_data: bytes) -> rtc.AudioFrame:
        """Convert WAV bytes to AudioFrame."""
        try:
            if len(wav_data) == 0:
                return self._create_empty_frame()

            with wave.open(io.BytesIO(wav_data), "rb") as wf:
                pcm_data = wf.readframes(wf.getnframes())
                sr = wf.getframerate()
                ch = wf.getnchannels()

            audio_array = np.frombuffer(pcm_data, dtype=np.int16).copy()

            # Convert to mono if needed
            if ch > 1:
                audio_array = audio_array.reshape(-1, ch).mean(axis=1).astype(np.int16)

            # Resample if needed
            if sr != self._sample_rate:
                ratio = self._sample_rate / sr
                new_length = int(len(audio_array) * ratio)
                if new_length > 0:
                    indices = np.linspace(0, len(audio_array) - 1, new_length)
                    audio_array = np.interp(
                        indices, np.arange(len(audio_array)), audio_array
                    ).astype(np.int16)

            return rtc.AudioFrame(
                data=audio_array.tobytes(),
                sample_rate=self._sample_rate,
                num_channels=self._num_channels,
                samples_per_channel=len(audio_array),
            )

        except Exception as e:
            logger.error(f"Custom TTS WAV conversion error: {e}")
            return self._create_empty_frame()

    def _create_empty_frame(self) -> rtc.AudioFrame:
        empty_samples = int(self._sample_rate * 0.1)
        empty_data = np.zeros(empty_samples, dtype=np.int16)
        return rtc.AudioFrame(
            data=empty_data.tobytes(),
            sample_rate=self._sample_rate,
            num_channels=self._num_channels,
            samples_per_channel=empty_samples,
        )


class CustomTTSStream:
    """TTS stream that yields SynthesizedAudio for each chunk."""

    def __init__(self, *, tts_instance: CustomTTS, text: str, voice: str) -> None:
        self._tts = tts_instance
        self._text = text
        self._voice = voice
        self._chunks = tts_instance._split_text(text)
        self._index = 0
        self._finished = False
        self._start = time.perf_counter()

    def __aiter__(self):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None

    async def __anext__(self) -> tts.SynthesizedAudio:
        if self._finished or self._index >= len(self._chunks):
            raise StopAsyncIteration

        try:
            chunk = self._chunks[self._index]
            logger.info(f"[Custom TTS {self._index + 1}/{len(self._chunks)}] '{chunk[:40]}'")

            wav_data = await self._tts._fetch_audio(chunk, self._voice)
            audio_frame = self._tts._wav_to_audio_frame(wav_data)

            result = tts.SynthesizedAudio(
                frame=audio_frame,
                request_id=f"custom_tts_{hash(self._text)}",
                segment_id=f"chunk_{self._index}",
            )

            self._index += 1
            if self._index >= len(self._chunks):
                self._finished = True
                total_ms = (time.perf_counter() - self._start) * 1000
                logger.info(f"[Custom TTS TOTAL] {total_ms:.0f}ms")
                cb = getattr(self._tts, "_on_tts_duration", None)
                if cb:
                    try:
                        cb(total_ms)
                    except Exception:
                        pass

            return result

        except Exception as e:
            logger.error(f"Custom TTS chunk {self._index} error: {e}")
            self._index += 1
            return tts.SynthesizedAudio(
                frame=self._tts._create_empty_frame(),
                request_id=f"custom_tts_error_{hash(self._text)}",
                segment_id=f"chunk_{self._index - 1}_error",
            )
