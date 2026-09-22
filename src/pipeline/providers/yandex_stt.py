"""
Yandex SpeechKit STT (Speech-to-Text) provider.
Uses Yandex SpeechKit API v1 for speech recognition with Uzbek language support.
"""

import asyncio
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

logger = logging.getLogger(__name__)


class YandexSTT(stt.STT):
    """
    Yandex SpeechKit STT v1 for Uzbek language.
    Supports API Key and IAM Token authentication.
    """

    def __init__(
        self,
        *,
        api_key: str = None,
        iam_token: str = None,
        folder_id: str = None,
        language: str = "uz-UZ",
        sample_rate: int = 16000,
        model: str = "general",
        profanity_filter: bool = False,
        partial_results: bool = False,
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=False,
                interim_results=partial_results,
            )
        )

        self._api_key = api_key or os.getenv("YANDEX_API_KEY")
        self._iam_token = iam_token or os.getenv("YANDEX_IAM_TOKEN")
        self._folder_id = folder_id or os.getenv("YANDEX_FOLDER_ID")

        if not self._api_key and not self._iam_token:
            raise ValueError("Either YANDEX_API_KEY or YANDEX_IAM_TOKEN must be provided")

        if not self._folder_id and not self._api_key:
            raise ValueError("YANDEX_FOLDER_ID is required when using IAM token authentication")

        self._base_url = "https://stt.api.cloud.yandex.net/speech/v1/stt:recognize"
        self._language = language
        self._sample_rate = sample_rate
        self._model = model
        self._profanity_filter = profanity_filter
        self._partial_results = partial_results
        self._wav_buffer = io.BytesIO()
        self._session: Optional[aiohttp.ClientSession] = None

        self._first_audio_signaled = False
        self._on_first_audio = None  # callback for telephony timing
        self._on_stt_duration = None  # callback to report STT duration

        logger.info(f"Yandex STT initialized: lang={self._language}, model={self._model}")

    def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(
                limit=20,
                limit_per_host=10,
                ttl_dns_cache=300,
                ssl=True,
                force_close=False,
                enable_cleanup_closed=True,
                keepalive_timeout=30,
            )
            timeout = aiohttp.ClientTimeout(total=10, connect=2, sock_read=8)
            headers = {"Accept": "application/json"}
            if self._api_key:
                headers["Authorization"] = f"Api-Key {self._api_key}"
            elif self._iam_token:
                headers["Authorization"] = f"Bearer {self._iam_token}"

            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers=headers,
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

            audio_data = self._prepare_audio_file(buffer)
            if len(audio_data) == 0:
                return self._empty_result(language or self._language)

            if isinstance(buffer, rtc.AudioFrame):
                audio_duration = buffer.samples_per_channel / buffer.sample_rate
            else:
                audio_duration = len(audio_data) / (self._sample_rate * 2)

            transcript = await self._transcribe_audio(audio_data, language)
            total_ms = (time.perf_counter() - overall_start) * 1000

            cb = getattr(self, "_on_stt_duration", None)
            if cb:
                try:
                    cb(total_ms)
                except Exception:
                    pass

            if transcript and transcript.strip():
                logger.info(
                    f"Yandex STT [{audio_duration:.2f}s -> {total_ms:.0f}ms]: '{transcript[:50]}'"
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
            logger.error(f"Error in Yandex STT: {e}")
            return self._empty_result(language or self._language)

    async def _transcribe_audio(self, audio_data: bytes, language: Optional[str] = None) -> str:
        session = self._ensure_session()
        lang = language or self._language

        try:
            params = {
                "lang": lang,
                "model": self._model,
                "sampleRateHertz": self._sample_rate,
                "profanityFilter": "true" if self._profanity_filter else "false",
                "format": "lpcm",
            }
            if self._folder_id and not self._api_key:
                params["folderId"] = self._folder_id

            pcm_data = self._extract_pcm_from_wav(audio_data)

            async with session.post(
                self._base_url,
                params=params,
                data=pcm_data,
                headers={"Content-Type": "audio/x-pcm;bit=16;rate=16000"},
                ssl=True,
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Yandex STT API error {response.status}: {error_text[:200]}")
                    return ""
                response_data = await response.json()
                return response_data.get("result", "").strip()

        except asyncio.TimeoutError:
            logger.error("Yandex STT timeout")
            return ""
        except aiohttp.ClientError as e:
            logger.error(f"Yandex STT network error: {e}")
            return ""
        except Exception as e:
            logger.error(f"Error transcribing with Yandex STT: {e}")
            return ""

    def _prepare_audio_file(self, buffer: Union[rtc.AudioFrame, bytes]) -> bytes:
        try:
            if isinstance(buffer, rtc.AudioFrame):
                audio_data = np.frombuffer(buffer.data, dtype=np.int16)

                if buffer.num_channels > 1:
                    audio_data = audio_data.reshape(-1, buffer.num_channels)
                    audio_data = audio_data.mean(axis=1).astype(np.int16)

                if buffer.sample_rate != self._sample_rate:
                    ratio = self._sample_rate / buffer.sample_rate
                    new_length = int(len(audio_data) * ratio)
                    if new_length > 0 and ratio != 1.0:
                        indices = np.linspace(0, len(audio_data) - 1, new_length)
                        audio_data = np.interp(
                            indices, np.arange(len(audio_data)), audio_data
                        ).astype(np.int16)

                self._wav_buffer.seek(0)
                self._wav_buffer.truncate()
                with wave.open(self._wav_buffer, "wb") as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(self._sample_rate)
                    wav_file.writeframes(audio_data.tobytes())
                return self._wav_buffer.getvalue()

            return buffer if isinstance(buffer, bytes) else b""
        except Exception as e:
            logger.error(f"Error preparing audio: {e}")
            return b""

    def _extract_pcm_from_wav(self, wav_data: bytes) -> bytes:
        try:
            with wave.open(io.BytesIO(wav_data), "rb") as wav_file:
                return wav_file.readframes(wav_file.getnframes())
        except Exception as e:
            logger.error(f"Error extracting PCM from WAV: {e}")
            return b""

    def _empty_result(self, language: str) -> stt.SpeechEvent:
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(text="", confidence=0.0, language=language)],
        )
