"""
Yandex SpeechKit TTS (Text-to-Speech) provider.
Uses Yandex SpeechKit API v3 for speech synthesis with Uzbek language voices.
Available voices: yulduz, zamira, nigora
"""

import base64
import logging
import os
import re
import time
from typing import List, Optional

import aiohttp
import numpy as np
from livekit import rtc
from livekit.agents import tts

logger = logging.getLogger(__name__)


def _normalize_tts_text(text: str) -> str:
    """Normalize text before synthesis (collapse whitespace).

    TODO(urdu): add tenant/language-specific spoken-form expansions here. NavAI's
    Uzbek government abbreviations were removed for the generic blueprint.
    """
    if not text:
        return text
    return re.sub(r"\s+", " ", text).strip()


class YandexTTS(tts.TTS):
    """
    Yandex SpeechKit TTS v3 for Uzbek language.
    Supports API Key and IAM Token authentication.
    """

    def __init__(
        self,
        *,
        api_key: str = None,
        iam_token: str = None,
        folder_id: str = None,
        voice: str = None,
        language: str = "uz-UZ",
        sample_rate: int = 16000,
        num_channels: int = 1,
        role: str = "neutral",
        speed: float = 1.0,
        max_chunk_length: int = 200,
        enable_streaming: bool = False,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=enable_streaming),
            sample_rate=sample_rate,
            num_channels=num_channels,
        )

        self._api_key = api_key or os.getenv("YANDEX_API_KEY")
        self._iam_token = iam_token or os.getenv("YANDEX_IAM_TOKEN")
        self._folder_id = folder_id or os.getenv("YANDEX_FOLDER_ID")

        if not self._api_key and not self._iam_token:
            raise ValueError("Either YANDEX_API_KEY or YANDEX_IAM_TOKEN must be provided")

        if not self._folder_id and not self._api_key:
            raise ValueError("YANDEX_FOLDER_ID is required when using IAM token authentication")

        self._base_url = "https://tts.api.cloud.yandex.net/tts/v3/utteranceSynthesis"
        self._voice = voice or os.getenv("YANDEX_VOICE_ID", "yulduz")
        self._language = language
        self._role = role
        self._speed = speed
        self._sample_rate = sample_rate
        self._num_channels = num_channels
        self._max_chunk_length = max_chunk_length
        self._session: Optional[aiohttp.ClientSession] = None

        logger.info(
            f"Yandex TTS v3 initialized: voice={self._voice}, "
            f"lang={self._language}, speed={self._speed}"
        )

    def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(
                limit=10,
                limit_per_host=5,
                ssl=True,
                force_close=False,
                enable_cleanup_closed=True,
                ttl_dns_cache=300,
            )
            timeout = aiohttp.ClientTimeout(total=10, connect=3, sock_read=7)
            headers = {"Content-Type": "application/json"}
            if self._api_key:
                headers["Authorization"] = f"Api-Key {self._api_key}"
            elif self._iam_token:
                headers["Authorization"] = f"Bearer {self._iam_token}"
                if self._folder_id:
                    headers["x-folder-id"] = self._folder_id

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

    def _split_text_into_chunks(self, text: str) -> List[str]:
        if len(text) <= self._max_chunk_length:
            return [text]

        chunks = []
        sentences = re.split(r"(?<=[.!?])\s+", text)
        current_chunk = ""

        for sentence in sentences:
            if len(sentence) > self._max_chunk_length:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                    current_chunk = ""

                phrases = re.split(r"(?<=[,;])\s+", sentence)
                for phrase in phrases:
                    if len(phrase) > self._max_chunk_length:
                        words = phrase.split()
                        temp = ""
                        for word in words:
                            if len(temp + " " + word) <= self._max_chunk_length:
                                temp += (" " if temp else "") + word
                            else:
                                if temp:
                                    chunks.append(temp.strip())
                                temp = word
                        if temp:
                            chunks.append(temp.strip())
                    else:
                        if len(current_chunk + " " + phrase) <= self._max_chunk_length:
                            current_chunk += (" " if current_chunk else "") + phrase
                        else:
                            if current_chunk.strip():
                                chunks.append(current_chunk.strip())
                            current_chunk = phrase
            else:
                if len(current_chunk + " " + sentence) <= self._max_chunk_length:
                    current_chunk += (" " if current_chunk else "") + sentence
                else:
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return [c for c in chunks if c.strip()]

    def synthesize(
        self,
        text: str,
        *,
        voice: Optional[str] = None,
        language: Optional[str] = None,
        conn_options: Optional[dict] = None,
    ) -> "YandexTTSStream":
        normalized_text = _normalize_tts_text(text)
        return YandexTTSStream(
            tts_instance=self,
            text=normalized_text,
            voice=voice or self._voice,
            language=language or self._language,
        )

    async def _fetch_audio_data(self, text: str, voice: str, language: str) -> bytes:
        if not text.strip():
            return b""

        try:
            session = self._ensure_session()

            hints = [{"voice": voice}]
            if self._role:
                hints.append({"role": self._role})
            if self._speed != 1.0:
                hints.append({"speed": self._speed})

            request_body = {
                "text": text.strip(),
                "hints": hints,
                "output_audio_spec": {
                    "raw_audio": {
                        "audio_encoding": "LINEAR16_PCM",
                        "sample_rate_hertz": self._sample_rate,
                    }
                },
            }

            request_start = time.perf_counter()

            async with session.post(self._base_url, json=request_body, ssl=True) as response:
                request_ms = (time.perf_counter() - request_start) * 1000
                logger.debug(f"Yandex TTS network: {request_ms:.0f}ms")

                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Yandex TTS API error {response.status}: {error_text}")
                    return b""

                response_json = await response.json()
                audio_data = b""
                if "result" in response_json and "audioChunk" in response_json["result"]:
                    audio_chunk = response_json["result"]["audioChunk"]
                    if "data" in audio_chunk:
                        audio_data = base64.b64decode(audio_chunk["data"])

                if len(audio_data) == 0:
                    logger.error("Empty audio from Yandex TTS")
                return audio_data

        except Exception as e:
            logger.error(f"Error in Yandex TTS API: {e}")
            return b""

    def _pcm_to_audio_frame(self, pcm_data: bytes) -> rtc.AudioFrame:
        try:
            if len(pcm_data) == 0:
                return self._create_empty_frame()

            audio_array = np.frombuffer(pcm_data, dtype=np.int16).copy()
            samples_per_channel = len(audio_array)

            return rtc.AudioFrame(
                data=audio_array.tobytes(),
                sample_rate=self._sample_rate,
                num_channels=self._num_channels,
                samples_per_channel=samples_per_channel,
            )
        except Exception as e:
            logger.error(f"Error converting PCM: {e}")
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

    def _concatenate_audio_frames(self, frames: List[rtc.AudioFrame]) -> rtc.AudioFrame:
        if not frames:
            return self._create_empty_frame()
        if len(frames) == 1:
            return frames[0]

        try:
            total_samples = sum(
                np.frombuffer(frame.data, dtype=np.int16).shape[0] for frame in frames
            )
            combined = np.empty(total_samples, dtype=np.int16)
            offset = 0
            for frame in frames:
                arr = np.frombuffer(frame.data, dtype=np.int16)
                combined[offset : offset + len(arr)] = arr
                offset += len(arr)

            return rtc.AudioFrame(
                data=combined.tobytes(),
                sample_rate=self._sample_rate,
                num_channels=self._num_channels,
                samples_per_channel=len(combined),
            )
        except Exception as e:
            logger.error(f"Error concatenating frames: {e}")
            return self._create_empty_frame()


class YandexTTSStream:
    """TTS stream that yields SynthesizedAudio for each chunk."""

    def __init__(self, *, tts_instance: YandexTTS, text: str, voice: str, language: str) -> None:
        self._tts = tts_instance
        self._text = text
        self._voice = voice
        self._language = language
        self._chunks = tts_instance._split_text_into_chunks(text)
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
            logger.info(
                f"[Yandex TTS {self._index + 1}/{len(self._chunks)}] "
                f"voice={self._voice} lang={self._language} chars={len(chunk)} "
                f"'{chunk[:80]}'"
            )

            pcm_data = await self._tts._fetch_audio_data(chunk, self._voice, self._language)
            audio_frame = self._tts._pcm_to_audio_frame(pcm_data)

            result = tts.SynthesizedAudio(
                frame=audio_frame,
                request_id=f"yandex_tts_{hash(self._text)}",
                segment_id=f"chunk_{self._index}",
            )

            self._index += 1
            if self._index >= len(self._chunks):
                self._finished = True
                total_ms = (time.perf_counter() - self._start) * 1000
                logger.info(f"[Yandex TTS TOTAL] {total_ms:.0f}ms")

            return result

        except Exception as e:
            logger.error(f"Error in Yandex TTS chunk {self._index}: {e}")
            self._index += 1
            return tts.SynthesizedAudio(
                frame=self._tts._create_empty_frame(),
                request_id=f"yandex_tts_error_{hash(self._text)}",
                segment_id=f"chunk_{self._index - 1}_error",
            )
