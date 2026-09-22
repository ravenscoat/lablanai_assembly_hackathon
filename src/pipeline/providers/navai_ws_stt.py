"""
NavAI WebSocket streaming STT provider.

Streams raw PCM16 (mono, little-endian, 16 kHz) audio to the team's self-hosted
streaming STT server and surfaces **incremental** transcripts — partials arrive as
the caller speaks, not only at the end. The latency win over the batch HTTP custom
STT (``/transcribe/transcribe-batch``) is incremental decoding + barge-in off the
partials.

Protocol (see the team's ``websocket-client.md``):

    connect ws://<host>:8080/transcribe/live   subprotocol "stt.v1"
    client → {"type":"start","sample_rate":16000,"num_channels":1,"language":"uz"}
             <binary PCM16 frame> <binary PCM16 frame> ...
             {"type":"end"}                              (flush → triggers `final`)
    server → {"type":"speech_start"} | {"type":"interim","text":...}
             {"type":"final","text":...} | {"type":"warn"|"error",...}
             {"type":"done"}                             (session closed)

The server emits exactly **one `final` per session**, only after ``{"type":"end"}``;
it does not auto-segment on silence. LiveKit's default ``stt_node`` for a streaming
STT pushes frames continuously and **never calls ``flush()`` per turn** — on
turn-commit it just pushes ~0.2s of silence and waits for a FINAL. So this provider
must do its own endpointing: it runs an internal Silero VAD (reusing the session's
prewarmed instance) purely as a control signal — ``START_OF_SPEECH`` opens a WS
session, ``END_OF_SPEECH`` sends ``end`` — and streams the live audio windows in
between. One WS session == one utterance, matching the server's one-shot lifecycle.

Config: ``voice.stt_provider: navai_ws``. Env: ``NAVAI_WS_STT_URL`` (base or full
``/transcribe/live`` URL; accepts scheme-less host:port and http(s)://),
``NAVAI_WS_STT_MAX_SECONDS`` (per-utterance ceiling).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Optional

import aiohttp
import numpy as np
from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    stt,
)
from livekit.agents import (
    vad as vad_module,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer, aio, is_given, merge_frames

from observability.network_topology import network_request_context, register_service_route

logger = logging.getLogger(__name__)

# We force-resample every input frame to 16 kHz (RecognizeStream(sample_rate=...))
# and declare 16 kHz to the server, which is one of its accepted rates. Keeping a
# single rate end to end avoids the consistency check in push_frame and any
# pitch/speed bugs from a rate mismatch.
SAMPLE_RATE = 16000
NUM_CHANNELS = 1
SUBPROTOCOL = "stt.v1"
# Streaming STT gateway endpoint. No hardcoded default host — set the
# NAVAI_WS_STT_URL env var (TLS + API-key auth) to your own WS STT gateway.
DEFAULT_WS_URL = ""
DEFAULT_LANGUAGE = "uz"
_TRANSCRIBE_PATH = "/api/stt/transcribe/live"
# API key for the gateway. Sent as the ``X-API-Key`` header on the WS upgrade —
# our client isn't a browser, so we skip the ``?api_key=`` query-param fallback
# (which would also leak the key into network-topology URL logs).
API_KEY_ENV = "NAVAI_API_KEY"
# Hard ceiling on a single utterance: bounds a server that streams interims forever
# without ever sending 'final'/'done' after we asked it to flush. Env-overridable.
DEFAULT_MAX_UTTERANCE_SECONDS = 60.0


def _to_ws_transcribe_url(raw: str) -> str:
    """Normalize a configured URL into a full ``ws(s)://host/api/stt/transcribe/live`` URL.

    Accepts scheme-less ``host[:port]``, ``http(s)://`` (auto-converted to
    ``ws(s)://``), and URLs that already carry a ``/transcribe/live`` path (old raw
    server or the new gateway path).
    """
    raw = (raw or "").strip().rstrip("/")
    if raw.startswith("http://"):
        raw = "ws://" + raw[len("http://") :]
    elif raw.startswith("https://"):
        raw = "wss://" + raw[len("https://") :]
    elif not raw.startswith(("ws://", "wss://")):
        raw = "ws://" + raw  # scheme-less, e.g. "stt-gateway.example.com"

    # Matches both the new gateway path and the legacy raw-server path.
    if raw.endswith(("/transcribe/live", "/transcribe/transcribe")):
        return raw
    return raw + _TRANSCRIBE_PATH


def _ws_to_http_origin(ws_url: str) -> str:
    """ws(s):// → http(s):// origin, for the network-topology service route label."""
    origin = ws_url.split("/transcribe/live")[0]
    if origin.startswith("wss://"):
        return "https://" + origin[len("wss://") :]
    if origin.startswith("ws://"):
        return "http://" + origin[len("ws://") :]
    return origin


class NavaiWSSTT(stt.STT):
    """NavAI streaming STT over a WebSocket (PCM16 mono @ 16 kHz)."""

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        language: str = DEFAULT_LANGUAGE,
        sample_rate: int = SAMPLE_RATE,
        num_channels: int = NUM_CHANNELS,
        vad: Optional[vad_module.VAD] = None,
        max_utterance_s: Optional[float] = None,
        api_key: Optional[str] = None,
    ) -> None:
        # streaming=True + interim_results=True: the framework drives us via
        # stream() (not the batch recognize path) and forwards our INTERIM events
        # for barge-in / low-latency display.
        super().__init__(capabilities=stt.STTCapabilities(streaming=True, interim_results=True))

        self._ws_url = _to_ws_transcribe_url(
            base_url or os.getenv("NAVAI_WS_STT_URL", DEFAULT_WS_URL)
        )
        self._language = language or DEFAULT_LANGUAGE
        self._sample_rate = sample_rate
        self._num_channels = num_channels
        self._vad = vad
        self._max_utterance_s = max_utterance_s or float(
            os.getenv("NAVAI_WS_STT_MAX_SECONDS", DEFAULT_MAX_UTTERANCE_SECONDS)
        )
        self._api_key = (api_key or os.getenv(API_KEY_ENV) or "").strip() or None

        # Telephony timing hooks, set by main.py (parity with the other providers).
        self._on_first_audio = None
        self._on_stt_duration = None

        register_service_route(
            "navai_ws_stt",
            _ws_to_http_origin(self._ws_url),
            provider="navai_ws_stt",
            metadata={"language": self._language, "sample_rate": self._sample_rate},
            notes="NavAI streaming STT over WebSocket (PCM16 @ 16 kHz)",
        )
        logger.info(
            "NavAI WS STT initialized: url=%s lang=%s sr=%d auth=%s",
            self._ws_url,
            self._language,
            self._sample_rate,
            "on" if self._api_key else "off",
        )

    def _ws_headers(self) -> dict:
        """Auth header for the gateway WS upgrade (empty when no key configured)."""
        return {"X-API-Key": self._api_key} if self._api_key else {}

    def _ensure_vad(self) -> vad_module.VAD:
        """Return the injected VAD, lazily loading a default Silero VAD if none was
        provided. Production always injects the session's prewarmed, tenant-tuned
        VAD; the lazy path covers tests and agent-rebuild call sites."""
        if self._vad is None:
            from livekit.plugins import silero

            logger.warning(
                "NavAI WS STT: no VAD injected; loading a default Silero VAD. "
                "Endpointing thresholds will not match the tenant's VAD config."
            )
            self._vad = silero.VAD.load()
        return self._vad

    async def aclose(self) -> None:
        await super().aclose()

    async def close(self) -> None:
        await self.aclose()

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "SpeechStream":
        return SpeechStream(
            stt=self,
            vad=self._ensure_vad(),
            language=language if is_given(language) else self._language,
            conn_options=conn_options,
        )

    # ------------------------------------------------------------------ #
    # Batch path — required by the abstract base, and used if someone calls
    # recognize() directly (e.g. a greeting-STT fallback). One-shot WS session.
    # ------------------------------------------------------------------ #
    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        lang = language if is_given(language) else self._language
        pcm = self._pcm16_16k_mono(buffer)
        if not pcm:
            return self._empty_result(lang)
        try:
            text = await self._transcribe_once(pcm, lang, conn_options)
        except Exception as e:  # graceful — never break the call on STT failure
            logger.warning("NavAI WS STT batch recognize failed: %s", e)
            return self._empty_result(lang)
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(text=text.strip(), language=lang, confidence=1.0)],
        )

    async def _transcribe_once(
        self, pcm: bytes, language: str, conn_options: APIConnectOptions
    ) -> str:
        """Open a WS session, stream `pcm`, flush, and return the final transcript
        (falling back to the last interim)."""
        last_interim = ""
        final_text = ""
        conn_timeout = aiohttp.ClientTimeout(
            total=None, connect=conn_options.timeout, sock_connect=conn_options.timeout
        )
        start_msg = json.dumps(
            {
                "type": "start",
                "sample_rate": self._sample_rate,
                "num_channels": self._num_channels,
                "language": language,
            }
        )
        with network_request_context(
            "navai_ws_stt", "transcribe", metadata={"language": language, "bytes": len(pcm)}
        ):
            async with aiohttp.ClientSession(timeout=conn_timeout) as session:
                async with session.ws_connect(
                    self._ws_url,
                    protocols=(SUBPROTOCOL,),
                    headers=self._ws_headers(),
                    timeout=aiohttp.ClientWSTimeout(ws_receive=conn_options.timeout, ws_close=5),
                    max_msg_size=0,
                    heartbeat=None,
                ) as ws:
                    await ws.send_str(start_msg)
                    # ~80 ms frames (2560 bytes @16k mono), the cadence the doc suggests.
                    step = SAMPLE_RATE // 100 * 2 * 8  # 2560 bytes
                    for i in range(0, len(pcm), step):
                        await ws.send_bytes(pcm[i : i + step])
                    await ws.send_str(json.dumps({"type": "end"}))

                    async with asyncio.timeout(self._max_utterance_s):
                        async for msg in ws:
                            if msg.type != aiohttp.WSMsgType.TEXT:
                                if msg.type in (
                                    aiohttp.WSMsgType.CLOSE,
                                    aiohttp.WSMsgType.CLOSING,
                                    aiohttp.WSMsgType.CLOSED,
                                    aiohttp.WSMsgType.ERROR,
                                ):
                                    break
                                continue
                            try:
                                ev = json.loads(msg.data)
                            except (json.JSONDecodeError, ValueError):
                                continue
                            t = ev.get("type")
                            if t == "interim":
                                last_interim = (ev.get("text") or "").strip() or last_interim
                            elif t == "final":
                                final_text = (ev.get("text") or "").strip()
                            elif t == "error":
                                raise APIConnectionError(
                                    ev.get("message") or "navai_ws_stt server error"
                                )
                            elif t == "done":
                                break
        return final_text or last_interim

    @staticmethod
    def _pcm16_16k_mono(buffer: AudioBuffer) -> bytes:
        """Coerce an AudioBuffer to PCM16 mono @ 16 kHz bytes."""
        try:
            merged = merge_frames(buffer)
            data = np.frombuffer(merged.data, dtype=np.int16)
            if merged.num_channels > 1:
                data = data.reshape(-1, merged.num_channels).mean(axis=1).astype(np.int16)
            if merged.sample_rate != SAMPLE_RATE and len(data) > 0:
                ratio = SAMPLE_RATE / merged.sample_rate
                new_len = int(len(data) * ratio)
                if new_len > 0 and ratio != 1.0:
                    idx = np.linspace(0, len(data) - 1, new_len)
                    data = np.interp(idx, np.arange(len(data)), data).astype(np.int16)
            return data.tobytes()
        except Exception as e:
            logger.error("NavAI WS STT audio prep error: %s", e)
            return b""

    @staticmethod
    def _empty_result(language: str) -> stt.SpeechEvent:
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(text="", confidence=0.0, language=language)],
        )


class SpeechStream(stt.RecognizeStream):
    """One streaming recognition session over the call's lifetime.

    A single ``consume_vad`` task owns the WebSocket end to end (sole writer), so
    there are no socket races: it opens a session on VAD ``START_OF_SPEECH``,
    streams the live audio windows (``INFERENCE_DONE.frames``) while speaking, and
    sends ``{"type":"end"}`` on ``END_OF_SPEECH``. A per-session recv task emits the
    INTERIM/FINAL transcripts concurrently. ``forward_input`` only feeds the VAD.
    """

    def __init__(
        self,
        *,
        stt: NavaiWSSTT,
        vad: vad_module.VAD,
        language: str,
        conn_options: APIConnectOptions,
    ) -> None:
        # sample_rate=16k makes the base auto-resample every pushed frame to 16 kHz
        # before it reaches _input_ch, so we (and the internal VAD) see one rate.
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=SAMPLE_RATE)
        self._stt: NavaiWSSTT = stt
        self._vad = vad
        self._language = language

        self._http_session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._recv_task: Optional[asyncio.Task] = None
        self._closing_segment = False
        self._final_emitted = False
        self._last_interim = ""
        self._seg_started_at = 0.0
        self._first_audio_signaled = False

    async def _run(self) -> None:
        vad_stream = self._vad.stream()

        async def _forward_input() -> None:
            # Feed the internal VAD only. The VAD's INFERENCE_DONE frames are the
            # contiguous, non-overlapping audio windows we forward to the server.
            try:
                async for data in self._input_ch:
                    if isinstance(data, self._FlushSentinel):
                        vad_stream.flush()
                        continue
                    vad_stream.push_frame(data)
            finally:
                vad_stream.end_input()

        async def _consume_vad() -> None:
            async for ev in vad_stream:
                if ev.type == vad_module.VADEventType.START_OF_SPEECH:
                    await self._begin_segment(ev.frames)
                elif ev.type == vad_module.VADEventType.INFERENCE_DONE:
                    if self._ws is not None and not self._closing_segment:
                        await self._send_frames(ev.frames)
                elif ev.type == vad_module.VADEventType.END_OF_SPEECH:
                    await self._end_segment()
            # VAD stream exhausted (end_input): close any segment still open.
            await self._end_segment()

        forward = asyncio.create_task(_forward_input(), name="navai_ws_stt_forward")
        consume = asyncio.create_task(_consume_vad(), name="navai_ws_stt_consume")
        try:
            await asyncio.gather(forward, consume)
        finally:
            await aio.cancel_and_wait(forward, consume)
            await self._abort_session()
            await vad_stream.aclose()

    # ------------------------------------------------------------------ #
    # segment lifecycle (all called from the single consume_vad task)
    # ------------------------------------------------------------------ #
    async def _begin_segment(self, onset_frames: list[rtc.AudioFrame]) -> None:
        if self._ws is not None:
            # Defensive: a START without an intervening END. Finalize the old one.
            await self._end_segment()

        self._closing_segment = False
        self._final_emitted = False
        self._last_interim = ""
        self._seg_started_at = time.perf_counter()
        try:
            ws = await self._connect()
        except Exception as e:
            # Graceful: server unreachable → emit an empty FINAL so the turn still
            # commits instead of hanging until the framework's transcript timeout.
            logger.warning("NavAI WS STT: failed to open session: %s", e)
            self._emit_final("")
            self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH))
            return

        self._ws = ws
        self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH))
        self._recv_task = asyncio.create_task(self._recv_loop(ws), name="navai_ws_stt_recv")
        # The onset the VAD buffered before START fired (prefix padding + speech).
        await self._send_frames(onset_frames)

    async def _send_frames(self, frames: list[rtc.AudioFrame]) -> None:
        ws = self._ws
        if ws is None:
            return
        if not self._first_audio_signaled:
            self._first_audio_signaled = True
            cb = self._stt._on_first_audio
            if cb:
                try:
                    cb()
                except Exception:
                    pass
        for f in frames:
            data = f.data.tobytes() if hasattr(f.data, "tobytes") else bytes(f.data)
            if not data:
                continue
            try:
                await ws.send_bytes(data)
            except (ConnectionResetError, aiohttp.ClientError) as e:
                logger.warning("NavAI WS STT: send dropped (session closed): %s", e)
                await self._abort_session()
                return

    async def _end_segment(self) -> None:
        ws = self._ws
        if ws is None:
            return
        self._closing_segment = True
        try:
            await ws.send_str(json.dumps({"type": "end"}))
        except (ConnectionResetError, aiohttp.ClientError):
            pass

        if self._recv_task is not None:
            try:
                await asyncio.wait_for(self._recv_task, timeout=self._stt._max_utterance_s)
            except asyncio.TimeoutError:
                logger.warning(
                    "NavAI WS STT: 'final' not received within %.0fs; using last interim",
                    self._stt._max_utterance_s,
                )
                await aio.cancel_and_wait(self._recv_task)
            except Exception as e:
                logger.warning("NavAI WS STT recv task error: %s", e)

        # recv_loop emits FINAL on the server's `final`; if it never arrived
        # (timeout / mid-stream death) fall back to the best interim we saw.
        if not self._final_emitted:
            self._emit_final(self._last_interim)
        self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH))

        self._recv_task = None
        await self._close_ws()
        self._closing_segment = False

    async def _recv_loop(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        ev = json.loads(msg.data)
                    except (json.JSONDecodeError, ValueError):
                        continue
                    t = ev.get("type")
                    if t == "interim":
                        text = (ev.get("text") or "").strip()
                        if text:
                            self._last_interim = text
                            self._event_ch.send_nowait(
                                stt.SpeechEvent(
                                    type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                                    alternatives=[
                                        stt.SpeechData(
                                            text=text, language=self._language, confidence=1.0
                                        )
                                    ],
                                )
                            )
                    elif t == "final":
                        self._emit_final((ev.get("text") or "").strip() or self._last_interim)
                        # 'final' is terminal for this turn; 'done' usually follows.
                    elif t == "warn":
                        logger.warning("NavAI WS STT warn: %s", ev.get("message"))
                    elif t == "error":
                        logger.warning("NavAI WS STT server error: %s", ev.get("message"))
                        return  # keep whatever interims we emitted; let _end_segment finalize
                    elif t == "done":
                        return
                    # speech_start and unknown types: ignore
                elif msg.type in (
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.ERROR,
                ):
                    return
        except (ConnectionResetError, aiohttp.ClientError) as e:
            logger.warning("NavAI WS STT recv error: %s", e)
            return

    def _emit_final(self, text: str) -> None:
        if self._final_emitted:
            return
        self._final_emitted = True
        cb = self._stt._on_stt_duration
        if cb and self._seg_started_at:
            try:
                cb((time.perf_counter() - self._seg_started_at) * 1000)
            except Exception:
                pass
        self._event_ch.send_nowait(
            stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[
                    stt.SpeechData(
                        text=text, language=self._language, confidence=1.0 if text else 0.0
                    )
                ],
            )
        )
        if text:
            logger.info("NavAI WS STT FINAL: '%s'", text[:80])

    async def _connect(self) -> aiohttp.ClientWebSocketResponse:
        conn_timeout = aiohttp.ClientTimeout(
            total=None,
            connect=self._conn_options.timeout,
            sock_connect=self._conn_options.timeout,
        )
        session = aiohttp.ClientSession(timeout=conn_timeout)
        try:
            with network_request_context(
                "navai_ws_stt", "transcribe", metadata={"language": self._language}
            ):
                ws = await session.ws_connect(
                    self._stt._ws_url,
                    protocols=(SUBPROTOCOL,),
                    headers=self._stt._ws_headers(),
                    timeout=aiohttp.ClientWSTimeout(
                        ws_receive=self._conn_options.timeout, ws_close=5
                    ),
                    max_msg_size=0,
                    heartbeat=None,
                )
            await ws.send_str(
                json.dumps(
                    {
                        "type": "start",
                        "sample_rate": self._stt._sample_rate,
                        "num_channels": self._stt._num_channels,
                        "language": self._language,
                    }
                )
            )
        except Exception:
            await session.close()
            raise
        self._http_session = session
        return ws

    async def _close_ws(self) -> None:
        ws, session = self._ws, self._http_session
        self._ws = None
        self._http_session = None
        if ws is not None:
            try:
                await ws.close()
            except Exception:
                pass
        if session is not None:
            try:
                await session.close()
            except Exception:
                pass

    async def _abort_session(self) -> None:
        """Cleanup path (stream torn down): drop the WS without emitting a FINAL."""
        if self._recv_task is not None:
            await aio.cancel_and_wait(self._recv_task)
            self._recv_task = None
        await self._close_ws()
        self._closing_segment = False
