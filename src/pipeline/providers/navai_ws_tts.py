"""
NavAI WebSocket streaming TTS provider.

Streams raw PCM16 (mono, little-endian, 24 kHz) audio over a WebSocket as the
NavAI TTS model produces it — the first audio frame lands in ~430 ms, well ahead
of the round-trip HTTP providers (Yandex/Navai HTTP), which is the whole point of
moving off Yandex for latency.

Protocol (see ``scripts/tts/ws_client_example.py`` and the team's doc):

    client → {"text": "...", "voice_id": "navai"}              (one JSON frame)
    server → {"event":"start","sample_rate":24000,...}         (JSON control)
             <binary PCM16 chunk> <binary PCM16 chunk> ...      (binary audio)
             {"event":"done","chunks":N,"first_packet_ms":...}  (JSON control)
    on error → {"event":"error","detail":"..."}

Implemented against the LiveKit Agents 1.5.x ``ChunkedStream`` API (full text in,
incremental audio out) instead of the legacy hand-rolled iterator the other local
providers (yandex/navai/custom) use. That means audio frames stream to the session
as they arrive over the socket, and TTS time-to-first-byte lands in LiveKit's
native metrics — giving the Yandex-vs-NavAI latency comparison the team wants for
free.

Config: ``voice.tts_provider: navai_ws``. Env: ``NAVAI_WS_TTS_URL`` (base or full
``/synthesize/ws`` URL; accepts scheme-less host:port and http(s):// too),
``NAVAI_WS_VOICE_ID``, optional ``NAVAI_WS_TTS_MODE`` (``local``/``grpc``).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Optional

import aiohttp
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tts,
)
from livekit.agents.utils import shortuuid

from observability.network_topology import network_request_context, register_service_route

logger = logging.getLogger(__name__)

# The WS server emits PCM16 mono LE @ 24 kHz. Declare 24 kHz everywhere and let
# LiveKit resample down to the room rate — manual resampling here is the classic
# pitched/sped-up-audio bug.
SAMPLE_RATE = 24000
NUM_CHANNELS = 1
DEFAULT_VOICE_ID = ""
# Streaming TTS gateway endpoint. No hardcoded default host — set the
# NAVAI_WS_TTS_URL env var (TLS + API-key auth) to your own WS TTS gateway.
DEFAULT_WS_URL = ""
_SYNTHESIZE_PATH = "/api/tts/synthesize/ws"
# API key for the gateway. Sent as the ``X-API-Key`` header on the WS upgrade —
# our client isn't a browser, so we skip the ``?api_key=`` query-param fallback
# (which would also leak the key into network-topology URL logs).
API_KEY_ENV = "NAVAI_API_KEY"
# Hard ceiling on a single synthesis. The per-message ws_receive timeout catches a
# *stalled* server (no frame for N s); this catches a server that streams binary
# frames endlessly without ever sending 'done' (which would otherwise hang the
# whole voice pipeline). Env-overridable for tuning on dev.
DEFAULT_MAX_SYNTHESIS_SECONDS = 60.0
# Yandex voice ids — the schema default tts_voice_id is "yulduz", which does NOT
# exist on the NavAI WS server. Used only to warn on an obvious misconfiguration.
_YANDEX_VOICE_IDS = frozenset({"yulduz", "zamira", "nigora"})


def _normalize_tts_text(text: str) -> str:
    """Normalize text before synthesis (collapse whitespace).

    TODO(urdu): add tenant/language-specific spoken-form expansions here. NavAI's
    Uzbek government abbreviations were removed for the generic blueprint.
    """
    if not text:
        return text
    return re.sub(r"\s+", " ", text).strip()


def _to_ws_synthesize_url(raw: str) -> str:
    """Normalize a configured URL into a full ``ws(s)://host/api/tts/synthesize/ws`` URL.

    Accepts scheme-less ``host[:port]``, ``http(s)://`` (auto-converted to
    ``ws(s)://``), and URLs that already carry a synth path (legacy ``/synthesize/ws``
    or the new gateway path).
    """
    raw = (raw or "").strip().rstrip("/")
    if raw.startswith("http://"):
        raw = "ws://" + raw[len("http://") :]
    elif raw.startswith("https://"):
        raw = "wss://" + raw[len("https://") :]
    elif not raw.startswith(("ws://", "wss://")):
        raw = "ws://" + raw  # scheme-less, e.g. "tts-gateway.example.com"

    if raw.endswith(("/synthesize/ws", "/ws/synthesize")):
        return raw
    return raw + _SYNTHESIZE_PATH


def _ws_to_http_origin(ws_url: str) -> str:
    """ws(s):// → http(s):// origin, for the network-topology service route label."""
    origin = ws_url.split("/synthesize/ws")[0].split("/ws/synthesize")[0]
    if origin.startswith("wss://"):
        return "https://" + origin[len("wss://") :]
    if origin.startswith("ws://"):
        return "http://" + origin[len("ws://") :]
    return origin


class NavaiWSTTS(tts.TTS):
    """NavAI streaming TTS over a WebSocket (PCM16 mono @ 24 kHz)."""

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        voice: str = DEFAULT_VOICE_ID,
        speed: float = 1.0,
        sample_rate: int = SAMPLE_RATE,
        num_channels: int = NUM_CHANNELS,
        mode: Optional[str] = None,
        reference_text: Optional[str] = None,
        max_synthesis_s: Optional[float] = None,
        api_key: Optional[str] = None,
    ) -> None:
        # The agent sentence-segments upstream and the server takes one full-text
        # message, so this is a ChunkedStream provider (streaming=False). Output
        # still streams frame-by-frame via incremental AudioEmitter.push().
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=sample_rate,
            num_channels=num_channels,
        )

        self._ws_url = _to_ws_synthesize_url(
            base_url or os.getenv("NAVAI_WS_TTS_URL", DEFAULT_WS_URL)
        )
        self._voice = voice or os.getenv("NAVAI_WS_VOICE_ID", DEFAULT_VOICE_ID)
        self._speed = speed  # not supported by the WS protocol; accepted for parity
        self._sample_rate = sample_rate
        self._num_channels = num_channels
        self._mode = (mode or os.getenv("NAVAI_WS_TTS_MODE") or "").strip() or None
        self._reference_text = reference_text
        self._max_synthesis_s = max_synthesis_s or float(
            os.getenv("NAVAI_WS_TTS_MAX_SECONDS", DEFAULT_MAX_SYNTHESIS_SECONDS)
        )
        self._api_key = (api_key or os.getenv(API_KEY_ENV) or "").strip() or None

        if self._speed != 1.0:
            logger.debug(
                "NavAI WS TTS: speed=%s is ignored (WS protocol has no speed field)", self._speed
            )
        if self._voice in _YANDEX_VOICE_IDS:
            logger.warning(
                "NavAI WS TTS voice=%r is a Yandex voice that does not exist on the NavAI "
                "server; set tenant voice.tts_voice_id (or voice.voices[lang]) to a NavAI "
                "voice such as 'navai'. Synthesis will likely fail until corrected.",
                self._voice,
            )

        register_service_route(
            "navai_ws_tts",
            _ws_to_http_origin(self._ws_url),
            provider="navai_ws_tts",
            metadata={"voice": self._voice, "sample_rate": self._sample_rate, "mode": self._mode},
            notes="NavAI streaming TTS over WebSocket (PCM16 @ 24 kHz)",
        )
        logger.info(
            "NavAI WS TTS initialized: url=%s voice=%s sr=%d mode=%s auth=%s",
            self._ws_url,
            self._voice,
            self._sample_rate,
            self._mode,
            "on" if self._api_key else "off",
        )

    def _ws_headers(self) -> dict:
        """Auth header for the gateway WS upgrade (empty when no key configured)."""
        return {"X-API-Key": self._api_key} if self._api_key else {}

    async def aclose(self) -> None:
        # Each synthesis opens and closes its own ClientSession via `async with`
        # (a WS connection isn't pooled/reused anyway), so there is no persistent
        # session to leak even if the framework never calls this.
        await super().aclose()

    # The other providers expose close(); keep an alias so existing shutdown
    # paths that call either method work.
    async def close(self) -> None:
        await self.aclose()

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "ChunkedStream":
        return ChunkedStream(
            tts=self,
            input_text=_normalize_tts_text(text),
            conn_options=conn_options,
        )


class ChunkedStream(tts.ChunkedStream):
    """Streams one full-text request's audio over the WebSocket into the emitter."""

    def __init__(
        self,
        *,
        tts: NavaiWSTTS,
        input_text: str,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: NavaiWSTTS = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = shortuuid()
        # raw PCM16 → audio/pcm hits the AudioEmitter raw path (no decoder). Pushing
        # bytes incrementally streams output regardless of stream=False (which only
        # gates multi-segment input). Must be called before returning so the base's
        # end_input()/join() after _run has a started emitter.
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._tts._sample_rate,
            num_channels=self._tts._num_channels,
            mime_type="audio/pcm",
        )

        text = (self._input_text or "").strip()
        if not text:
            return

        payload: dict[str, str] = {"text": text, "voice_id": self._tts._voice}
        if self._tts._reference_text:
            payload["reference_text"] = self._tts._reference_text
        if self._tts._mode:
            payload["mode"] = self._tts._mode

        # Once we've pushed any audio, an error must NOT propagate: the base
        # ChunkedStream retries _run on APIError up to conn_options.max_retry times
        # (default 3) using the SAME output channel, with no pushed-audio guard
        # (unlike SynthesizeStream). A second attempt would append a full take after
        # the partial one → the caller hears doubled/garbled speech. So after any
        # bytes are delivered we break/return and keep the partial audio. Only a
        # failure with zero audio raises (a clean, safe retry).
        n_bytes = 0
        try:
            # Total-synthesis ceiling: bounds a server that streams forever without
            # 'done'. The per-message ws_receive timeout only bounds a stalled server.
            async with asyncio.timeout(self._tts._max_synthesis_s):
                # One ClientSession per synthesis, always closed by `async with`
                # (WS connections aren't pooled), so nothing leaks if aclose() is
                # never called. Bound the TCP/upgrade connect explicitly.
                conn_timeout = aiohttp.ClientTimeout(
                    total=None,
                    connect=self._conn_options.timeout,
                    sock_connect=self._conn_options.timeout,
                )
                with network_request_context(
                    "navai_ws_tts",
                    "synthesize",
                    metadata={"voice": self._tts._voice, "text_length": len(text)},
                ):
                    async with aiohttp.ClientSession(timeout=conn_timeout) as session:
                        async with session.ws_connect(
                            self._tts._ws_url,
                            headers=self._tts._ws_headers(),
                            timeout=aiohttp.ClientWSTimeout(
                                ws_receive=self._conn_options.timeout, ws_close=5
                            ),
                            max_msg_size=0,  # unlimited, matches the doc's max_size=None
                            heartbeat=None,
                        ) as ws:
                            await ws.send_str(json.dumps(payload, ensure_ascii=False))

                            async for msg in ws:
                                if msg.type == aiohttp.WSMsgType.BINARY:
                                    if msg.data:
                                        output_emitter.push(msg.data)
                                        n_bytes += len(msg.data)
                                elif msg.type == aiohttp.WSMsgType.TEXT:
                                    try:
                                        event = json.loads(msg.data)
                                    except (json.JSONDecodeError, ValueError):
                                        logger.warning(
                                            "NavAI WS TTS: ignoring non-JSON control frame: %r",
                                            msg.data[:200] if msg.data else msg.data,
                                        )
                                        continue
                                    etype = event.get("event")
                                    if etype == "start":
                                        server_sr = event.get("sample_rate")
                                        if server_sr and server_sr != self._tts._sample_rate:
                                            logger.warning(
                                                "NavAI WS TTS sample_rate mismatch: server=%s "
                                                "configured=%s — audio will be mispitched; align "
                                                "NavaiWSTTS sample_rate.",
                                                server_sr,
                                                self._tts._sample_rate,
                                            )
                                    elif etype == "done":
                                        logger.debug(
                                            "NavAI WS TTS done: bytes=%d first_packet_ms=%s "
                                            "audio_seconds=%s",
                                            n_bytes,
                                            event.get("first_packet_ms"),
                                            event.get("audio_seconds"),
                                        )
                                        break
                                    elif etype == "error":
                                        detail = event.get("detail", "navai_ws_tts server error")
                                        if n_bytes > 0:
                                            logger.warning(
                                                "NavAI WS TTS error after %d bytes; keeping "
                                                "partial audio (no retry): %s",
                                                n_bytes,
                                                detail,
                                            )
                                            break
                                        raise APIStatusError(
                                            detail,
                                            status_code=-1,
                                            request_id=request_id,
                                            body=msg.data,
                                        )
                                    else:
                                        logger.debug(
                                            "NavAI WS TTS: ignoring unknown event %r", etype
                                        )
                                elif msg.type == aiohttp.WSMsgType.ERROR:
                                    if n_bytes > 0:
                                        logger.warning(
                                            "NavAI WS TTS socket error after %d bytes; keeping "
                                            "partial audio (no retry): %s",
                                            n_bytes,
                                            ws.exception(),
                                        )
                                        break
                                    raise APIConnectionError(
                                        f"navai_ws_tts websocket error: {ws.exception()}"
                                    )
                                elif msg.type in (
                                    aiohttp.WSMsgType.CLOSE,
                                    aiohttp.WSMsgType.CLOSING,
                                    aiohttp.WSMsgType.CLOSED,
                                ):
                                    # Server closed before 'done'. Keep whatever audio
                                    # we got; the base raises only if zero was pushed.
                                    logger.warning(
                                        "NavAI WS TTS socket closed before 'done' (bytes=%d)",
                                        n_bytes,
                                    )
                                    break
        except (APIStatusError, APIConnectionError, APITimeoutError):
            raise
        except (asyncio.TimeoutError, TimeoutError) as e:
            if n_bytes > 0:
                logger.warning(
                    "NavAI WS TTS hit the %.0fs synthesis ceiling after %d bytes; "
                    "keeping partial audio (no retry).",
                    self._tts._max_synthesis_s,
                    n_bytes,
                )
                return
            raise APITimeoutError() from e
        except aiohttp.ClientError as e:
            if n_bytes > 0:
                logger.warning(
                    "NavAI WS TTS connection dropped after %d bytes; keeping partial "
                    "audio (no retry): %s",
                    n_bytes,
                    e,
                )
                return
            raise APIConnectionError(f"navai_ws_tts connection failed: {e}") from e
        # Do NOT call flush()/end_input() — the ChunkedStream base does that and
        # joins after _run returns (matches the bundled google/openai plugins).
