"""Tests for the NavAI WebSocket streaming TTS provider.

Runs the provider against an in-process aiohttp WebSocket server that speaks the
real protocol (start JSON → binary PCM16 frames → done JSON), so the streaming
loop, raw-PCM emitter path, error mapping, and mid-stream disconnect are all
exercised without the (firewalled) dev server.
"""

from __future__ import annotations

import asyncio
import json

import numpy as np
import pytest
from aiohttp import web
from livekit.agents import APIConnectOptions, APIStatusError

from pipeline.providers.navai_ws_tts import (
    NavaiWSTTS,
    _to_ws_synthesize_url,
    _ws_to_http_origin,
)

SAMPLE_RATE = 24000


def _make_pcm(seconds: float) -> bytes:
    """A quiet sine tone as PCM16 mono @ 24 kHz."""
    n = int(seconds * SAMPLE_RATE)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    wave_ = (np.sin(2 * np.pi * 220 * t) * 3000).astype("<i2")
    return wave_.tobytes()


def _make_app(*, behavior: str, audio: bytes, n_chunks: int = 10):
    """Build an aiohttp app whose /synthesize/ws handler mimics the TTS server.

    behavior: "ok" | "error" | "disconnect" | "error_after_audio" | "bad_json" | "never_done"
    captured["connections"] counts handler invocations (to assert no-retry).
    """
    captured: dict = {"connections": 0, "api_key": None}

    async def handler(request: web.Request) -> web.WebSocketResponse:
        captured["connections"] += 1
        captured["api_key"] = request.headers.get("X-API-Key")
        ws = web.WebSocketResponse(max_msg_size=0)
        await ws.prepare(request)

        first = await ws.receive()
        captured["request"] = json.loads(first.data)

        await ws.send_str(
            json.dumps(
                {
                    "event": "start",
                    "sample_rate": SAMPLE_RATE,
                    "channels": 1,
                    "bit_depth": 16,
                    "format": "pcm16",
                }
            )
        )

        if behavior == "error":
            await ws.send_str(json.dumps({"event": "error", "detail": "synthesis failed"}))
            await ws.close()
            return ws

        # stream the audio as binary chunks
        step = max(1, len(audio) // n_chunks)
        # keep step even so we never split a PCM16 sample across frames
        if step % 2:
            step += 1
        sent = 0
        injected_bad_json = False
        for i in range(0, len(audio), step):
            await ws.send_bytes(audio[i : i + step])
            sent += len(audio[i : i + step])
            half = sent >= len(audio) // 2

            if behavior == "disconnect" and half:
                await ws.close()
                return ws
            if behavior == "error_after_audio" and half:
                await ws.send_str(json.dumps({"event": "error", "detail": "died mid-stream"}))
                await ws.close()
                return ws
            if behavior == "never_done" and half:
                # never send 'done'; block until the client times out and closes
                # (returns promptly on disconnect, so runner.cleanup() stays fast)
                await ws.receive()
                return ws
            if behavior == "bad_json" and half and not injected_bad_json:
                await ws.send_str("this is not json {{{")  # must be tolerated
                injected_bad_json = True

        await ws.send_str(
            json.dumps(
                {
                    "event": "done",
                    "sample_rate": SAMPLE_RATE,
                    "chunks": n_chunks,
                    "first_packet_ms": 430.0,
                    "audio_seconds": len(audio) / 2 / SAMPLE_RATE,
                }
            )
        )
        await ws.close()
        return ws

    app = web.Application()
    app.router.add_get("/api/tts/synthesize/ws", handler)
    return app, captured


async def _run_server(app) -> tuple[web.AppRunner, str]:
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = list(runner.addresses)[0][1]
    return runner, f"ws://127.0.0.1:{port}"


async def _collect_frames(stream) -> list:
    frames = []
    async for ev in stream:
        frames.append(ev.frame)
    return frames


# --------------------------------------------------------------------------- #
# pure helpers
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "raw,expected",
    [
        # bare host / scheme-less → ws:// + gateway path
        ("tts.example.com", "ws://tts.example.com/api/tts/synthesize/ws"),
        ("https://tts.example.com", "wss://tts.example.com/api/tts/synthesize/ws"),
        ("https://tts.example.com/", "wss://tts.example.com/api/tts/synthesize/ws"),
        # full gateway URL preserved (ends with /synthesize/ws)
        (
            "wss://tts.example.com/api/tts/synthesize/ws",
            "wss://tts.example.com/api/tts/synthesize/ws",
        ),
        # legacy raw-server full URLs still preserved (both path aliases)
        ("ws://h:5000/synthesize/ws", "ws://h:5000/synthesize/ws"),
        ("ws://h:5000/ws/synthesize", "ws://h:5000/ws/synthesize"),
    ],
)
def test_url_normalization(raw, expected):
    assert _to_ws_synthesize_url(raw) == expected


def test_ws_to_http_origin():
    assert (
        _ws_to_http_origin("wss://tts.example.com/api/tts/synthesize/ws")
        == "https://tts.example.com/api/tts"
    )
    assert _ws_to_http_origin("ws://h:5000/synthesize/ws") == "http://h:5000"


def test_capabilities_and_rate():
    t = NavaiWSTTS(base_url="ws://127.0.0.1:9", voice="navai")
    assert t.sample_rate == SAMPLE_RATE
    assert t.num_channels == 1
    assert t.capabilities.streaming is False


# --------------------------------------------------------------------------- #
# protocol behavior against the in-process server
# --------------------------------------------------------------------------- #


async def test_happy_path_streams_all_audio():
    audio = _make_pcm(1.0)  # 1 s → 48000 bytes → multiple 200ms frames
    app, captured = _make_app(behavior="ok", audio=audio, n_chunks=12)
    runner, base = await _run_server(app)
    try:
        tts = NavaiWSTTS(base_url=base, voice="navai", mode="grpc", reference_text="ref")
        stream = tts.synthesize("Salom dunyo, bu test.")
        frames = await _collect_frames(stream)

        # request was well-formed
        req = captured["request"]
        assert req["text"] == "Salom dunyo, bu test."
        assert req["voice_id"] == "navai"
        assert req["mode"] == "grpc"
        assert req["reference_text"] == "ref"

        # output streamed as several frames and lost no samples (frame.data is a
        # memoryview of int16 — count bytes via .nbytes, not len()).
        assert len(frames) >= 2, "expected streamed (multi-frame) output"
        total_bytes = sum(f.data.nbytes for f in frames)
        assert total_bytes == len(audio)
        assert all(f.sample_rate == SAMPLE_RATE for f in frames)
        assert all(f.num_channels == 1 for f in frames)
        assert captured["api_key"] is None  # no key configured → header absent
        await tts.aclose()
    finally:
        await runner.cleanup()


async def test_api_key_sent_as_header_on_upgrade():
    app, captured = _make_app(behavior="ok", audio=_make_pcm(0.5), n_chunks=6)
    runner, base = await _run_server(app)
    try:
        tts = NavaiWSTTS(base_url=base, voice="navai", api_key="secret-key")
        await _collect_frames(tts.synthesize("Salom"))
        # key travels in the X-API-Key header, not the URL (no log leak)
        assert captured["api_key"] == "secret-key"
        assert "secret-key" not in tts._ws_url
        await tts.aclose()
    finally:
        await runner.cleanup()


async def test_server_error_event_raises_api_error():
    app, _ = _make_app(behavior="error", audio=b"")
    runner, base = await _run_server(app)
    try:
        tts = NavaiWSTTS(base_url=base, voice="navai")
        # max_retry=0 so we assert on the first attempt, not 4 retries
        stream = tts.synthesize("Salom", conn_options=APIConnectOptions(max_retry=0, timeout=5))
        with pytest.raises(APIStatusError):
            await _collect_frames(stream)
        await tts.aclose()
    finally:
        await runner.cleanup()


async def test_mid_stream_disconnect_keeps_received_audio():
    audio = _make_pcm(1.0)
    app, _ = _make_app(behavior="disconnect", audio=audio, n_chunks=12)
    runner, base = await _run_server(app)
    try:
        tts = NavaiWSTTS(base_url=base, voice="navai")
        stream = tts.synthesize(
            "Salom dunyo", conn_options=APIConnectOptions(max_retry=0, timeout=5)
        )
        frames = await _collect_frames(stream)
        # got the first half before the socket closed, without raising
        total_bytes = sum(f.data.nbytes for f in frames)
        assert 0 < total_bytes <= len(audio)
        await tts.aclose()
    finally:
        await runner.cleanup()


async def test_empty_text_produces_no_request():
    app, captured = _make_app(behavior="ok", audio=_make_pcm(0.5))
    runner, base = await _run_server(app)
    try:
        tts = NavaiWSTTS(base_url=base, voice="navai")
        stream = tts.synthesize("   ")
        frames = await _collect_frames(stream)
        assert frames == []
        assert "request" not in captured  # never connected
        await tts.aclose()
    finally:
        await runner.cleanup()


async def test_error_after_partial_audio_keeps_audio_and_does_not_retry():
    # The base ChunkedStream retries _run on a raised APIError with no pushed-audio
    # guard — retrying after partial audio would double the speech. The provider must
    # break (keep partial) instead of raising once any bytes were delivered.
    audio = _make_pcm(1.0)
    app, captured = _make_app(behavior="error_after_audio", audio=audio, n_chunks=12)
    runner, base = await _run_server(app)
    try:
        tts = NavaiWSTTS(base_url=base, voice="navai")
        # Default max_retry=3: a buggy raise here would reconnect 4 times.
        stream = tts.synthesize("Salom dunyo", conn_options=APIConnectOptions(timeout=5))
        frames = await _collect_frames(stream)
        total_bytes = sum(f.data.nbytes for f in frames)
        assert 0 < total_bytes <= len(audio)  # kept the partial audio
        assert captured["connections"] == 1  # did NOT retry over delivered audio
        await tts.aclose()
    finally:
        await runner.cleanup()


async def test_malformed_json_control_frame_is_ignored():
    audio = _make_pcm(1.0)
    app, _ = _make_app(behavior="bad_json", audio=audio, n_chunks=12)
    runner, base = await _run_server(app)
    try:
        tts = NavaiWSTTS(base_url=base, voice="navai")
        stream = tts.synthesize(
            "Salom dunyo", conn_options=APIConnectOptions(max_retry=0, timeout=5)
        )
        frames = await _collect_frames(stream)  # must not raise on the bad frame
        total_bytes = sum(f.data.nbytes for f in frames)
        assert total_bytes == len(audio)  # all audio still received
        await tts.aclose()
    finally:
        await runner.cleanup()


async def test_total_synthesis_timeout_keeps_partial_without_hanging():
    audio = _make_pcm(1.0)
    app, _ = _make_app(behavior="never_done", audio=audio, n_chunks=12)
    runner, base = await _run_server(app)
    try:
        # tiny ceiling so the test is fast; server never sends 'done'
        tts = NavaiWSTTS(base_url=base, voice="navai", max_synthesis_s=0.5)
        stream = tts.synthesize(
            "Salom dunyo", conn_options=APIConnectOptions(max_retry=0, timeout=5)
        )
        frames = await asyncio.wait_for(_collect_frames(stream), timeout=5)
        total_bytes = sum(f.data.nbytes for f in frames)
        assert total_bytes > 0  # delivered the partial audio, then bailed out
        await tts.aclose()
    finally:
        await runner.cleanup()


# --------------------------------------------------------------------------- #
# factory wiring
# --------------------------------------------------------------------------- #


def _factory_tts(voice_block: dict):
    from config.schema import TenantConfig
    from pipeline.voice_factory import VoiceFactory

    config = TenantConfig.model_validate(
        {"tenant": {"id": "t", "slug": "t", "name": "T"}, "voice": voice_block}
    )
    return VoiceFactory.create_tts(config)


def test_factory_returns_navai_ws_provider():
    tts = _factory_tts({"tts_provider": "navai_ws", "tts_voice_id": "navai"})
    assert isinstance(tts, NavaiWSTTS)
    assert tts._voice == "navai"


def test_factory_navai_ws_defaults_to_empty_when_voice_unset():
    # A tenant that flips tts_provider -> navai_ws WITHOUT setting tts_voice_id must
    # NOT send the Yandex schema default ("yulduz") to the WS server. The provider
    # default voice id is now empty (genericized) — the server applies its own
    # default voice when an empty voice id is sent.
    tts = _factory_tts({"tts_provider": "navai_ws"})
    assert isinstance(tts, NavaiWSTTS)
    assert tts._voice == ""


def test_factory_navai_ws_honors_explicit_per_language_voice():
    tts = _factory_tts({"tts_provider": "navai_ws", "voices": {"uz": "navai_custom"}})
    assert isinstance(tts, NavaiWSTTS)
    assert tts._voice == "navai_custom"
