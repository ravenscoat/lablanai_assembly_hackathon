"""Tests for the NavAI WebSocket streaming STT provider.

Runs the provider against an in-process aiohttp WebSocket server that speaks the
real protocol (start JSON → binary PCM16 → interim/final/done JSON) and drives the
internal endpointing with a fake VAD that scripts START/INFERENCE_DONE/END events.
This exercises the segment lifecycle, interim→final emission, the VAD-driven `end`,
and graceful degradation without the (firewalled) dev server or real Silero VAD.
"""

from __future__ import annotations

import asyncio
import json

import aiohttp
import numpy as np
import pytest
from aiohttp import web
from livekit import rtc
from livekit.agents import APIConnectOptions, stt
from livekit.agents import vad as vad_module

from pipeline.providers.navai_ws_stt import (
    NavaiWSSTT,
    _to_ws_transcribe_url,
    _ws_to_http_origin,
)

SAMPLE_RATE = 16000


# --------------------------------------------------------------------------- #
# fakes
# --------------------------------------------------------------------------- #
def _frame(seconds: float) -> rtc.AudioFrame:
    """Silence PCM16 mono @ 16 kHz (content is irrelevant — the VAD is faked)."""
    n = int(seconds * SAMPLE_RATE)
    return rtc.AudioFrame(
        data=np.zeros(n, dtype="<i2").tobytes(),
        sample_rate=SAMPLE_RATE,
        num_channels=1,
        samples_per_channel=n,
    )


def _vad_event(t: vad_module.VADEventType, frames=None) -> vad_module.VADEvent:
    return vad_module.VADEvent(
        type=t,
        samples_index=0,
        timestamp=0.0,
        speech_duration=0.0,
        silence_duration=0.0,
        frames=frames or [],
    )


class _FakeVADStream:
    """Turns pushed frames into a scripted START → INFERENCE_DONE* → END sequence.

    First frame of a segment → START_OF_SPEECH (carrying that frame as the onset).
    Subsequent frames → INFERENCE_DONE (one window each). flush()/end_input() while
    in a segment → END_OF_SPEECH. end_input() also yields the stop sentinel.
    """

    def __init__(self) -> None:
        self._q: asyncio.Queue = asyncio.Queue()
        self._in_segment = False

    def push_frame(self, frame: rtc.AudioFrame) -> None:
        if not self._in_segment:
            self._in_segment = True
            self._q.put_nowait(_vad_event(vad_module.VADEventType.START_OF_SPEECH, [frame]))
        else:
            self._q.put_nowait(_vad_event(vad_module.VADEventType.INFERENCE_DONE, [frame]))

    def flush(self) -> None:
        if self._in_segment:
            self._in_segment = False
            self._q.put_nowait(_vad_event(vad_module.VADEventType.END_OF_SPEECH))

    def end_input(self) -> None:
        if self._in_segment:
            self._in_segment = False
            self._q.put_nowait(_vad_event(vad_module.VADEventType.END_OF_SPEECH))
        self._q.put_nowait(None)

    def __aiter__(self) -> "_FakeVADStream":
        return self

    async def __anext__(self) -> vad_module.VADEvent:
        item = await self._q.get()
        if item is None:
            raise StopAsyncIteration
        return item

    async def aclose(self) -> None:
        pass


class _FakeVAD:
    def stream(self) -> _FakeVADStream:
        return _FakeVADStream()


# --------------------------------------------------------------------------- #
# in-process server
# --------------------------------------------------------------------------- #
def _make_app(*, behavior: str = "ok"):
    """behavior: "ok" | "no_final" | "error" | "error_mid"."""
    captured: dict = {
        "start": None,
        "audio_bytes": 0,
        "got_end": False,
        "connections": 0,
        "api_key": None,
    }

    async def handler(request: web.Request) -> web.WebSocketResponse:
        captured["connections"] += 1
        captured["api_key"] = request.headers.get("X-API-Key")
        ws = web.WebSocketResponse(protocols=("stt.v1",))
        await ws.prepare(request)

        first = await ws.receive()
        captured["start"] = json.loads(first.data)
        await ws.send_str(json.dumps({"type": "speech_start"}))

        sent_interim = False
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.BINARY:
                captured["audio_bytes"] += len(msg.data)
                if not sent_interim:
                    sent_interim = True
                    await ws.send_str(
                        json.dumps({"type": "interim", "text": "salom", "language": "uz"})
                    )
                    if behavior == "error_mid":
                        await ws.send_str(json.dumps({"type": "error", "message": "died"}))
                        await ws.close()
                        return ws
            elif msg.type == aiohttp.WSMsgType.TEXT:
                ev = json.loads(msg.data)
                if ev.get("type") == "end":
                    captured["got_end"] = True
                    if behavior == "ok":
                        await ws.send_str(
                            json.dumps({"type": "final", "text": "salom dunyo", "language": "uz"})
                        )
                        await ws.send_str(json.dumps({"type": "done"}))
                    elif behavior == "no_final":
                        await ws.send_str(json.dumps({"type": "done"}))
                    elif behavior == "error":
                        await ws.send_str(json.dumps({"type": "error", "message": "boom"}))
                    await ws.close()
                    return ws
        return ws

    app = web.Application()
    app.router.add_get("/api/stt/transcribe/live", handler)
    return app, captured


async def _run_server(app) -> tuple[web.AppRunner, str]:
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = list(runner.addresses)[0][1]
    return runner, f"ws://127.0.0.1:{port}"


async def _collect(stream) -> list:
    evs = []
    async for ev in stream:
        evs.append(ev)
    return evs


def _types(evs) -> list:
    return [e.type for e in evs]


def _finals(evs) -> list[str]:
    return [
        e.alternatives[0].text
        for e in evs
        if e.type == stt.SpeechEventType.FINAL_TRANSCRIPT and e.alternatives
    ]


# --------------------------------------------------------------------------- #
# pure helpers
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "raw,expected",
    [
        # bare host / scheme-less → ws:// + gateway path
        ("stt.example.com", "ws://stt.example.com/api/stt/transcribe/live"),
        ("https://stt.example.com", "wss://stt.example.com/api/stt/transcribe/live"),
        # full gateway URL preserved as-is (ends with /transcribe/live)
        (
            "wss://stt.example.com/api/stt/transcribe/live",
            "wss://stt.example.com/api/stt/transcribe/live",
        ),
        ("https://stt.example.com/", "wss://stt.example.com/api/stt/transcribe/live"),
        # legacy raw-server full URL still preserved
        ("ws://h:8080/transcribe/live", "ws://h:8080/transcribe/live"),
        ("http://h:8080", "ws://h:8080/api/stt/transcribe/live"),
    ],
)
def test_url_normalization(raw, expected):
    assert _to_ws_transcribe_url(raw) == expected


def test_ws_to_http_origin():
    assert (
        _ws_to_http_origin("wss://stt.example.com/api/stt/transcribe/live")
        == "https://stt.example.com/api/stt"
    )
    assert _ws_to_http_origin("ws://h:8080/transcribe/live") == "http://h:8080"


def test_capabilities_are_streaming():
    s = NavaiWSSTT(base_url="ws://127.0.0.1:9", vad=_FakeVAD())
    assert s.capabilities.streaming is True
    assert s.capabilities.interim_results is True


# --------------------------------------------------------------------------- #
# streaming behavior against the in-process server
# --------------------------------------------------------------------------- #
async def test_streaming_emits_interim_and_final():
    app, captured = _make_app(behavior="ok")
    runner, base = await _run_server(app)
    try:
        s = NavaiWSSTT(base_url=base, language="uz", vad=_FakeVAD())
        stream = s.stream(conn_options=APIConnectOptions(max_retry=0, timeout=5))
        for _ in range(5):
            stream.push_frame(_frame(0.08))
        stream.end_input()

        evs = await asyncio.wait_for(_collect(stream), timeout=10)

        assert stt.SpeechEventType.INTERIM_TRANSCRIPT in _types(evs)
        assert stt.SpeechEventType.FINAL_TRANSCRIPT in _types(evs)
        assert _finals(evs)[-1] == "salom dunyo"
        # well-formed start handshake + audio streamed + flush sent
        assert captured["start"] == {
            "type": "start",
            "sample_rate": 16000,
            "num_channels": 1,
            "language": "uz",
        }
        assert captured["got_end"] is True
        assert captured["audio_bytes"] > 0
        assert captured["api_key"] is None  # no key configured → header absent
        await s.aclose()
    finally:
        await runner.cleanup()


async def test_api_key_sent_as_header_on_upgrade():
    app, captured = _make_app(behavior="ok")
    runner, base = await _run_server(app)
    try:
        s = NavaiWSSTT(base_url=base, language="uz", vad=_FakeVAD(), api_key="secret-key")
        stream = s.stream(conn_options=APIConnectOptions(max_retry=0, timeout=5))
        for _ in range(3):
            stream.push_frame(_frame(0.08))
        stream.end_input()
        await asyncio.wait_for(_collect(stream), timeout=10)
        # key travels in the X-API-Key header, not the URL (no log leak)
        assert captured["api_key"] == "secret-key"
        assert "secret-key" not in s._ws_url
        await s.aclose()
    finally:
        await runner.cleanup()


async def test_final_falls_back_to_last_interim_when_server_sends_no_final():
    app, _ = _make_app(behavior="no_final")
    runner, base = await _run_server(app)
    try:
        s = NavaiWSSTT(base_url=base, language="uz", vad=_FakeVAD())
        stream = s.stream(conn_options=APIConnectOptions(max_retry=0, timeout=5))
        for _ in range(3):
            stream.push_frame(_frame(0.08))
        stream.end_input()

        evs = await asyncio.wait_for(_collect(stream), timeout=10)
        # no `final` from the server → the last interim is promoted to FINAL
        assert _finals(evs)[-1] == "salom"
        await s.aclose()
    finally:
        await runner.cleanup()


async def test_server_error_after_interim_keeps_partial_no_crash():
    app, _ = _make_app(behavior="error_mid")
    runner, base = await _run_server(app)
    try:
        s = NavaiWSSTT(base_url=base, language="uz", vad=_FakeVAD())
        stream = s.stream(conn_options=APIConnectOptions(max_retry=0, timeout=5))
        for _ in range(3):
            stream.push_frame(_frame(0.08))
        stream.end_input()

        evs = await asyncio.wait_for(_collect(stream), timeout=10)
        # error mid-stream must not raise; the interim we got is kept as FINAL
        assert _finals(evs)[-1] == "salom"
        assert stt.SpeechEventType.END_OF_SPEECH in _types(evs)
        await s.aclose()
    finally:
        await runner.cleanup()


async def test_connect_failure_emits_empty_final_gracefully():
    # No server here — connection is refused, so opening the segment fails.
    s = NavaiWSSTT(base_url="ws://127.0.0.1:1", language="uz", vad=_FakeVAD())
    stream = s.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
    for _ in range(2):
        stream.push_frame(_frame(0.08))
    stream.end_input()

    evs = await asyncio.wait_for(_collect(stream), timeout=10)
    # graceful: an (empty) FINAL so the turn still commits, and no exception
    assert stt.SpeechEventType.FINAL_TRANSCRIPT in _types(evs)
    assert _finals(evs)[-1] == ""
    await s.aclose()


async def test_batch_recognize_returns_final():
    app, captured = _make_app(behavior="ok")
    runner, base = await _run_server(app)
    try:
        s = NavaiWSSTT(base_url=base, language="uz", vad=_FakeVAD())
        ev = await s.recognize(
            buffer=_frame(0.2), conn_options=APIConnectOptions(max_retry=0, timeout=5)
        )
        assert ev.type == stt.SpeechEventType.FINAL_TRANSCRIPT
        assert ev.alternatives[0].text == "salom dunyo"
        assert captured["got_end"] is True
        await s.aclose()
    finally:
        await runner.cleanup()


# --------------------------------------------------------------------------- #
# factory wiring
# --------------------------------------------------------------------------- #
def _factory_stt(voice_block: dict, vad=None):
    from config.schema import TenantConfig
    from pipeline.voice_factory import VoiceFactory

    config = TenantConfig.model_validate(
        {"tenant": {"id": "t", "slug": "t", "name": "T"}, "voice": voice_block}
    )
    return VoiceFactory.create_stt_for_language(config, "uz", vad=vad)


def test_factory_returns_navai_ws_stt_and_threads_vad():
    sentinel = _FakeVAD()
    s = _factory_stt({"stt_provider": "navai_ws"}, vad=sentinel)
    assert isinstance(s, NavaiWSSTT)
    assert s._vad is sentinel
    assert s._language == "uz"
