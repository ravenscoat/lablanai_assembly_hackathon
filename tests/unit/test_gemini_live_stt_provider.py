from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from livekit.agents import stt

from pipeline.providers.gemini_live_stt import GeminiLiveSTT


class _FakeResponse:
    def __init__(self, text: str | None):
        self.text = text


class _FakeSession:
    def __init__(self, responses):
        self._responses = responses
        self.sent = []

    async def send(self, **kwargs):
        self.sent.append(kwargs)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def receive(self):
        async def _gen():
            for r in self._responses:
                yield r

        return _gen()


@pytest.mark.asyncio
async def test_gemini_live_stt_returns_transcript_from_live_responses():
    fake_session = _FakeSession([_FakeResponse("рус"), _FakeResponse(" тили")])
    fake_client = MagicMock()
    fake_client.aio.live.connect.return_value = fake_session

    with patch("pipeline.providers.gemini_live_stt.genai.Client", return_value=fake_client):
        provider = GeminiLiveSTT(language_codes=["ru", "uz"])

    event = await provider._recognize_impl(b"\x01\x00" * 4000)
    text = event.alternatives[0].text
    assert "рус" in text
    assert "тили" in text
    assert fake_session.sent[0]["end_of_turn"] is True


@pytest.mark.asyncio
async def test_gemini_live_stt_falls_back_on_transcribe_exception():
    fallback_event = stt.SpeechEvent(
        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
        alternatives=[stt.SpeechData(text="fallback", confidence=1.0, language="ru")],
    )
    fallback = MagicMock()
    fallback._recognize_impl = AsyncMock(return_value=fallback_event)

    with patch("pipeline.providers.gemini_live_stt.genai.Client", return_value=MagicMock()):
        provider = GeminiLiveSTT(language_codes=["ru", "uz"], fallback_stt=fallback)

    with patch.object(provider, "_transcribe_pcm", new=AsyncMock(side_effect=RuntimeError("boom"))):
        event = await provider._recognize_impl(b"\x01\x00" * 3000, language="ru")

    assert event.alternatives[0].text == "fallback"
    fallback._recognize_impl.assert_awaited_once()


@pytest.mark.asyncio
async def test_gemini_live_stt_retries_and_recovers(monkeypatch):
    with patch("pipeline.providers.gemini_live_stt.genai.Client", return_value=MagicMock()):
        provider = GeminiLiveSTT(language_codes=["ru", "uz"])

    calls = {"n": 0}

    async def _fake_transcribe(pcm_bytes: bytes, *, model: str):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient live ws error")
        return "русский"

    monkeypatch.setattr(provider, "_transcribe_pcm", _fake_transcribe)
    event = await provider._recognize_impl(b"\x01\x00" * 3000, language="ru")
    assert event.alternatives[0].text == "русский"
    assert calls["n"] >= 2
