"""Unit tests for CustomSTT request shape.

The custom STT server auto-detects language; the previously-sent `languages`
form field was redundant and the server rejected anything other than short
ISO codes (uz/ru/en) with HTTP 400 — even when the upstream caller passed
a full locale like "uz-UZ". These tests pin the request shape so the field
cannot accidentally be re-introduced.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

from pipeline.providers.custom_stt import CustomSTT


class _FakeResponse:
    def __init__(self, status: int, payload):
        self.status = status
        self._payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def json(self):
        return self._payload

    async def text(self):
        return str(self._payload)


def _make_session(response: _FakeResponse) -> MagicMock:
    session = MagicMock()
    session.post = MagicMock(return_value=response)
    return session


def test_transcribe_does_not_send_languages_field():
    """Regression: the server auto-detects, the legacy `languages` field must not be sent."""
    stt_obj = CustomSTT(base_url="http://test:8080", language="uz-UZ")

    class _CaptureFormData:
        def __init__(self):
            self._fields = []

        def add_field(self, name, value, **kwargs):
            self._fields.append((name, value))

        def __iter__(self):
            return iter(self._fields)

    fake_form = _CaptureFormData()

    fake_response = _FakeResponse(200, [{"transcription": "salom", "language": "uz"}])
    fake_session = _make_session(fake_response)

    with (
        patch.object(stt_obj, "_ensure_session", return_value=fake_session),
        patch("pipeline.providers.custom_stt.aiohttp.FormData", return_value=fake_form),
    ):
        result = asyncio.run(stt_obj._transcribe(b"FAKEWAVDATA", language="uz-UZ"))

    field_names = [name for name, _ in fake_form]
    assert (
        "languages" not in field_names
    ), f"`languages` field must not be sent; got fields: {field_names}"
    assert "audio_files" in field_names
    assert result == "salom"


def test_transcribe_handles_empty_response():
    stt_obj = CustomSTT(base_url="http://test:8080", language="uz")

    fake_response = _FakeResponse(200, [{"transcription": "", "language": "uz"}])
    fake_session = _make_session(fake_response)

    with patch.object(stt_obj, "_ensure_session", return_value=fake_session):
        result = asyncio.run(stt_obj._transcribe(b"FAKEWAVDATA"))

    assert result == ""


def test_transcribe_propagates_non_200_as_empty():
    stt_obj = CustomSTT(base_url="http://test:8080", language="uz")

    fake_response = _FakeResponse(500, "boom")
    fake_session = _make_session(fake_response)

    with patch.object(stt_obj, "_ensure_session", return_value=fake_session):
        result = asyncio.run(stt_obj._transcribe(b"FAKEWAVDATA"))

    assert result == ""
