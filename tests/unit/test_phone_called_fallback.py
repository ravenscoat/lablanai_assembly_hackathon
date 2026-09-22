from __future__ import annotations

from types import SimpleNamespace

from utils.phone import extract_called_phone


def _ctx_no_metadata():
    return SimpleNamespace(
        job=None,
        room=SimpleNamespace(metadata=None),
    )


def test_extract_called_phone_skips_env_fallback_when_single_tenant_disabled(monkeypatch):
    monkeypatch.setenv("SINGLE_TENANT_MODE", "false")
    monkeypatch.setenv("AGENT_PHONE_NUMBER", "+923781225399")

    phone = extract_called_phone(_ctx_no_metadata())
    assert phone is None


def test_extract_called_phone_uses_env_fallback_when_single_tenant_enabled(monkeypatch):
    monkeypatch.setenv("SINGLE_TENANT_MODE", "true")
    monkeypatch.setenv("AGENT_PHONE_NUMBER", "+923781225399")

    phone = extract_called_phone(_ctx_no_metadata())
    assert phone == "+923781225399"
