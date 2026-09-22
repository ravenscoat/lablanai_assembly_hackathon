from __future__ import annotations

import pytest

from observability.langfuse_setup import setup_langfuse


def test_langfuse_is_disabled_without_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "LANGFUSE_ENABLED",
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_HOST",
        "LANGFUSE_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)

    assert setup_langfuse(metadata={"tenant": "yoshlar"}) is None


def test_langfuse_enabled_requires_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_ENABLED", "true")
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_HOST", raising=False)
    monkeypatch.delenv("LANGFUSE_BASE_URL", raising=False)

    with pytest.raises(RuntimeError, match="missing"):
        setup_langfuse(metadata={"tenant": "yoshlar"})


def test_langfuse_rejects_invalid_enabled_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_ENABLED", "maybe")

    with pytest.raises(ValueError, match="LANGFUSE_ENABLED"):
        setup_langfuse()
