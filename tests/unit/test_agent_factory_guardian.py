"""Tests for guardian-block rendering in the system prompt (NAV-150/153/154)."""

from __future__ import annotations

from unittest.mock import MagicMock

from config.schema import TenantConfig


def _make_config(**overrides):
    base = {
        "tenant": {"id": "test", "slug": "test", "name": "Test"},
        "transfer": {"enabled": True},
    }
    base.update(overrides)
    return TenantConfig.model_validate(base)


def _build_instructions(config, language="uz"):
    """Invoke the factory's _build_instructions helper in isolation.

    AgentFactory requires a tool_registry; a MagicMock is sufficient for
    _build_instructions which never touches the registry.
    """
    from agents.factory import AgentFactory

    factory = AgentFactory(tool_registry=MagicMock())
    return factory._build_instructions(config, language=language, skip_language_prompt=True)


def test_guardian_operator_block_rendered_when_set():
    config = _make_config(
        transfer={
            "enabled": True,
            "guardian": {"operator_instructions": "GUARDIAN_OPERATOR_MARKER: ask first, try KB."},
        }
    )
    prompt = _build_instructions(config)
    assert "GUARDIAN_OPERATOR_MARKER" in prompt


def test_guardian_operator_block_absent_when_null():
    config = _make_config(transfer={"enabled": True, "guardian": {"operator_instructions": None}})
    prompt = _build_instructions(config)
    assert "GUARDIAN_OPERATOR_MARKER" not in prompt


def test_guardian_operator_block_absent_when_guardian_null():
    config = _make_config(transfer={"enabled": True})
    prompt = _build_instructions(config)
    # No crash, no guardian section.
    assert "GUARDIAN_OPERATOR_MARKER" not in prompt


def test_guardian_psychological_block_rendered_when_set():
    config = _make_config(
        transfer={
            "enabled": True,
            "guardian": {
                "operator_instructions": "op",
                "psychological_instructions": "PSYCH_MARKER: reply empathetically first.",
            },
        }
    )
    prompt = _build_instructions(config)
    assert "PSYCH_MARKER" in prompt


def test_guardian_psychological_block_absent_when_null():
    config = _make_config(
        transfer={
            "enabled": True,
            "guardian": {
                "operator_instructions": "op",
                "psychological_instructions": None,
            },
        }
    )
    prompt = _build_instructions(config)
    assert "PSYCH_MARKER" not in prompt
