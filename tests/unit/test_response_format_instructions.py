"""Tests for language-aware response-format instructions."""

from __future__ import annotations

from agents.factory import AgentFactory
from config.schema import LanguagesConfig, TenantConfig, TenantIdentity


def test_response_format_uzbek():
    text = AgentFactory._response_format_instructions("uz")
    assert "JAVOB FORMATI" in text
    assert "Raqamlarni so'z bilan" in text


def test_response_format_russian():
    text = AgentFactory._response_format_instructions("ru")
    assert "ФОРМАТ ОТВЕТА" in text
    assert "Произносите числа словами" in text


def test_response_format_unknown_falls_back_to_uzbek():
    text = AgentFactory._response_format_instructions("tr")
    assert "JAVOB FORMATI" in text


def _multi_config():
    return TenantConfig(
        tenant=TenantIdentity(id="t1", slug="paynet"),
        languages=LanguagesConfig(default="ru", available=["ru", "uz"], ask_on_start=True),
    )


def _config_with_system_prompt(prompt: str) -> TenantConfig:
    return TenantConfig(
        tenant=TenantIdentity(id="t2", slug="any"),
        transfer={"enabled": True},
        personality={"system_prompt": prompt},
    )


def test_build_instructions_includes_language_policy_when_multilingual():
    from agents.factory import AgentFactory
    from tools.registry import ToolRegistry

    factory = AgentFactory(tool_registry=ToolRegistry())
    text = factory._build_instructions(_multi_config(), language="ru")
    assert "MULTILINGUAL POLICY" in text
    assert "select_language" in text
    assert "ФОРМАТ ОТВЕТА" in text


def test_build_instructions_omits_multilingual_policy_when_skip_prompt():
    """Returning caller / post-handoff path."""
    from agents.factory import AgentFactory
    from tools.registry import ToolRegistry

    factory = AgentFactory(tool_registry=ToolRegistry())
    text = factory._build_instructions(
        _multi_config(),
        language="uz",
        skip_language_prompt=True,
    )
    assert "MULTILINGUAL POLICY" not in text
    assert "LANGUAGE POLICY" in text
    assert "select_language" in text


def test_build_instructions_omits_language_policy_for_single_lang():
    from agents.factory import AgentFactory
    from tools.registry import ToolRegistry

    factory = AgentFactory(tool_registry=ToolRegistry())
    config = TenantConfig(tenant=TenantIdentity(id="t1"))
    text = factory._build_instructions(config, language="uz")
    assert "MULTILINGUAL POLICY" not in text
    assert "LANGUAGE POLICY" not in text
    assert "JAVOB FORMATI" in text


def test_resolve_tools_auto_includes_select_language_for_multilingual():
    from agents.factory import AgentFactory
    from tools.registry import ToolRegistry

    factory = AgentFactory(tool_registry=ToolRegistry())
    tools = factory._resolve_tools(_multi_config())
    tool_names = [getattr(t, "name", None) or getattr(t, "__name__", "") for t in tools]
    assert any("select_language" in n for n in tool_names)


def test_build_instructions_includes_personality_system_prompt():
    """Tenant-supplied system_prompt is the canonical home for tenant-specific
    rules (phone-number, operator-scope, time-of-day, etc.). Any text the
    YAML provides should land in the final instructions verbatim."""
    from agents.factory import AgentFactory
    from tools.registry import ToolRegistry

    rules = (
        "## TELEFON RAQAMI QOIDASI:\n"
        "Agar foydalanuvchi raqamini aytib bermoqchi bo'lsa, "
        '"raqamingiz bizda bor, tashvishlanmang" deb ayting.\n'
        "## OPERATOR QOIDASI:\n"
        "Operator jadvali haqida foydalanuvchi so'ramaguncha gapirmang."
    )
    factory = AgentFactory(tool_registry=ToolRegistry())
    text = factory._build_instructions(_config_with_system_prompt(rules), language="uz")

    assert "TELEFON RAQAMI QOIDASI" in text
    assert "raqamingiz bizda bor, tashvishlanmang" in text
    assert "OPERATOR QOIDASI" in text


def test_build_instructions_no_hardcoded_yoshlar_branch():
    """Regression guard: factory must not inject yoshlar-specific text
    behind a `slug == "yoshlar"` check. Tenant rules live in YAML now."""
    from agents.factory import AgentFactory
    from tools.registry import ToolRegistry

    factory = AgentFactory(tool_registry=ToolRegistry())
    cfg = TenantConfig(
        tenant=TenantIdentity(id="t-yoshlar", slug="yoshlar"),
        transfer={"enabled": True},
    )  # note: no personality.system_prompt set
    text = factory._build_instructions(cfg, language="uz")

    assert "YOSHLAR UCHUN" not in text  # no slug-gated hardcoded block
