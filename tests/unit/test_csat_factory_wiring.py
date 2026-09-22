"""Factory wires CSAT into every tenant (NAV-156).

Covers:
- The CSAT policy block is injected into _build_instructions for both UZ and RU.
- _resolve_tools auto-includes collect_csat even when the YAML doesn't list it.
- The rule mentions the transferred-call exclusion so the LLM does not collect
  CSAT before escalating to a human.
"""

from __future__ import annotations

from agents.factory import AgentFactory
from config.schema import LanguagesConfig, TenantConfig, TenantIdentity
from tools.registry import ToolRegistry


def _uz_config() -> TenantConfig:
    return TenantConfig(tenant=TenantIdentity(id="t1", slug="any"))


def _ru_config() -> TenantConfig:
    return TenantConfig(
        tenant=TenantIdentity(id="t2", slug="paynet"),
        languages=LanguagesConfig(default="ru", available=["ru", "uz"], ask_on_start=True),
    )


def test_csat_policy_uzbek_block_present():
    text = AgentFactory._csat_policy_instructions("uz")
    assert "CSAT QOIDASI" in text
    assert "1 dan 5 gacha baholang" in text
    assert "collect_csat" in text
    assert "end_call" in text
    # Transferred-call exclusion is part of the contract.
    assert "request_escalation" in text


def test_csat_policy_russian_block_present():
    text = AgentFactory._csat_policy_instructions("ru")
    assert "ПРАВИЛО CSAT" in text
    assert "от 1 до 5" in text
    assert "collect_csat" in text
    assert "end_call" in text
    assert "request_escalation" in text


def test_csat_policy_unknown_language_falls_back_to_uzbek():
    text = AgentFactory._csat_policy_instructions("tr")
    assert "CSAT QOIDASI" in text


def test_build_instructions_includes_csat_block_for_uz_tenant():
    factory = AgentFactory(tool_registry=ToolRegistry())
    text = factory._build_instructions(_uz_config(), language="uz")
    assert "CSAT QOIDASI" in text


def test_build_instructions_includes_csat_block_for_ru_tenant():
    factory = AgentFactory(tool_registry=ToolRegistry())
    text = factory._build_instructions(_ru_config(), language="ru")
    assert "ПРАВИЛО CSAT" in text


def test_resolve_tools_auto_includes_collect_csat_when_yaml_omits_it():
    factory = AgentFactory(tool_registry=ToolRegistry())
    tools = factory._resolve_tools(_uz_config())
    tool_names = [getattr(t, "name", None) or getattr(t, "__name__", "") for t in tools]
    assert any("collect_csat" in n for n in tool_names)


def test_resolve_tools_does_not_duplicate_collect_csat_when_yaml_lists_it():
    factory = AgentFactory(tool_registry=ToolRegistry())
    cfg = TenantConfig(
        tenant=TenantIdentity(id="t3"),
        tools={"platform": ["collect_csat"], "tenant": []},
    )
    tools = factory._resolve_tools(cfg)
    tool_names = [getattr(t, "name", None) or getattr(t, "__name__", "") for t in tools]
    csat_count = sum(1 for n in tool_names if "collect_csat" in n)
    assert csat_count == 1
