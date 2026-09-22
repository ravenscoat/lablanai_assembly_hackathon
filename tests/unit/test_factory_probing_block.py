"""AgentFactory builds a probing-block system prompt when goals are declared."""

from __future__ import annotations

from config.schema import SubAgentConfig, TenantConfig


def _config_with_goals():
    return TenantConfig.model_validate(
        {
            "tenant": {"id": "ya-test", "slug": "yoshlar", "name": "YA"},
            "languages": {"default": "uz", "available": ["uz"]},
            "sub_agents": {
                "appeal": {
                    "type": "appeal",
                    "instructions": "Sen murojaat yig'uvchisan.",
                    "fields": [{"name": "full_name", "prompt": "Ism?"}],
                    "goals": [
                        {"key": "problem_specifics", "description": {"uz": "Tafsilot"}},
                        {"key": "desired_outcome", "description": {"uz": "Natija"}},
                    ],
                }
            },
        }
    )


def _config_without_goals():
    return TenantConfig.model_validate(
        {
            "tenant": {"id": "ya-test", "slug": "yoshlar", "name": "YA"},
            "languages": {"default": "uz", "available": ["uz"]},
            "sub_agents": {
                "appeal": {
                    "type": "appeal",
                    "instructions": "Sen murojaat yig'uvchisan.",
                    "fields": [{"name": "content", "prompt": "?"}],
                }
            },
        }
    )


def test_probing_block_appended_when_goals_present():
    from agents.factory import _build_probing_block

    cfg: SubAgentConfig = _config_with_goals().sub_agents["appeal"]
    block = _build_probing_block(cfg, language="uz")
    assert block is not None
    assert "problem_specifics" in block
    assert "desired_outcome" in block
    assert "Tafsilot" in block
    assert "Natija" in block
    # Instructs the LLM to call confirm_and_submit with a summary.
    assert "summary" in block.lower() or "xulosa" in block.lower()


def test_probing_block_none_when_no_goals():
    from agents.factory import _build_probing_block

    cfg = _config_without_goals().sub_agents["appeal"]
    assert _build_probing_block(cfg, language="uz") is None


def test_probing_block_uses_summary_instructions_override():
    from agents.factory import _build_probing_block

    cfg: SubAgentConfig = _config_with_goals().sub_agents["appeal"]
    # Inject a language-specific override
    cfg = cfg.model_copy(update={"summary_instructions": {"uz": "CUSTOM_GUIDANCE_MARKER"}})
    block = _build_probing_block(cfg, language="uz")
    assert "CUSTOM_GUIDANCE_MARKER" in block
