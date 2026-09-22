"""Schema validation for the new goals / summary_instructions fields."""

from __future__ import annotations

import pytest
from pydantic import ValidationError


def test_goal_config_parses_minimal_input():
    from config.schema import GoalConfig

    g = GoalConfig.model_validate({"key": "problem_specifics"})
    assert g.key == "problem_specifics"
    assert g.description == {}
    assert g.required is True


def test_goal_config_parses_full_input():
    from config.schema import GoalConfig

    g = GoalConfig.model_validate(
        {
            "key": "tried_so_far",
            "description": {"uz": "Sinovlar", "ru": "Попытки"},
            "required": False,
        }
    )
    assert g.description["uz"] == "Sinovlar"
    assert g.description["ru"] == "Попытки"
    assert g.required is False


def test_goal_config_rejects_missing_key():
    from config.schema import GoalConfig

    with pytest.raises(ValidationError):
        GoalConfig.model_validate({"description": {"uz": "x"}})


def test_sub_agent_config_accepts_goals_and_summary_instructions():
    from config.schema import SubAgentConfig

    cfg = SubAgentConfig.model_validate(
        {
            "type": "appeal",
            "goals": [
                {"key": "problem_specifics", "description": {"uz": "Tafsilot"}},
                {"key": "tried_so_far", "description": {"uz": "Sinov"}, "required": False},
            ],
            "summary_instructions": {"uz": "2-3 gap"},
        }
    )
    assert len(cfg.goals) == 2
    assert cfg.goals[0].key == "problem_specifics"
    assert cfg.goals[1].required is False
    assert cfg.summary_instructions["uz"] == "2-3 gap"


def test_sub_agent_config_defaults_preserved_for_legacy_tenants():
    from config.schema import SubAgentConfig

    cfg = SubAgentConfig.model_validate({"type": "appeal"})
    assert cfg.goals == []
    assert cfg.summary_instructions == {}
