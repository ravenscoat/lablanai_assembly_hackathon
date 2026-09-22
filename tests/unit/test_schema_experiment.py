from __future__ import annotations

import pytest
from pydantic import ValidationError

from config.schema import ExperimentBlock, TenantConfig


def test_experiment_block_accepts_well_formed_input():
    block = ExperimentBlock(id="youth_prompt_v2", variant="a", weight=50)
    assert block.id == "youth_prompt_v2"
    assert block.variant == "a"
    assert block.weight == 50


def test_experiment_block_rejects_zero_weight():
    with pytest.raises(ValidationError):
        ExperimentBlock(id="exp", variant="a", weight=0)


def test_experiment_block_rejects_negative_weight():
    with pytest.raises(ValidationError):
        ExperimentBlock(id="exp", variant="a", weight=-1)


def test_experiment_block_rejects_invalid_id_pattern():
    with pytest.raises(ValidationError):
        ExperimentBlock(id="Has Spaces", variant="a", weight=50)


def test_experiment_block_rejects_invalid_variant_pattern():
    with pytest.raises(ValidationError):
        ExperimentBlock(id="exp", variant="A!", weight=50)


def test_tenant_config_experiment_field_defaults_to_none():
    config = TenantConfig.model_validate({"tenant": {"id": "t1"}})
    assert config.experiment is None


def test_tenant_config_experiment_field_parses_when_present():
    config = TenantConfig.model_validate(
        {
            "tenant": {"id": "t1"},
            "experiment": {"id": "exp", "variant": "a", "weight": 50},
        }
    )
    assert config.experiment is not None
    assert config.experiment.variant == "a"
