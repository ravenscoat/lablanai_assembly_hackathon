from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from config.loader import ConfigLoader
from config.registry import TenantRegistry
from config.routing import resolve_tenant_for_call

_DEFAULTS = """\
tenant:
  id: default
  slug: default
"""

_VARIANT_TEMPLATE = """\
tenant:
  id: youth-agency-id
  slug: youth-agency
  name: Yoshlar Agentligi
  phone_numbers: ["+923900000001"]
experiment:
  id: youth_prompt_v2
  variant: {variant}
  weight: {weight}
personality:
  greeting: "Variant {variant} greeting"
"""


@pytest.fixture
def fixture_dir(tmp_path: Path) -> Path:
    (tmp_path / "_defaults.yaml").write_text(_DEFAULTS)
    (tmp_path / "youth-agency-a.yaml").write_text(_VARIANT_TEMPLATE.format(variant="a", weight=50))
    (tmp_path / "youth-agency-b.yaml").write_text(_VARIANT_TEMPLATE.format(variant="b", weight=50))
    return tmp_path


@pytest.mark.asyncio
async def test_routing_returns_variant_a_when_rng_picks_first(fixture_dir, monkeypatch):
    monkeypatch.setenv("MULTI_TENANT_STRICT_ROUTING", "true")
    monkeypatch.setenv("SINGLE_TENANT_MODE", "false")

    loader = ConfigLoader(config_dir=fixture_dir)
    registry = TenantRegistry(loader=loader)

    # Force assign_variant deterministic: patch the module-level random.choices
    # that assign_variant uses when no rng is passed.
    with patch("config.experiments.random.choices") as mock_choices:
        mock_choices.side_effect = lambda members, weights, k: [members[0]]
        config, experiment_meta, reason = await resolve_tenant_for_call(
            called_phone="+923900000001",
            tenant_registry=registry,
        )

    assert reason is None
    assert config.personality.greeting == "Variant a greeting"
    assert experiment_meta == {"experiment_id": "youth_prompt_v2", "variant": "a"}


@pytest.mark.asyncio
async def test_routing_returns_variant_b_when_rng_picks_second(fixture_dir, monkeypatch):
    monkeypatch.setenv("MULTI_TENANT_STRICT_ROUTING", "true")
    monkeypatch.setenv("SINGLE_TENANT_MODE", "false")

    loader = ConfigLoader(config_dir=fixture_dir)
    registry = TenantRegistry(loader=loader)

    with patch("config.experiments.random.choices") as mock_choices:
        mock_choices.side_effect = lambda members, weights, k: [members[1]]
        config, experiment_meta, reason = await resolve_tenant_for_call(
            called_phone="+923900000001",
            tenant_registry=registry,
        )

    assert reason is None
    assert config.personality.greeting == "Variant b greeting"
    assert experiment_meta == {"experiment_id": "youth_prompt_v2", "variant": "b"}
