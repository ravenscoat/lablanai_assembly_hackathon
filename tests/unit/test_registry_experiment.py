from __future__ import annotations

from pathlib import Path

import pytest

from config.loader import ConfigLoader
from config.registry import TenantRegistry

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
"""


@pytest.mark.asyncio
async def test_registry_resolve_with_experiment_proxies_loader(tmp_path: Path):
    (tmp_path / "_defaults.yaml").write_text(_DEFAULTS)
    (tmp_path / "youth-agency-a.yaml").write_text(_VARIANT_TEMPLATE.format(variant="a", weight=50))
    (tmp_path / "youth-agency-b.yaml").write_text(_VARIANT_TEMPLATE.format(variant="b", weight=50))

    loader = ConfigLoader(config_dir=tmp_path)
    registry = TenantRegistry(loader=loader)

    config, meta = await registry.resolve_with_experiment("+923900000001")

    assert config is not None
    assert meta is not None
    assert meta["experiment_id"] == "youth_prompt_v2"
    assert meta["variant"] in {"a", "b"}
