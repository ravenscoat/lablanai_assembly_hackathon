from __future__ import annotations

from pathlib import Path

import pytest

from config.loader import ConfigLoader

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


def _write_defaults(tmp_path: Path) -> None:
    (tmp_path / "_defaults.yaml").write_text(_DEFAULTS)


def _write_variant(tmp_path: Path, name: str, *, variant: str, weight: int) -> None:
    (tmp_path / f"{name}.yaml").write_text(_VARIANT_TEMPLATE.format(variant=variant, weight=weight))


def test_two_variant_cohort_loads_and_indexes_phone(tmp_path: Path):
    _write_defaults(tmp_path)
    _write_variant(tmp_path, "youth-agency-a", variant="a", weight=50)
    _write_variant(tmp_path, "youth-agency-b", variant="b", weight=50)

    loader = ConfigLoader(config_dir=tmp_path)

    assert "+923900000001" in loader._experiments_by_phone
    cohort = loader._experiments_by_phone["+923900000001"]
    assert cohort.experiment_id == "youth_prompt_v2"
    assert {m.variant_name for m in cohort.members} == {"a", "b"}
    assert "+923900000001" not in loader._configs_by_phone


def test_incomplete_cohort_raises(tmp_path: Path):
    _write_defaults(tmp_path)
    _write_variant(tmp_path, "youth-agency-a", variant="a", weight=50)

    with pytest.raises(ValueError, match="incomplete experiment cohort"):
        ConfigLoader(config_dir=tmp_path)


def test_drifted_tenant_name_raises(tmp_path: Path):
    _write_defaults(tmp_path)
    _write_variant(tmp_path, "youth-agency-a", variant="a", weight=50)
    # B has a drifted name:
    (tmp_path / "youth-agency-b.yaml").write_text("""\
tenant:
  id: youth-agency-id
  slug: youth-agency
  name: DIFFERENT NAME
  phone_numbers: ["+923900000001"]
experiment:
  id: youth_prompt_v2
  variant: b
  weight: 50
""")

    with pytest.raises(ValueError, match="disagree on tenant.name"):
        ConfigLoader(config_dir=tmp_path)


def test_drifted_phone_numbers_raises(tmp_path: Path):
    _write_defaults(tmp_path)
    _write_variant(tmp_path, "youth-agency-a", variant="a", weight=50)
    (tmp_path / "youth-agency-b.yaml").write_text("""\
tenant:
  id: youth-agency-id
  slug: youth-agency
  name: Yoshlar Agentligi
  phone_numbers: ["+923900000002"]
experiment:
  id: youth_prompt_v2
  variant: b
  weight: 50
""")

    with pytest.raises(ValueError, match="disagree on tenant.phone_numbers"):
        ConfigLoader(config_dir=tmp_path)


def test_duplicate_variant_name_raises(tmp_path: Path):
    _write_defaults(tmp_path)
    _write_variant(tmp_path, "youth-agency-a1", variant="a", weight=50)
    _write_variant(tmp_path, "youth-agency-a2", variant="a", weight=50)

    with pytest.raises(ValueError, match="duplicate variant name"):
        ConfigLoader(config_dir=tmp_path)


def test_phone_collision_with_non_experiment_tenant_raises(tmp_path: Path):
    _write_defaults(tmp_path)
    _write_variant(tmp_path, "youth-agency-a", variant="a", weight=50)
    _write_variant(tmp_path, "youth-agency-b", variant="b", weight=50)
    # Non-experiment tenant claiming the same phone:
    (tmp_path / "other-tenant.yaml").write_text("""\
tenant:
  id: other
  slug: other
  phone_numbers: ["+923900000001"]
""")

    with pytest.raises(ValueError, match="claimed by both"):
        ConfigLoader(config_dir=tmp_path)


def test_non_experiment_tenant_still_indexed_by_phone(tmp_path: Path):
    """Regression guard: tenants without an experiment block still route via
    _configs_by_phone exactly as before."""
    _write_defaults(tmp_path)
    (tmp_path / "paynet.yaml").write_text("""\
tenant:
  id: paynet
  slug: paynet
  phone_numbers: ["+923700000007"]
""")

    loader = ConfigLoader(config_dir=tmp_path)

    assert "+923700000007" in loader._configs_by_phone
    assert loader._configs_by_phone["+923700000007"].tenant.slug == "paynet"
    assert loader._experiments_by_phone == {}


@pytest.mark.asyncio
async def test_resolve_with_experiment_returns_variant_meta(tmp_path: Path):
    _write_defaults(tmp_path)
    _write_variant(tmp_path, "youth-agency-a", variant="a", weight=50)
    _write_variant(tmp_path, "youth-agency-b", variant="b", weight=50)

    loader = ConfigLoader(config_dir=tmp_path)

    config, meta = await loader.resolve_with_experiment("+923900000001")

    assert config is not None
    assert meta is not None
    assert meta["experiment_id"] == "youth_prompt_v2"
    assert meta["variant"] in {"a", "b"}


@pytest.mark.asyncio
async def test_resolve_with_experiment_returns_none_meta_for_non_experiment(
    tmp_path: Path,
):
    _write_defaults(tmp_path)
    (tmp_path / "paynet.yaml").write_text("""\
tenant:
  id: paynet
  slug: paynet
  phone_numbers: ["+923700000007"]
""")

    loader = ConfigLoader(config_dir=tmp_path)
    config, meta = await loader.resolve_with_experiment("+923700000007")

    assert config is not None
    assert config.tenant.slug == "paynet"
    assert meta is None


@pytest.mark.asyncio
async def test_resolve_with_experiment_returns_none_for_unknown_phone(tmp_path: Path):
    _write_defaults(tmp_path)
    loader = ConfigLoader(config_dir=tmp_path)

    config, meta = await loader.resolve_with_experiment("+923000000000")

    assert config is None
    assert meta is None


def test_reload_failure_restores_previous_state(tmp_path: Path):
    """If reload encounters a malformed cohort, the registry is rolled back
    to the previous valid state instead of being left empty."""
    _write_defaults(tmp_path)
    _write_variant(tmp_path, "youth-agency-a", variant="a", weight=50)
    _write_variant(tmp_path, "youth-agency-b", variant="b", weight=50)

    loader = ConfigLoader(config_dir=tmp_path)

    # Snapshot the valid state for later comparison.
    saved_slug_keys = set(loader._configs_by_slug.keys())
    cohort_phone_keys = set(loader._experiments_by_phone.keys())
    assert "+923900000001" in cohort_phone_keys

    # Break the cohort on disk: rewrite B with a drifted tenant.name so
    # the reload's _group_experiments raises.
    (tmp_path / "youth-agency-b.yaml").write_text("""\
tenant:
  id: youth-agency-id
  slug: youth-agency
  name: DRIFTED NAME
  phone_numbers: ["+923900000001"]
experiment:
  id: youth_prompt_v2
  variant: b
  weight: 50
""")

    with pytest.raises(ValueError, match="disagree on tenant.name"):
        loader.reload()

    # Registry must still hold the previous valid state — no silent wipe.
    assert set(loader._configs_by_slug.keys()) == saved_slug_keys
    assert set(loader._experiments_by_phone.keys()) == cohort_phone_keys
    assert "+923900000001" in loader._experiments_by_phone
