from __future__ import annotations

import random

from config.experiments import CohortMember, ExperimentCohort, assign_variant
from config.schema import TenantConfig


def _config(slug: str = "t1") -> TenantConfig:
    return TenantConfig.model_validate({"tenant": {"id": slug, "slug": slug}})


def test_assign_variant_returns_only_member_when_one_dominant():
    cohort = ExperimentCohort(
        experiment_id="exp",
        members=(
            CohortMember(config=_config("a"), weight=1.0, variant_name="a"),
            CohortMember(config=_config("b"), weight=0.0001, variant_name="b"),
        ),
    )
    rng = random.Random(0)
    counts = {"a": 0, "b": 0}
    for _ in range(1000):
        chosen = assign_variant(cohort, rng=rng)
        counts[chosen.variant_name] += 1
    assert counts["a"] > 990  # heavy weight should dominate


def test_assign_variant_distribution_matches_weights():
    cohort = ExperimentCohort(
        experiment_id="exp",
        members=(
            CohortMember(config=_config("a"), weight=90, variant_name="a"),
            CohortMember(config=_config("b"), weight=10, variant_name="b"),
        ),
    )
    rng = random.Random(42)
    counts = {"a": 0, "b": 0}
    for _ in range(10000):
        chosen = assign_variant(cohort, rng=rng)
        counts[chosen.variant_name] += 1
    ratio_b = counts["b"] / 10000
    assert 0.08 <= ratio_b <= 0.12  # within ±2% of the 10% target


def test_assign_variant_returns_cohort_member():
    member_a = CohortMember(config=_config("a"), weight=50, variant_name="a")
    cohort = ExperimentCohort(experiment_id="exp", members=(member_a,))
    rng = random.Random(0)
    chosen = assign_variant(cohort, rng=rng)
    assert chosen is member_a
