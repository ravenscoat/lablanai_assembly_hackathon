"""
Experiment cohort dataclasses and weighted-random variant assignment.

A cohort is a group of TenantConfigs that share tenant identity (id, slug,
name, phone_numbers) and an experiment id. At runtime, when a SIP call lands
on one of the shared phones, `assign_variant` performs a weighted-random
pick over the cohort and the chosen variant's TenantConfig drives the call.

Cohort construction and identity validation live in `src/config/loader.py`.
This module is pure logic with no I/O.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .schema import TenantConfig


@dataclass(frozen=True)
class CohortMember:
    config: TenantConfig
    weight: float
    variant_name: str


@dataclass(frozen=True)
class ExperimentCohort:
    experiment_id: str
    members: tuple[CohortMember, ...]


def assign_variant(
    cohort: ExperimentCohort,
    rng: random.Random | None = None,
) -> CohortMember:
    """Weighted random pick from a cohort. Always returns one member.

    `rng` is injectable for deterministic tests; production passes None and
    uses module-level `random`.
    """
    chooser = rng if rng is not None else random
    [chosen] = chooser.choices(
        cohort.members,
        weights=[m.weight for m in cohort.members],
        k=1,
    )
    return chosen
