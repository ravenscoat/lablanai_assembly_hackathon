"""Explicit specialist routing and capability boundaries for RelayDesk."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SpecialistProfile:
    name: str
    purpose: str
    allowed_tools: frozenset[str]
    instructions: str


SPECIALISTS = {
    "front_desk": SpecialistProfile(
        name="front_desk",
        purpose="Identify the caller, open one durable case, and route it.",
        allowed_tools=frozenset({"identify_customer", "create_case", "route_case", "get_case"}),
        instructions=(
            "You are the RelayDesk front desk. Identify the customer, create one case, "
            "then route to the correct specialist. Do not diagnose or mutate billing or access."
        ),
    ),
    "billing": SpecialistProfile(
        name="billing",
        purpose="Investigate charges and safely resolve verified duplicate billing.",
        allowed_tools=frozenset({"inspect_billing", "refund_duplicate", "route_case", "get_case"}),
        instructions=(
            "You are the billing specialist. Read the shared case first. Inspect billing before "
            "any refund. Refund only a verified duplicate and report the persisted verification."
        ),
    ),
    "subscriptions": SpecialistProfile(
        name="subscriptions",
        purpose="Explain and investigate subscription state without inventing account data.",
        allowed_tools=frozenset({"inspect_billing", "route_case", "get_case"}),
        instructions=(
            "You are the subscription specialist. Read the shared case, inspect account evidence, "
            "and explain only verified subscription or charge state."
        ),
    ),
    "permissions": SpecialistProfile(
        name="permissions",
        purpose="Investigate project membership and restore verified access.",
        allowed_tools=frozenset(
            {"inspect_permissions", "restore_access", "route_case", "get_case"}
        ),
        instructions=(
            "You are the permissions specialist. Read the shared case. Inspect membership before "
            "restoring access and confirm the persisted active state afterward."
        ),
    ),
}


def specialist_context(name: str, handoff_packet: dict) -> str:
    profile = SPECIALISTS[name]
    return (
        f"ACTIVE SPECIALIST: {profile.name}. {profile.instructions}\n"
        f"SHARED CASE PACKET: {handoff_packet}\n"
        "Never ask the caller to repeat a fact already present in the packet."
    )
