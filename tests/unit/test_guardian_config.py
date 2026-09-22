"""Tests for GuardianConfig schema (NAV-150/153/154)."""

from __future__ import annotations

from config.schema import GuardianConfig, TenantConfig, TransferConfig


def test_guardian_config_defaults_to_none_for_both_fields():
    g = GuardianConfig()
    assert g.operator_instructions is None
    assert g.psychological_instructions is None


def test_guardian_config_accepts_strings():
    g = GuardianConfig(
        operator_instructions="Ask caller what they need first.",
        psychological_instructions="Respond empathetically before offering psychologist.",
    )
    assert g.operator_instructions == "Ask caller what they need first."
    assert g.psychological_instructions.startswith("Respond")


def test_transfer_config_guardian_defaults_to_none():
    t = TransferConfig()
    assert t.guardian is None


def test_tenant_config_defaults_keep_guardian_none():
    cfg = TenantConfig.model_validate({"tenant": {"id": "x", "slug": "x", "name": "X"}})
    assert cfg.transfer.guardian is None


def test_tenant_config_accepts_guardian_block_in_transfer():
    cfg = TenantConfig.model_validate(
        {
            "tenant": {"id": "x", "slug": "x", "name": "X"},
            "transfer": {
                "enabled": True,
                "guardian": {
                    "operator_instructions": "try kb first",
                    "psychological_instructions": "be gentle",
                },
            },
        }
    )
    assert cfg.transfer.guardian is not None
    assert cfg.transfer.guardian.operator_instructions == "try kb first"
    assert cfg.transfer.guardian.psychological_instructions == "be gentle"
