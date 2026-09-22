from __future__ import annotations

from types import SimpleNamespace

import pytest

from config.routing import resolve_tenant_for_call


@pytest.mark.asyncio
async def test_resolve_tenant_for_call_strict_blocks_missing_phone(monkeypatch):
    monkeypatch.setenv("MULTI_TENANT_STRICT_ROUTING", "true")
    monkeypatch.setenv("SINGLE_TENANT_MODE", "false")

    registry = SimpleNamespace()
    config, experiment_meta, reason = await resolve_tenant_for_call(
        called_phone=None, tenant_registry=registry
    )

    assert config is None
    assert experiment_meta is None
    assert reason == "missing_called_phone"


@pytest.mark.asyncio
async def test_resolve_tenant_for_call_allows_single_tenant_default_fallback(monkeypatch):
    monkeypatch.setenv("MULTI_TENANT_STRICT_ROUTING", "false")
    monkeypatch.setenv("SINGLE_TENANT_MODE", "true")

    class _Registry:
        async def resolve_with_experiment(self, phone):
            return None, None

        def get_default(self):
            return {"tenant": "default"}

    registry = _Registry()
    config, experiment_meta, reason = await resolve_tenant_for_call(
        called_phone="+923700000001",
        tenant_registry=registry,
    )

    assert config == {"tenant": "default"}
    assert experiment_meta is None
    assert reason is None


@pytest.mark.asyncio
async def test_resolve_tenant_for_call_returns_experiment_meta(monkeypatch):
    monkeypatch.setenv("MULTI_TENANT_STRICT_ROUTING", "true")
    monkeypatch.setenv("SINGLE_TENANT_MODE", "false")

    class _Registry:
        async def resolve_with_experiment(self, phone):
            return {"tenant": "youth-agency"}, {
                "experiment_id": "exp",
                "variant": "b",
            }

    registry = _Registry()
    config, experiment_meta, reason = await resolve_tenant_for_call(
        called_phone="+923900000001",
        tenant_registry=registry,
    )

    assert config == {"tenant": "youth-agency"}
    assert experiment_meta == {"experiment_id": "exp", "variant": "b"}
    assert reason is None
