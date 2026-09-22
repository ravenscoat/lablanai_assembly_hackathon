"""
Routing policy for tenant resolution.

This module centralizes strict multi-tenant routing behavior so unresolved
called-phone lookups fail closed by default.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_TRUE_VALUES = {"1", "true", "yes", "on"}
_BLOCKED_ROUTING_COUNT = 0


def env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in _TRUE_VALUES


def blocked_routing_count() -> int:
    return _BLOCKED_ROUTING_COUNT


def _log_routing_block(
    *,
    reason: str,
    called_phone: str | None,
    strict_routing: bool,
    single_tenant_mode: bool,
) -> None:
    global _BLOCKED_ROUTING_COUNT
    _BLOCKED_ROUTING_COUNT += 1
    logger.error(
        "[tenant_routing_blocked] reason=%s called_phone=%s strict_routing=%s "
        "single_tenant_mode=%s blocked_count=%s",
        reason,
        called_phone or "none",
        strict_routing,
        single_tenant_mode,
        _BLOCKED_ROUTING_COUNT,
    )


async def resolve_tenant_for_call(
    *,
    called_phone: str | None,
    tenant_registry: Any,
) -> tuple[Any | None, dict | None, str | None]:
    """
    Resolve tenant config with safe routing behavior.

    Returns:
      (config, experiment_meta, None) on success
      (None, None, reason) when routing is blocked
    """
    strict_routing = env_flag("MULTI_TENANT_STRICT_ROUTING", True)
    single_tenant_mode = env_flag("SINGLE_TENANT_MODE", False)

    if called_phone:
        config, experiment_meta = await tenant_registry.resolve_with_experiment(called_phone)
        if config:
            return config, experiment_meta, None

        if single_tenant_mode and not strict_routing:
            logger.warning(
                "[tenant_routing_fallback] reason=unknown_called_phone called_phone=%s "
                "strict_routing=%s single_tenant_mode=%s action=default_tenant",
                called_phone,
                strict_routing,
                single_tenant_mode,
            )
            return tenant_registry.get_default(), None, None

        reason = "unknown_called_phone"
        _log_routing_block(
            reason=reason,
            called_phone=called_phone,
            strict_routing=strict_routing,
            single_tenant_mode=single_tenant_mode,
        )
        return None, None, reason

    # No called_phone was extracted from metadata.
    if single_tenant_mode and not strict_routing:
        logger.warning(
            "[tenant_routing_fallback] reason=missing_called_phone called_phone=none "
            "strict_routing=%s single_tenant_mode=%s action=default_tenant",
            strict_routing,
            single_tenant_mode,
        )
        return tenant_registry.get_default(), None, None

    reason = "missing_called_phone"
    _log_routing_block(
        reason=reason,
        called_phone=called_phone,
        strict_routing=strict_routing,
        single_tenant_mode=single_tenant_mode,
    )
    return None, None, reason
