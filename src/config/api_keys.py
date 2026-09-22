"""
Tenant-aware API key resolution.

Resolves per-tenant Platform API keys for strict multi-tenant isolation.

Generic, data-driven resolver: a tenant's key lives in the environment under
``{SLUG_UPPER}_AGENT_API_KEY`` where ``SLUG_UPPER`` is the tenant slug
uppercased with every non-alphanumeric character replaced by ``_`` (so slug
``example-tenant`` → ``EXAMPLE_TENANT_AGENT_API_KEY``). Onboarding a tenant is
therefore an env/config change, not a code edit — no per-tenant branches.

Resolution order for a tenant-scoped key:
  1. explicit value passed by the caller (if any)
  2. ``{SLUG_UPPER}_AGENT_API_KEY``
  3. legacy fallback (only when ENABLE_LEGACY_INTERNAL_API_KEY_FALLBACK is on)
"""

from __future__ import annotations

import os
import re

_TRUE_VALUES = {"1", "true", "yes", "on"}

_NON_ALNUM_RE = re.compile(r"[^A-Za-z0-9]+")


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in _TRUE_VALUES


def _normalize_phone(phone: str | None) -> str:
    if not phone:
        return ""
    return phone.strip().replace(" ", "").replace("-", "")


def slug_to_env_var(slug: str) -> str:
    """Map a tenant slug to its per-tenant API key env var name.

    ``example-tenant`` -> ``EXAMPLE_TENANT_AGENT_API_KEY``
    Returns an empty string for an empty/blank slug.
    """
    cleaned = _NON_ALNUM_RE.sub("_", (slug or "").strip()).strip("_").upper()
    if not cleaned:
        return ""
    return f"{cleaned}_AGENT_API_KEY"


def _legacy_fallback_key() -> str:
    if not _env_flag("ENABLE_LEGACY_INTERNAL_API_KEY_FALLBACK", False):
        return ""
    return (
        os.getenv("AGENT_API_KEY")
        or os.getenv("AGENT_SERVICE_API_KEY")
        or os.getenv("INTERNAL_API_KEY")
        or ""
    )


def resolve_default_api_key(explicit_api_key: str | None = None) -> str:
    """Resolve default API key for non-tenant-scoped requests."""
    if explicit_api_key:
        return explicit_api_key
    return (
        os.getenv("AGENT_API_KEY")
        or os.getenv("AGENT_SERVICE_API_KEY")
        or os.getenv("INTERNAL_API_KEY")
        or ""
    )


def resolve_tenant_api_key(tenant_hint: str | None) -> str:
    """
    Resolve API key by tenant identifier (slug).

    Looks up ``{SLUG_UPPER}_AGENT_API_KEY`` from the environment. Returns the
    legacy fallback (when enabled) if the slug is blank or the env var is
    unset/empty. Returns an empty string when no allowed key is available.
    """
    hint = (tenant_hint or "").strip()
    if not hint:
        return _legacy_fallback_key()
    env_var = slug_to_env_var(hint)
    if env_var:
        key = os.getenv(env_var, "").strip()
        if key:
            return key
    return _legacy_fallback_key()


def resolve_tenant_api_key_for_phone(phone: str | None) -> str:
    """
    Resolve API key by called phone number.

    Used during pre-tenant API config lookup, BEFORE the tenant slug is known
    (so a phone→slug map is not yet available here). With the data-driven
    resolver there is no static phone→key table; this falls back to the
    default/legacy key. Once the tenant is resolved, callers should use
    ``resolve_tenant_api_key(slug)`` for the scoped key.
    """
    normalized = _normalize_phone(phone)
    if not normalized:
        return _legacy_fallback_key()
    return _legacy_fallback_key()
