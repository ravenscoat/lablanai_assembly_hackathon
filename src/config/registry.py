"""
TenantRegistry: singleton that provides tenant config resolution.
"""

from __future__ import annotations

import logging

from .loader import ConfigLoader
from .schema import TenantConfig

logger = logging.getLogger(__name__)

_registry: TenantRegistry | None = None


class TenantRegistry:
    """Resolves phone numbers and slugs to TenantConfig."""

    def __init__(self, loader: ConfigLoader | None = None):
        self._loader = loader or ConfigLoader()

    async def resolve(self, phone: str | None) -> TenantConfig | None:
        """Resolve a phone number to a TenantConfig."""
        if not phone:
            return None
        return await self._loader.get_by_phone(phone)

    async def resolve_with_experiment(
        self, phone: str | None
    ) -> tuple[TenantConfig | None, dict | None]:
        """Resolve a phone, picking a variant if it belongs to a cohort.

        Returns (TenantConfig | None, experiment_meta | None).
        """
        if not phone:
            return None, None
        return await self._loader.resolve_with_experiment(phone)

    def resolve_by_slug(self, slug: str) -> TenantConfig | None:
        """Resolve a tenant slug/id to a TenantConfig."""
        return self._loader.get_by_slug(slug)

    def get_default(self) -> TenantConfig:
        """Get a fallback default config."""
        return self._loader.get_default()

    def list_tenants(self) -> list[str]:
        """List all loaded tenant slugs."""
        return self._loader.list_tenants()

    def reload(self) -> None:
        """Reload configs from disk."""
        self._loader.reload()


def get_registry() -> TenantRegistry:
    """Get singleton registry instance."""
    global _registry
    if _registry is None:
        _registry = TenantRegistry()
    return _registry
