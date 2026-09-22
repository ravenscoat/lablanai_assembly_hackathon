"""
Config loader: loads tenant configs from YAML files + Platform API.
YAML files are the primary source; Platform API is secondary with 5-min cache.
"""

from __future__ import annotations

import copy
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

import httpx
import yaml

from config.api_keys import resolve_default_api_key, resolve_tenant_api_key_for_phone
from observability.network_topology import network_request_context, register_service_route

from .experiments import CohortMember, ExperimentCohort, assign_variant
from .schema import TenantConfig

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_DIR = Path(__file__).parent.parent.parent / "configs" / "tenants"
ENVIRONMENTS_FILE = Path(__file__).parent.parent.parent / "configs" / "environments.yaml"


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. Override wins on conflicts."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


class ConfigLoader:
    """
    Loads tenant configs from:
    1. YAML files in configs/tenants/ (primary, for known tenants)
    2. Platform API by phone number (dynamic, with 5-min cache)
    3. Defaults from _defaults.yaml (fallback)
    """

    def __init__(
        self,
        config_dir: Path | None = None,
        api_base_url: str | None = None,
        api_key: str | None = None,
        cache_ttl_seconds: int = 300,
    ):
        self._config_dir = config_dir or DEFAULT_CONFIG_DIR
        self._api_base_url = api_base_url or os.getenv("PLATFORM_API_URL", "http://localhost:3000")
        self._api_key = resolve_default_api_key(api_key)
        self._cache_ttl = cache_ttl_seconds
        register_service_route(
            "platform_api",
            self._api_base_url,
            provider="platform_api",
            notes="Tenant config lookup",
        )

        # Loaded configs
        self._defaults: dict[str, Any] = {}
        self._configs_by_slug: dict[str, TenantConfig] = {}
        self._configs_by_phone: dict[str, TenantConfig] = {}
        self._experiments_by_phone: dict[str, ExperimentCohort] = {}

        # Per-env identity overrides applied at YAML load. None when no
        # CONFIG_ENV is set (local dev path) — tenant YAMLs are used as-is.
        # Shape: {slug: {"id": str, "phone": str}}.
        self._env_overrides: dict[str, dict[str, str]] | None = None

        # API cache
        self._api_cache: dict[str, TenantConfig] = {}
        self._api_cache_ts: dict[str, float] = {}

        # Load on init
        self._load_defaults()
        self._load_environment_overrides()
        self._load_all_yaml()

    def _load_defaults(self) -> None:
        defaults_path = self._config_dir / "_defaults.yaml"
        if defaults_path.exists():
            with open(defaults_path, encoding="utf-8") as f:
                self._defaults = yaml.safe_load(f) or {}
            logger.info("Loaded defaults from _defaults.yaml")
        else:
            logger.warning(f"No _defaults.yaml found at {defaults_path}")

    def _load_environment_overrides(self) -> None:
        # When CONFIG_ENV is set, override tenant identity (id + phone) per
        # slug from configs/environments.yaml. Unset = local dev path: YAMLs
        # used verbatim.
        env_name = os.getenv("CONFIG_ENV", "").strip()
        if not env_name:
            return
        data = yaml.safe_load(ENVIRONMENTS_FILE.read_text(encoding="utf-8")) or {}
        self._env_overrides = data[env_name]
        logger.info(
            "CONFIG_ENV=%s: %d tenant override(s) loaded", env_name, len(self._env_overrides)
        )

    def _load_all_yaml(self) -> None:
        """Load all tenant YAML files at startup."""
        if not self._config_dir.exists():
            logger.warning(f"Config directory does not exist: {self._config_dir}")
            return

        loaded: list[TenantConfig] = []
        for yaml_file in sorted(self._config_dir.glob("*.yaml")):
            if yaml_file.name.startswith("_"):
                continue
            try:
                config = self._load_tenant_yaml(yaml_file)
                loaded.append(config)
            except Exception as e:
                # Match existing behavior: log per-file errors and continue.
                # Cohort validation errors (raised from _group_experiments)
                # are intentionally NOT caught here — they fail the load.
                logger.error(f"Failed to load {yaml_file.name}: {e}")

        # Pass 1: group experiment variants into cohorts. Raises on
        # cohort validation failure (intentional fail-fast).
        non_experiment_configs = self._group_experiments(loaded)

        # Pass 2: index non-experiment tenants by slug and phone.
        for config in non_experiment_configs:
            self._index_tenant_slug(config)
            self._index_tenant_phone(config)

        # Index variants under their shared slug so resolve_by_slug returns one
        # of them (last-loaded wins; intentional and documented).
        for cohort in self._unique_cohorts():
            for member in cohort.members:
                self._index_tenant_slug(member.config)

        logger.info(
            "Loaded tenant configs: %d single-tenant + %d experiment cohorts",
            len(non_experiment_configs),
            len(self._unique_cohorts()),
        )

    def _unique_cohorts(self) -> list[ExperimentCohort]:
        """Return the unique cohorts (deduplicating across phone keys)."""
        seen: set[int] = set()
        unique: list[ExperimentCohort] = []
        for cohort in self._experiments_by_phone.values():
            if id(cohort) in seen:
                continue
            seen.add(id(cohort))
            unique.append(cohort)
        return unique

    def _load_tenant_yaml(self, path: Path) -> TenantConfig:
        """Load a single tenant YAML, merge with defaults, validate."""
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        merged = deep_merge(self._defaults, raw)
        self._apply_env_override(merged)
        config = TenantConfig.model_validate(merged)
        return config

    def _apply_env_override(self, merged: dict[str, Any]) -> None:
        if self._env_overrides is None:
            return
        slug = merged["tenant"]["slug"]
        override = self._env_overrides.get(slug)
        if override is None:
            # New tenant not yet added to environments.yaml — warn but let
            # the YAML's own id/phone through rather than crash the agent.
            logger.warning("No env override for slug %r; using YAML values", slug)
            return
        merged["tenant"]["id"] = override["id"]
        merged["tenant"]["phone_numbers"] = [override["phone"]]

    def _index_tenant_slug(self, config: TenantConfig) -> None:
        self._configs_by_slug[config.tenant.slug] = config
        self._configs_by_slug[config.tenant.id] = config

    def _index_tenant_phone(self, config: TenantConfig) -> None:
        for phone in config.tenant.phone_numbers:
            normalized = self._normalize_phone(phone)
            self._configs_by_phone[normalized] = config

    def _group_experiments(self, loaded: list[TenantConfig]) -> list[TenantConfig]:
        """Group variant TenantConfigs into ExperimentCohorts.

        Validates cohort integrity (identity field consistency, >=2 variants,
        unique variant names, no phone collisions). Populates
        `self._experiments_by_phone`. Returns the list of non-experiment
        configs that should be indexed normally.

        Raises ValueError on any validation failure.
        """
        non_experiment: list[TenantConfig] = []
        groups: dict[tuple[str, str], list[TenantConfig]] = {}

        for config in loaded:
            if config.experiment is None:
                non_experiment.append(config)
                continue
            key = (config.tenant.slug, config.experiment.id)
            groups.setdefault(key, []).append(config)

        for (slug, exp_id), variants in groups.items():
            if len(variants) < 2:
                raise ValueError(
                    f"incomplete experiment cohort: expected >=2 variants for "
                    f"experiment '{exp_id}' on tenant '{slug}', found {len(variants)}"
                )

            self._validate_cohort_identity(exp_id, variants)

            seen_names: set[str] = set()
            members: list[CohortMember] = []
            for v in variants:
                assert v.experiment is not None  # guaranteed by groups grouping
                if v.experiment.variant in seen_names:
                    raise ValueError(
                        f"experiment '{exp_id}' has duplicate variant name "
                        f"'{v.experiment.variant}'"
                    )
                seen_names.add(v.experiment.variant)
                members.append(
                    CohortMember(
                        config=v,
                        weight=v.experiment.weight,
                        variant_name=v.experiment.variant,
                    )
                )

            cohort = ExperimentCohort(
                experiment_id=exp_id,
                members=tuple(members),
            )

            # Register the cohort under every phone of (any) variant — they
            # all match per identity validation above.
            for phone in variants[0].tenant.phone_numbers:
                normalized = self._normalize_phone(phone)
                self._experiments_by_phone[normalized] = cohort

        # Phone-collision check: a non-experiment tenant cannot claim a phone
        # that belongs to an experiment cohort.
        for config in non_experiment:
            for phone in config.tenant.phone_numbers:
                normalized = self._normalize_phone(phone)
                if normalized in self._experiments_by_phone:
                    cohort = self._experiments_by_phone[normalized]
                    raise ValueError(
                        f"phone {normalized} claimed by both single tenant "
                        f"'{config.tenant.slug}' and experiment "
                        f"'{cohort.experiment_id}'"
                    )

        return non_experiment

    def _validate_cohort_identity(self, exp_id: str, variants: list[TenantConfig]) -> None:
        """All variants must share tenant identity fields."""
        first = variants[0].tenant
        first_phones = set(first.phone_numbers)
        for v in variants[1:]:
            t = v.tenant
            if t.id != first.id:
                raise ValueError(
                    f"experiment '{exp_id}' variants disagree on tenant.id: "
                    f"{first.id!r} vs {t.id!r}"
                )
            if t.slug != first.slug:
                raise ValueError(
                    f"experiment '{exp_id}' variants disagree on tenant.slug: "
                    f"{first.slug!r} vs {t.slug!r}"
                )
            if t.name != first.name:
                raise ValueError(
                    f"experiment '{exp_id}' variants disagree on tenant.name: "
                    f"{first.name!r} vs {t.name!r}"
                )
            if set(t.phone_numbers) != first_phones:
                raise ValueError(
                    f"experiment '{exp_id}' variants disagree on tenant.phone_numbers: "
                    f"{sorted(first_phones)} vs {sorted(t.phone_numbers)}"
                )

    async def get_by_phone(self, phone: str) -> TenantConfig | None:
        """Look up config: YAML first, then Platform API with cache."""
        normalized = self._normalize_phone(phone)

        # 1. Check YAML-loaded configs
        if normalized in self._configs_by_phone:
            return self._configs_by_phone[normalized]

        # 2. Check API cache
        if normalized in self._api_cache:
            cache_time = self._api_cache_ts.get(normalized, 0)
            if time.time() - cache_time < self._cache_ttl:
                return self._api_cache[normalized]

        # 3. Fetch from Platform API
        return await self._fetch_from_api(normalized)

    async def resolve_with_experiment(self, phone: str) -> tuple[TenantConfig | None, dict | None]:
        """Resolve a phone, weighted-picking a variant if part of a cohort.

        Returns (config, experiment_meta).
          - experiment_meta is {"experiment_id": ..., "variant": ...} when a
            variant was picked.
          - experiment_meta is None when the phone is not in any cohort.
          - config is None when the phone resolves nowhere.
        """
        normalized = self._normalize_phone(phone)
        cohort = self._experiments_by_phone.get(normalized)
        if cohort is not None:
            member = assign_variant(cohort)
            meta = {
                "experiment_id": cohort.experiment_id,
                "variant": member.variant_name,
            }
            return member.config, meta

        config = await self.get_by_phone(phone)
        return config, None

    def get_by_slug(self, slug: str) -> TenantConfig | None:
        """Look up config by tenant slug or id."""
        return self._configs_by_slug.get(slug)

    def get_default(self) -> TenantConfig:
        """Return a default config (first loaded tenant or bare defaults)."""
        if self._configs_by_slug:
            return next(iter(self._configs_by_slug.values()))
        return TenantConfig.model_validate(
            deep_merge(self._defaults, {"tenant": {"id": "default", "name": "Default Agent"}})
        )

    def list_tenants(self) -> list[str]:
        """List all loaded tenant slugs."""
        return list({c.tenant.slug for c in self._configs_by_slug.values()})

    async def _fetch_from_api(self, phone: str) -> TenantConfig | None:
        """Fetch tenant config from Platform API with caching."""
        api_key = resolve_tenant_api_key_for_phone(phone)
        if not api_key:
            logger.error(
                "[tenant_api_key_missing] flow=config_lookup phone=%s action=skip_api_fetch",
                phone,
            )
            return None
        try:
            with network_request_context(
                "platform_api",
                "resolve_tenant_config",
                metadata={"phone": phone},
            ):
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.get(
                        f"{self._api_base_url}/api/v1/internal/config",
                        params={"phone": phone},
                        headers={"X-API-Key": api_key},
                    )

                    if response.status_code == 404:
                        logger.warning(f"No tenant config from API for phone: {phone}")
                        return None

                    response.raise_for_status()
                    data = response.json()

                    if data.get("success") and data.get("data"):
                        api_data = data["data"]
                        config_data = api_data.get("config", api_data)

                        # Build tenant identity from API response
                        config_data.setdefault("tenant", {})
                        config_data["tenant"]["id"] = api_data.get("tenant_id", "api-tenant")
                        config_data["tenant"]["slug"] = api_data.get("tenant_slug", "api-tenant")
                        config_data["tenant"]["name"] = config_data.get("name", "API Agent")
                        config_data["tenant"].setdefault("phone_numbers", [phone])

                        merged = deep_merge(self._defaults, config_data)
                        config = TenantConfig.model_validate(merged)

                        # Cache
                        self._api_cache[phone] = config
                        self._api_cache_ts[phone] = time.time()

                        logger.info(f"Loaded config from API for tenant: {config.tenant.slug}")
                        return config

                    return None

        except httpx.TimeoutException:
            logger.error(f"Timeout loading config from API for {phone}")
            return self._api_cache.get(phone)
        except httpx.RequestError as e:
            logger.error(f"Network error loading config: {e}")
            return self._api_cache.get(phone)
        except Exception as e:
            logger.error(f"Error loading config from API: {e}")
            return self._api_cache.get(phone)

    def reload(self) -> None:
        """Reload all YAML configs from disk.

        Atomic: if grouping/validation fails, the previous state is restored
        and the error is re-raised. The registry is never left in a partially
        cleared state by a failed reload.
        """
        # Snapshot for rollback. dict.copy is a shallow copy; references to
        # TenantConfig instances stay valid during the swap.
        saved_slug = self._configs_by_slug.copy()
        saved_phone = self._configs_by_phone.copy()
        saved_experiments = self._experiments_by_phone.copy()
        saved_defaults = self._defaults.copy() if self._defaults else {}

        self._configs_by_slug.clear()
        self._configs_by_phone.clear()
        self._experiments_by_phone.clear()

        try:
            self._load_defaults()
            self._load_all_yaml()
        except Exception:
            # Roll back to the previous state so callers continue to see
            # a consistent registry.
            self._configs_by_slug = saved_slug
            self._configs_by_phone = saved_phone
            self._experiments_by_phone = saved_experiments
            self._defaults = saved_defaults
            raise

    @staticmethod
    def _normalize_phone(phone: str) -> str:
        """Normalize phone number for consistent lookups."""
        raw = (phone or "").strip()
        if not raw:
            return ""

        lowered = raw.lower()

        # Accept SIP/TEL URIs coming from telephony metadata and normalize them
        # to their phone-like identity so YAML tenant mappings still match.
        for prefix in ("sip:", "tel:"):
            if lowered.startswith(prefix):
                lowered = lowered[len(prefix) :]
                break

        # Strip URI params and domain parts:
        #   +998...@host;user=phone?x=1 -> +998...
        for sep in ("@", ";", "?"):
            if sep in lowered:
                lowered = lowered.split(sep, 1)[0]

        # Preserve extension notation if present.
        if lowered.startswith("ext:"):
            digits = "".join(ch for ch in lowered[4:] if ch.isdigit())
            return f"ext:{digits}" if digits else "ext:"

        # Keep only digits and an optional leading plus.
        cleaned = re.sub(r"[^0-9+]", "", lowered)
        if cleaned.startswith("+"):
            return "+" + re.sub(r"[^0-9]", "", cleaned[1:])
        return re.sub(r"[^0-9]", "", cleaned)
