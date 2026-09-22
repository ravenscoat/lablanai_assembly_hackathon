"""
Langfuse/OpenTelemetry wiring for LiveKit Agents.

The integration is intentionally env-gated: if Langfuse credentials are absent,
the agent behaves exactly as it did before. If LANGFUSE_ENABLED is explicitly
true, missing or invalid configuration fails fast during startup.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}
_trace_provider: Any | None = None


@dataclass
class LangfuseTracing:
    """Holds Langfuse tracing handles that need flushing on call shutdown."""

    trace_provider: Any
    client: Any

    def flush(self) -> None:
        self.trace_provider.force_flush()
        flush = getattr(self.client, "flush", None)
        if callable(flush):
            flush()


def _read_bool_env(name: str) -> bool | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    normalized = raw.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ValueError(f"{name} must be one of: {', '.join(sorted(_TRUE_VALUES | _FALSE_VALUES))}")


def _clean_metadata(metadata: dict[str, Any] | None) -> dict[str, str | bool | int | float]:
    clean: dict[str, str | bool | int | float] = {}
    for key, value in (metadata or {}).items():
        if value is None:
            continue
        if isinstance(value, (str, bool, int, float)):
            clean[key] = value
        else:
            clean[key] = str(value)
    return clean


def setup_langfuse(metadata: dict[str, Any] | None = None) -> LangfuseTracing | None:
    """
    Configure LiveKit's OpenTelemetry provider to export spans to Langfuse.

    Env vars:
    - LANGFUSE_ENABLED: optional; when true, config errors are fatal.
    - LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY
    - LANGFUSE_HOST or LANGFUSE_BASE_URL
    """

    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST") or os.getenv("LANGFUSE_BASE_URL")
    explicit_enabled = _read_bool_env("LANGFUSE_ENABLED")
    has_config = bool(public_key and secret_key and host)
    enabled = explicit_enabled if explicit_enabled is not None else has_config

    if not enabled:
        return None

    missing = [
        name
        for name, value in (
            ("LANGFUSE_PUBLIC_KEY", public_key),
            ("LANGFUSE_SECRET_KEY", secret_key),
            ("LANGFUSE_HOST or LANGFUSE_BASE_URL", host),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(f"Langfuse is enabled but missing: {', '.join(missing)}")

    try:
        from langfuse import Langfuse
        from livekit.agents.telemetry import set_tracer_provider
        from opentelemetry.sdk.trace import TracerProvider
    except ImportError as exc:
        raise RuntimeError(
            "Langfuse tracing requires langfuse and opentelemetry-sdk dependencies"
        ) from exc

    global _trace_provider
    trace_provider = TracerProvider()
    _trace_provider = trace_provider
    set_tracer_provider(trace_provider, metadata=_clean_metadata(metadata))
    client = Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        base_url=host,
        tracer_provider=trace_provider,
        should_export_span=lambda span: True,
    )
    logger.info("Langfuse tracing enabled: host=%s", host)
    return LangfuseTracing(trace_provider=trace_provider, client=client)


def record_latency_span(
    name: str,
    duration_ms: float,
    attributes: dict[str, Any] | None = None,
) -> None:
    """Record a manual OpenTelemetry span for custom pipeline components."""
    if duration_ms <= 0:
        return

    end_time = time.time_ns()
    start_time = end_time - int(duration_ms * 1_000_000)
    if _trace_provider is not None:
        tracer = _trace_provider.get_tracer("navai-agent.pipeline")
    else:
        try:
            from opentelemetry import trace
        except ImportError:
            return
        tracer = trace.get_tracer("navai-agent.pipeline")
    span = tracer.start_span(
        name,
        start_time=start_time,
        attributes=_clean_metadata(
            {
                "latency_ms": round(duration_ms, 3),
                "latency_seconds": round(duration_ms / 1000, 6),
                **(attributes or {}),
            }
        ),
    )
    span.end(end_time=end_time)
