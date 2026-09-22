"""
Outbound network topology and request tracing.

Captures outbound HTTP requests made through aiohttp/httpx, groups them by
service, and exposes a recent request log plus endpoint resolution details.
"""

from __future__ import annotations

import contextlib
import contextvars
import ipaddress
import json
import logging
import os
import socket
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_REQUEST_CONTEXT: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "network_request_context",
    default=None,
)
_SUPPRESSED: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "network_observer_suppressed",
    default=False,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_port(scheme: str) -> int | None:
    if scheme == "https":
        return 443
    if scheme == "http":
        return 80
    if scheme == "wss":
        return 443
    if scheme == "ws":
        return 80
    return None


def classify_network_location(value: str) -> str:
    """Classify an IP or special hostname into a simple network zone."""
    if not value:
        return "unknown"

    lowered = value.lower()
    if lowered == "localhost":
        return "loopback"

    try:
        addr = ipaddress.ip_address(value)
    except ValueError:
        return "hostname"

    if addr.is_loopback:
        return "loopback"
    if addr.is_private:
        return "private-network"
    if addr.is_link_local:
        return "link-local"
    if addr.is_multicast:
        return "multicast"
    if addr.is_reserved:
        return "reserved"
    return "public-internet"


@dataclass
class EndpointResolution:
    host: str
    scheme: str
    port: int | None
    network_location: str
    addresses: list[dict[str, Any]] = field(default_factory=list)
    resolved_at: str | None = None
    resolution_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "scheme": self.scheme,
            "port": self.port,
            "network_location": self.network_location,
            "addresses": self.addresses,
            "resolved_at": self.resolved_at,
            "resolution_error": self.resolution_error,
        }


@dataclass
class ServiceRegistration:
    service: str
    provider: str
    base_url: str
    metadata: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    registered_at: str = field(default_factory=_utc_now)

    def to_dict(self, resolution: EndpointResolution) -> dict[str, Any]:
        return {
            "service": self.service,
            "provider": self.provider,
            "base_url": self.base_url,
            "metadata": self.metadata,
            "notes": self.notes,
            "registered_at": self.registered_at,
            "endpoint": resolution.to_dict(),
        }


class NetworkTopologyTracker:
    """Tracks outbound request topology and recent network activity."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        max_events: int | None = None,
        persist_path: str | None = None,
    ):
        self.enabled = enabled
        self.max_events = max_events or int(os.getenv("NETWORK_OBS_MAX_EVENTS", "500"))
        self.geo_enabled = os.getenv("NETWORK_GEO_ENABLED", "0").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.geo_base_url = os.getenv("NETWORK_GEO_BASE_URL", "https://ipwho.is").rstrip("/")
        self.persist_path = persist_path or os.getenv(
            "NETWORK_OBS_PERSIST_PATH",
            "/tmp/navai-network-topology.jsonl",
        )
        self.process_id = os.getpid()
        self._lock = threading.RLock()
        self._requests: deque[dict[str, Any]] = deque(maxlen=self.max_events)
        self._services: dict[str, ServiceRegistration] = {}
        self._service_by_endpoint: dict[tuple[str, str, int | None], str] = {}
        self._stats: dict[str, dict[str, Any]] = {}
        self._resolution_cache: dict[tuple[str, str, int | None], EndpointResolution] = {}
        self._geo_cache: dict[str, dict[str, Any] | None] = {}
        self._installed = False

    @contextlib.contextmanager
    def request_context(
        self,
        *,
        service: str | None = None,
        operation: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """Attach service metadata to nested outbound requests."""
        parent = _REQUEST_CONTEXT.get() or {}
        merged_metadata = dict(parent.get("metadata") or {})
        if metadata:
            merged_metadata.update(metadata)

        token = _REQUEST_CONTEXT.set(
            {
                "service": service or parent.get("service"),
                "operation": operation or parent.get("operation"),
                "metadata": merged_metadata,
            }
        )
        try:
            yield
        finally:
            _REQUEST_CONTEXT.reset(token)

    @contextlib.contextmanager
    def suppress(self):
        """Disable tracing for internal observer operations."""
        token = _SUPPRESSED.set(True)
        try:
            yield
        finally:
            _SUPPRESSED.reset(token)

    def install(self) -> None:
        """Patch outbound HTTP clients once per process."""
        if not self.enabled or self._installed:
            return

        self._patch_aiohttp()
        self._patch_httpx()
        self._installed = True
        logger.info("Network topology observer installed")

    def register_service(
        self,
        *,
        service: str,
        base_url: str,
        provider: str = "",
        metadata: dict[str, Any] | None = None,
        notes: str = "",
    ) -> None:
        """Register a known outbound dependency for topology grouping."""
        if not self.enabled or not base_url:
            return

        parsed = urlparse(base_url)
        host = parsed.hostname or ""
        if not host:
            return

        endpoint_key = (parsed.scheme or "http", host, parsed.port or _default_port(parsed.scheme))

        with self._lock:
            self._services[service] = ServiceRegistration(
                service=service,
                provider=provider or service,
                base_url=base_url,
                metadata=metadata or {},
                notes=notes,
            )
            self._service_by_endpoint[endpoint_key] = service
            registration = self._services[service]

        self._append_persisted_event(
            "service_registration",
            {
                "service": registration.service,
                "provider": registration.provider,
                "base_url": registration.base_url,
                "metadata": registration.metadata,
                "notes": registration.notes,
                "registered_at": registration.registered_at,
                "process_id": self.process_id,
            },
        )

    def record_request(
        self,
        *,
        transport: str,
        method: str,
        url: str,
        status_code: int | None,
        duration_ms: float,
        error: str | None = None,
    ) -> None:
        """Record one outbound request."""
        if not self.enabled:
            return

        parsed = urlparse(url)
        scheme = parsed.scheme or "http"
        host = parsed.hostname or ""
        port = parsed.port or _default_port(scheme)
        path = parsed.path or "/"
        endpoint_key = (scheme, host, port)
        ctx = _REQUEST_CONTEXT.get() or {}
        service = ctx.get("service") or self._infer_service(endpoint_key) or host or "unknown"
        operation = ctx.get("operation") or ""
        metadata = dict(ctx.get("metadata") or {})

        event = {
            "timestamp": _utc_now(),
            "process_id": self.process_id,
            "transport": transport,
            "service": service,
            "operation": operation,
            "method": method.upper(),
            "url": url,
            "scheme": scheme,
            "host": host,
            "port": port,
            "path": path,
            "network_location": classify_network_location(host),
            "status_code": status_code,
            "duration_ms": round(duration_ms, 2),
            "ok": error is None and (status_code is None or status_code < 400),
            "error": error,
            "metadata": metadata,
        }

        with self._lock:
            self._requests.append(event)
            stats = self._stats.setdefault(
                service,
                {
                    "request_count": 0,
                    "error_count": 0,
                    "total_duration_ms": 0.0,
                    "last_request": None,
                },
            )
            stats["request_count"] += 1
            stats["total_duration_ms"] += duration_ms
            if error is not None or (status_code is not None and status_code >= 400):
                stats["error_count"] += 1
            stats["last_request"] = event

        self._append_persisted_event("request", event)

        operation_suffix = f"/{operation}" if operation else ""
        status_label = status_code if status_code is not None else "error"
        logger.info(
            "[NET][%s%s] %s %s -> %s in %.0fms",
            service,
            operation_suffix,
            method.upper(),
            url,
            status_label,
            duration_ms,
        )

    def snapshot(self, *, limit: int = 100) -> dict[str, Any]:
        """Return services, aggregate stats, and recent requests."""
        persisted = self._load_persisted_snapshot(limit=limit)
        if persisted is not None:
            return persisted

        with self._lock:
            services = list(self._services.values())
            stats = dict(self._stats)
            requests = list(self._requests)[-limit:]

        service_items = []
        for registration in services:
            resolution = self.resolve_endpoint(registration.base_url)
            service_stats = stats.get(registration.service, {})
            item = registration.to_dict(resolution)
            item["stats"] = self._stats_to_dict(service_stats)
            service_items.append(item)

        unresolved_services = {
            req["service"] for req in requests if req["service"] not in self._services
        }
        for service in sorted(unresolved_services):
            service_items.append(
                {
                    "service": service,
                    "provider": service,
                    "base_url": "",
                    "metadata": {},
                    "notes": "Observed dynamically from outbound traffic",
                    "registered_at": None,
                    "endpoint": None,
                    "stats": self._stats_to_dict(stats.get(service, {})),
                }
            )

        return {
            "generated_at": _utc_now(),
            "geo_enabled": self.geo_enabled,
            "services": service_items,
            "recent_requests": requests,
        }

    def _load_persisted_snapshot(self, *, limit: int) -> dict[str, Any] | None:
        if not self.persist_path or not os.path.exists(self.persist_path):
            return None

        services: dict[str, dict[str, Any]] = {}
        stats: dict[str, dict[str, Any]] = {}
        requests: deque[dict[str, Any]] = deque(maxlen=max(limit, 1))

        try:
            with open(self.persist_path, encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    event_type = item.get("type")
                    data = item.get("data") or {}
                    if event_type == "service_registration":
                        services[data.get("service", f"unknown-{len(services)}")] = data
                    elif event_type == "request":
                        service = data.get("service", "unknown")
                        requests.append(data)
                        service_stats = stats.setdefault(
                            service,
                            {
                                "request_count": 0,
                                "error_count": 0,
                                "total_duration_ms": 0.0,
                                "last_request": None,
                            },
                        )
                        service_stats["request_count"] += 1
                        service_stats["total_duration_ms"] += float(data.get("duration_ms", 0.0))
                        if (not data.get("ok", True)) or (
                            data.get("status_code") is not None
                            and data.get("status_code", 0) >= 400
                        ):
                            service_stats["error_count"] += 1
                        service_stats["last_request"] = data
        except Exception as exc:
            logger.warning("Failed to load persisted network topology: %s", exc)
            return None

        service_items = []
        for service_name, item in sorted(services.items()):
            base_url = item.get("base_url", "")
            endpoint = self.resolve_endpoint(base_url).to_dict() if base_url else None
            service_items.append(
                {
                    "service": service_name,
                    "provider": item.get("provider", service_name),
                    "base_url": base_url,
                    "metadata": item.get("metadata", {}),
                    "notes": item.get("notes", ""),
                    "registered_at": item.get("registered_at"),
                    "process_id": item.get("process_id"),
                    "endpoint": endpoint,
                    "stats": self._stats_to_dict(stats.get(service_name, {})),
                }
            )

        unresolved_services = {req["service"] for req in requests if req["service"] not in services}
        for service_name in sorted(unresolved_services):
            service_items.append(
                {
                    "service": service_name,
                    "provider": service_name,
                    "base_url": "",
                    "metadata": {},
                    "notes": "Observed dynamically from outbound traffic",
                    "registered_at": None,
                    "process_id": None,
                    "endpoint": None,
                    "stats": self._stats_to_dict(stats.get(service_name, {})),
                }
            )

        return {
            "generated_at": _utc_now(),
            "geo_enabled": self.geo_enabled,
            "persist_path": self.persist_path,
            "services": service_items,
            "recent_requests": list(requests),
        }

    def resolve_endpoint(self, base_url: str) -> EndpointResolution:
        """Resolve hostnames to addresses and enrich with network location."""
        parsed = urlparse(base_url)
        scheme = parsed.scheme or "http"
        host = parsed.hostname or ""
        port = parsed.port or _default_port(scheme)
        endpoint_key = (scheme, host, port)

        with self._lock:
            cached = self._resolution_cache.get(endpoint_key)
        if cached:
            return cached

        resolution = self._resolve_host(scheme=scheme, host=host, port=port)
        with self._lock:
            self._resolution_cache[endpoint_key] = resolution
        return resolution

    def _resolve_host(
        self,
        *,
        scheme: str,
        host: str,
        port: int | None,
    ) -> EndpointResolution:
        if not host:
            return EndpointResolution(
                host="",
                scheme=scheme,
                port=port,
                network_location="unknown",
                resolution_error="missing host",
            )

        host_location = classify_network_location(host)
        addresses: list[dict[str, Any]] = []
        resolution_error: str | None = None

        try:
            if host_location == "hostname":
                resolved = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
                seen = set()
                for item in resolved:
                    ip_value = item[4][0]
                    if ip_value in seen:
                        continue
                    seen.add(ip_value)
                    addresses.append(self._address_entry(ip_value))
            else:
                addresses.append(self._address_entry(host))
        except Exception as exc:
            resolution_error = str(exc)

        return EndpointResolution(
            host=host,
            scheme=scheme,
            port=port,
            network_location=host_location,
            addresses=addresses,
            resolved_at=_utc_now(),
            resolution_error=resolution_error,
        )

    def _address_entry(self, ip_value: str) -> dict[str, Any]:
        entry = {
            "ip": ip_value,
            "network_location": classify_network_location(ip_value),
        }
        if self.geo_enabled and entry["network_location"] == "public-internet":
            geo = self._lookup_geo(ip_value)
            if geo:
                entry["geo"] = geo
        return entry

    def _lookup_geo(self, ip_value: str) -> dict[str, Any] | None:
        with self._lock:
            if ip_value in self._geo_cache:
                return self._geo_cache[ip_value]

        try:
            import httpx

            with self.suppress():
                response = httpx.get(
                    f"{self.geo_base_url}/{ip_value}",
                    timeout=2.5,
                    headers={"Accept": "application/json"},
                )
            if response.status_code != 200:
                geo = None
            else:
                data = response.json()
                if data.get("success") is False:
                    geo = None
                else:
                    geo = {
                        "city": data.get("city"),
                        "region": data.get("region") or data.get("region_name"),
                        "country": data.get("country"),
                        "country_code": data.get("country_code"),
                        "org": data.get("connection", {}).get("org") or data.get("org"),
                    }
        except Exception:
            geo = None

        with self._lock:
            self._geo_cache[ip_value] = geo
        return geo

    def _stats_to_dict(self, stats: dict[str, Any]) -> dict[str, Any]:
        request_count = int(stats.get("request_count", 0))
        total_duration_ms = float(stats.get("total_duration_ms", 0.0))
        avg_ms = round(total_duration_ms / request_count, 2) if request_count else 0.0
        return {
            "request_count": request_count,
            "error_count": int(stats.get("error_count", 0)),
            "avg_duration_ms": avg_ms,
            "last_request": stats.get("last_request"),
        }

    def _infer_service(self, endpoint_key: tuple[str, str, int | None]) -> str | None:
        with self._lock:
            return self._service_by_endpoint.get(endpoint_key)

    def _append_persisted_event(self, event_type: str, data: dict[str, Any]) -> None:
        if not self.persist_path or not self.enabled:
            return

        try:
            payload = (
                json.dumps(
                    {
                        "type": event_type,
                        "data": data,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            directory = os.path.dirname(self.persist_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            fd = os.open(
                self.persist_path,
                os.O_APPEND | os.O_CREAT | os.O_WRONLY,
                0o644,
            )
            try:
                os.write(fd, payload.encode("utf-8"))
            finally:
                os.close(fd)
        except Exception as exc:
            logger.debug("Failed to persist network event: %s", exc)

    def _patch_aiohttp(self) -> None:
        try:
            import aiohttp
        except Exception:
            return

        original = aiohttp.ClientSession._request
        if getattr(original, "__network_observer_patched__", False):
            return

        tracker = self

        async def traced_request(session, method, str_or_url, *args, **kwargs):
            if _SUPPRESSED.get():
                return await original(session, method, str_or_url, *args, **kwargs)

            start = time.perf_counter()
            try:
                response = await original(session, method, str_or_url, *args, **kwargs)
                tracker.record_request(
                    transport="aiohttp",
                    method=method,
                    url=str(str_or_url),
                    status_code=response.status,
                    duration_ms=(time.perf_counter() - start) * 1000,
                )
                return response
            except Exception as exc:
                tracker.record_request(
                    transport="aiohttp",
                    method=method,
                    url=str(str_or_url),
                    status_code=None,
                    duration_ms=(time.perf_counter() - start) * 1000,
                    error=type(exc).__name__,
                )
                raise

        traced_request.__network_observer_patched__ = True
        aiohttp.ClientSession._request = traced_request

    def _patch_httpx(self) -> None:
        try:
            import httpx
        except Exception:
            return

        tracker = self

        async_original = httpx.AsyncClient.send
        if not getattr(async_original, "__network_observer_patched__", False):

            async def traced_async_send(client, request, *args, **kwargs):
                if _SUPPRESSED.get():
                    return await async_original(client, request, *args, **kwargs)

                start = time.perf_counter()
                try:
                    response = await async_original(client, request, *args, **kwargs)
                    tracker.record_request(
                        transport="httpx_async",
                        method=request.method,
                        url=str(request.url),
                        status_code=response.status_code,
                        duration_ms=(time.perf_counter() - start) * 1000,
                    )
                    return response
                except Exception as exc:
                    tracker.record_request(
                        transport="httpx_async",
                        method=request.method,
                        url=str(request.url),
                        status_code=None,
                        duration_ms=(time.perf_counter() - start) * 1000,
                        error=type(exc).__name__,
                    )
                    raise

            traced_async_send.__network_observer_patched__ = True
            httpx.AsyncClient.send = traced_async_send

        sync_original = httpx.Client.send
        if not getattr(sync_original, "__network_observer_patched__", False):

            def traced_sync_send(client, request, *args, **kwargs):
                if _SUPPRESSED.get():
                    return sync_original(client, request, *args, **kwargs)

                start = time.perf_counter()
                try:
                    response = sync_original(client, request, *args, **kwargs)
                    tracker.record_request(
                        transport="httpx",
                        method=request.method,
                        url=str(request.url),
                        status_code=response.status_code,
                        duration_ms=(time.perf_counter() - start) * 1000,
                    )
                    return response
                except Exception as exc:
                    tracker.record_request(
                        transport="httpx",
                        method=request.method,
                        url=str(request.url),
                        status_code=None,
                        duration_ms=(time.perf_counter() - start) * 1000,
                        error=type(exc).__name__,
                    )
                    raise

            traced_sync_send.__network_observer_patched__ = True
            httpx.Client.send = traced_sync_send


_network_tracker: NetworkTopologyTracker | None = None


def _get_tracker() -> NetworkTopologyTracker:
    global _network_tracker
    if _network_tracker is None:
        _network_tracker = NetworkTopologyTracker(
            enabled=os.getenv("NETWORK_OBS_ENABLED", "1").lower() not in {"0", "false", "no", "off"}
        )
    return _network_tracker


def install_network_observer() -> None:
    _get_tracker().install()


def register_service_route(
    service: str,
    base_url: str,
    *,
    provider: str = "",
    metadata: dict[str, Any] | None = None,
    notes: str = "",
) -> None:
    _get_tracker().register_service(
        service=service,
        base_url=base_url,
        provider=provider,
        metadata=metadata,
        notes=notes,
    )


@contextlib.contextmanager
def network_request_context(
    service: str | None = None,
    operation: str | None = None,
    metadata: dict[str, Any] | None = None,
):
    with _get_tracker().request_context(
        service=service,
        operation=operation,
        metadata=metadata,
    ):
        yield


@contextlib.contextmanager
def suppress_network_observer():
    with _get_tracker().suppress():
        yield


def get_network_topology_snapshot(limit: int = 100) -> dict[str, Any]:
    return _get_tracker().snapshot(limit=limit)
