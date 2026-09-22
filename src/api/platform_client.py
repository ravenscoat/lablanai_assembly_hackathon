"""
PlatformClient — centralized async HTTP client for the Platform API.

Replaces scattered httpx.AsyncClient creation across the codebase with
a single reusable client that handles auth, timeouts, and error handling.
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone
from typing import Any

import httpx

from api.models import CallerHistory, MurojaatResult, OperatorStatus
from config.api_keys import resolve_default_api_key, resolve_tenant_api_key

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Validators — mirror server-side regex in openapi.json so we fail fast
# ---------------------------------------------------------------------------

# Matches server schema for caller_phone / agent_phone on /internal/calls*.
_PHONE_RE = re.compile(r"^(\+?[1-9]\d{1,14}|ext:\d{3,6}|unknown)$")

# MongoDB ObjectId — used for call DB id and murojat_id in PATCH body.
_OBJECT_ID_RE = re.compile(r"^[0-9a-fA-F]{24}$")

# Transcript item role values accepted by PATCH /internal/calls/:id.
_TRANSCRIPT_ROLES = {"agent", "user"}


def _normalize_phone(value: str | None) -> str:
    """Return a phone value that matches the server regex.

    Falls back to "unknown" for empty/unknown/malformed inputs so calls
    don't 400 on a stray character. Logs when a normalization happens.
    """
    if not value:
        return "unknown"
    candidate = value.strip()
    if _PHONE_RE.match(candidate):
        return candidate
    # Common fixup: strip spaces / dashes / parens, then retry.
    cleaned = re.sub(r"[\s\-()]", "", candidate)
    if _PHONE_RE.match(cleaned):
        return cleaned
    logger.warning("phone %r does not match server pattern; sending 'unknown'", value)
    return "unknown"


def _is_object_id(value: str | None) -> bool:
    return bool(value) and bool(_OBJECT_ID_RE.match(value))


def _sanitize_transcript(
    transcript: list[dict] | None,
) -> list[dict] | None:
    """Drop entries missing role/text and coerce role to the enum."""
    if transcript is None:
        return None
    cleaned: list[dict] = []
    for entry in transcript:
        role = entry.get("role")
        if role == "assistant":
            role = "agent"
        if role not in _TRANSCRIPT_ROLES:
            # Default unknown roles (e.g., "system") to "agent" so the
            # record still lands — skipping entirely would lose context.
            role = "agent"
        text = entry.get("text") or entry.get("content") or ""
        if not text:
            continue
        item: dict[str, Any] = {"role": role, "text": text}
        ts = entry.get("timestamp")
        if isinstance(ts, str) and ts:
            item["timestamp"] = ts
        cleaned.append(item)
    return cleaned


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class PlatformClient:
    """Async HTTP client for all Platform API communication."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        jwt: str | None = None,
        timeout: float = 10.0,
    ) -> None:
        raw_base = base_url or os.getenv("PLATFORM_API_URL", "http://localhost:3000")
        # All route paths in this client start with "/api/v1/...". If the
        # configured base already ends with "/api" (a common copy/paste
        # mistake) we'd end up hitting "/api/api/v1/..." and 404. Normalize.
        self._base_url = raw_base.rstrip("/")
        if self._base_url.endswith("/api"):
            logger.warning("PLATFORM_API_URL ends with '/api'; stripping to avoid /api/api/v1 404s")
            self._base_url = self._base_url[: -len("/api")]
        self._timeout = timeout

        self._explicit_api_key = api_key or ""
        self._api_key = resolve_default_api_key(api_key)
        # Optional JWT for endpoints whose openapi spec declares `bearerAuth`
        # (e.g. /api/v1/murojatlar/voice-assistant). Falls back to x-api-key
        # when unset so existing deployments keep working.
        self._jwt = jwt or os.getenv("PLATFORM_API_JWT", "")
        self._http = httpx.AsyncClient(
            timeout=timeout,
            headers={"X-API-Key": self._api_key},
        )

    def _tenant_api_key(self, tenant_hint: str | None) -> str:
        return resolve_tenant_api_key(tenant_hint) or self._explicit_api_key

    @staticmethod
    def _resolve_tenant_hint(tenant_id: str, tenant_slug: str | None) -> str:
        """Prefer tenant slug (from YAML) for key resolution, fallback to id."""
        return (tenant_slug or "").strip() or tenant_id

    def _require_tenant_headers(self, tenant_hint: str | None, flow: str) -> dict[str, str] | None:
        api_key = self._tenant_api_key(tenant_hint)
        if not api_key:
            logger.error(
                "[tenant_api_key_missing] flow=%s tenant_hint=%s action=skip_request",
                flow,
                tenant_hint or "none",
            )
            return None
        return {"X-API-Key": api_key}

    def _bearer_headers(self) -> dict[str, str]:
        """Return auth headers for routes documented as bearerAuth.

        Uses `Authorization: Bearer <jwt>` when `PLATFORM_API_JWT` is set;
        otherwise returns the x-api-key header as a compatibility fallback
        for backends that accept the internal api-key on these routes.
        """
        if self._jwt:
            return {"Authorization": f"Bearer {self._jwt}"}
        return {"X-API-Key": self._api_key}

    # ------------------------------------------------------------------
    # Calls
    # ------------------------------------------------------------------

    async def create_call(
        self,
        *,
        tenant_id: str,
        tenant_slug: str | None = None,
        call_sid: str,
        caller_phone: str,
        agent_phone: str,
        direction: str = "inbound",
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        """POST /api/v1/internal/calls — create a call record, return DB id.

        Server schema (openapi.json): required=[call_sid, caller_phone,
        agent_phone, tenant_id]; direction ∈ {inbound, outbound}; status ∈
        {ringing, in_progress, completed, failed, transferred, abandoned,
        in_queue, missed}; phone fields must match ^(\\+?[1-9]\\d{1,14}|
        ext:\\d{3,6}|unknown)$.
        """
        if not tenant_id:
            logger.error("create_call: tenant_id is required")
            return None
        if not call_sid:
            logger.error("create_call: call_sid is required")
            return None
        if direction not in ("inbound", "outbound"):
            logger.warning("create_call: invalid direction %r, coercing to 'inbound'", direction)
            direction = "inbound"
        tenant_hint = self._resolve_tenant_hint(tenant_id, tenant_slug)
        headers = self._require_tenant_headers(tenant_hint, "create_call")
        if not headers:
            return None
        try:
            body: dict[str, Any] = {
                "tenant_id": tenant_id,
                "call_sid": call_sid,
                "caller_phone": _normalize_phone(caller_phone),
                "agent_phone": _normalize_phone(agent_phone),
                "direction": direction,
                "status": "in_progress",
                "started_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            if metadata:
                body["metadata"] = metadata

            resp = await self._http.post(
                f"{self._base_url}/api/v1/internal/calls",
                json=body,
                headers=headers,
            )
            if resp.status_code in (200, 201):
                data = resp.json()
                inner = data.get("data", {})
                return inner.get("_id") or inner.get("id") or data.get("_id") or data.get("id")

            logger.warning(
                "create_call got status %s | body=%s | response=%s",
                resp.status_code,
                body,
                resp.text[:500],
            )
        except Exception:
            logger.exception("create_call failed")
        return None

    async def update_call(
        self,
        *,
        call_db_id: str,
        tenant_id: str,
        tenant_slug: str | None = None,
        status: str = "completed",
        duration_seconds: int = 0,
        transcript: list[dict] | None = None,
        transcript_text: str | None = None,
        ai_summary: str | None = None,
        transfer_reason: str | None = None,
        ended_at: str | None = None,
        metrics: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        murojat_id: str | None = None,
    ) -> bool:
        """PATCH /api/v1/internal/calls/:id — update a call record.

        Server schema marks `tenant_id` as required, and both the path `:id`
        and the optional body `murojat_id` must match ^[0-9a-fA-F]{24}$.
        Status must be in the enum; transcript items must have role ∈
        {agent, user} and a non-empty text.
        """
        if not tenant_id:
            logger.error("update_call: tenant_id is required by server schema")
            return False
        if not _is_object_id(call_db_id):
            logger.error("update_call: call_db_id %r is not a 24-hex ObjectId", call_db_id)
            return False
        valid_statuses = {
            "ringing",
            "in_progress",
            "completed",
            "failed",
            "transferred",
            "abandoned",
            "in_queue",
            "missed",
        }
        if status not in valid_statuses:
            logger.warning("update_call: invalid status %r, coercing to 'completed'", status)
            status = "completed"
        tenant_hint = self._resolve_tenant_hint(tenant_id, tenant_slug)
        headers = self._require_tenant_headers(tenant_hint, "update_call")
        if not headers:
            return False
        try:
            body: dict[str, Any] = {
                "tenant_id": tenant_id,
                "status": status,
                "duration_seconds": duration_seconds,
            }
            sanitized_transcript = _sanitize_transcript(transcript)
            if sanitized_transcript is not None:
                body["transcript"] = sanitized_transcript
            if transcript_text:
                body["transcript_text"] = transcript_text
            if ai_summary:
                body["ai_summary"] = ai_summary
            if transfer_reason:
                body["transfer_reason"] = transfer_reason
            if ended_at:
                body["ended_at"] = ended_at
            if metrics is not None:
                body["metrics"] = metrics
            if murojat_id:
                if _is_object_id(murojat_id):
                    body["murojat_id"] = murojat_id
                else:
                    logger.warning(
                        "update_call: dropping murojat_id %r (not a 24-hex ObjectId)",
                        murojat_id,
                    )
            if metadata is not None:
                body["metadata"] = metadata

            resp = await self._http.patch(
                f"{self._base_url}/api/v1/internal/calls/{call_db_id}",
                json=body,
                headers=headers,
            )
            if resp.status_code not in (200, 204):
                logger.warning("update_call got %s: %s", resp.status_code, resp.text[:200])
            return resp.status_code in (200, 204)
        except Exception:
            logger.exception("update_call failed for %s", call_db_id)
        return False

    # ------------------------------------------------------------------
    # Caller history
    # ------------------------------------------------------------------

    async def get_caller_history(
        self,
        *,
        phone: str,
        tenant_id: str,
        tenant_slug: str | None = None,
        exclude_call_id: str | None = None,
    ) -> CallerHistory:
        """GET /api/v1/internal/caller-history — fetch previous call info.

        Spec requires `caller_phone` (not `phone`) as the query param name.
        OpenAPI does not document the query params, but the server regex
        for phone matches the /internal/calls pattern, so we normalize.
        """
        if not tenant_id:
            logger.error("get_caller_history: tenant_id is required")
            return CallerHistory()
        tenant_hint = self._resolve_tenant_hint(tenant_id, tenant_slug)
        headers = self._require_tenant_headers(tenant_hint, "get_caller_history")
        if not headers:
            return CallerHistory()
        try:
            params: dict[str, str] = {
                "caller_phone": _normalize_phone(phone),
                "tenant_id": tenant_id,
            }
            if exclude_call_id:
                if _is_object_id(exclude_call_id):
                    params["exclude_call_id"] = exclude_call_id
                else:
                    logger.warning(
                        "get_caller_history: dropping exclude_call_id %r (not a 24-hex ObjectId)",
                        exclude_call_id,
                    )
            resp = await self._http.get(
                f"{self._base_url}/api/v1/internal/caller-history",
                params=params,
                headers=headers,
            )
            if resp.status_code == 200:
                raw = resp.json().get("data", {})
                prev_calls = raw.get("previous_calls") or raw.get("total_calls") or 0
                return CallerHistory(
                    has_history=prev_calls > 0,
                    total_calls=prev_calls,
                    last_topic=raw.get("last_topic") or "",
                    last_summary=raw.get("last_summary") or "",
                    last_call_status=raw.get("last_resolved") or raw.get("last_call_status") or "",
                    last_language=(
                        raw.get("last_language")
                        or raw.get("lastLanguage")
                        or (raw.get("metadata") or {}).get("language")
                        or ""
                    ),
                )
        except Exception:
            logger.exception("get_caller_history failed for %s", phone)
        return CallerHistory()

    # ------------------------------------------------------------------
    # Operator handoff
    # ------------------------------------------------------------------

    async def request_operator(
        self,
        *,
        call_db_id: str,
        tenant_id: str,
        tenant_slug: str | None = None,
        transfer_reason: str,
        ai_summary: str,
        room_name: str,
        agent_identity: str,
    ) -> bool:
        """POST /api/v1/internal/calls/:id/operator — request operator transfer.

        Path `id` must be a 24-hex ObjectId. Body is not formally documented
        in openapi.json but the express route expects tenant_id alongside
        handoff metadata.
        """
        if not tenant_id:
            logger.error("request_operator: tenant_id is required")
            return False
        if not _is_object_id(call_db_id):
            logger.error("request_operator: call_db_id %r is not a 24-hex ObjectId", call_db_id)
            return False
        tenant_hint = self._resolve_tenant_hint(tenant_id, tenant_slug)
        headers = self._require_tenant_headers(tenant_hint, "request_operator")
        if not headers:
            return False
        try:
            body: dict[str, Any] = {
                "tenant_id": tenant_id,
                "transfer_reason": transfer_reason,
                "ai_summary": ai_summary,
                "room_name": room_name,
                "agent_identity": agent_identity,
                "metadata": {"handoff_mode": "web_livekit_only"},
            }
            resp = await self._http.post(
                f"{self._base_url}/api/v1/internal/calls/{call_db_id}/operator",
                json=body,
                headers=headers,
            )
            return resp.status_code in (200, 201)
        except Exception:
            logger.exception("request_operator failed for %s", call_db_id)
        return False

    async def get_operator_status(
        self,
        *,
        call_db_id: str,
        tenant_id: str,
        tenant_slug: str | None = None,
    ) -> OperatorStatus:
        """GET /api/v1/internal/calls/:id/operator/status — poll operator status.

        Path `id` must be a 24-hex ObjectId per the openapi path schema.
        """
        if not tenant_id:
            logger.error("get_operator_status: tenant_id is required")
            return OperatorStatus()
        if not _is_object_id(call_db_id):
            logger.error("get_operator_status: call_db_id %r is not a 24-hex ObjectId", call_db_id)
            return OperatorStatus()
        tenant_hint = self._resolve_tenant_hint(tenant_id, tenant_slug)
        headers = self._require_tenant_headers(tenant_hint, "get_operator_status")
        if not headers:
            return OperatorStatus()
        try:
            params: dict[str, str] = {"tenant_id": tenant_id}
            resp = await self._http.get(
                f"{self._base_url}/api/v1/internal/calls/{call_db_id}/operator/status",
                params=params,
                headers=headers,
            )
            if resp.status_code == 200:
                raw = resp.json().get("data", {})
                return OperatorStatus(
                    status=raw.get("status", "unknown"),
                    operator_id=raw.get("operatorId") or raw.get("operator_id"),
                    operator_name=raw.get("operator_name"),
                )
        except Exception:
            logger.exception("get_operator_status failed for %s", call_db_id)
        return OperatorStatus()

    # ------------------------------------------------------------------
    # Murojaat (appeal)
    # ------------------------------------------------------------------

    async def submit_murojaat(
        self,
        *,
        tenant_id: str,
        tenant_slug: str | None = None,
        content: str,
        full_name: str,
        age: int,
        region: str,
        district: str,
        position: str = "Fuqaro",
        murojaat_id: str = "",
        call_id: str | None = None,
        caller_phone: str | None = None,
        turi: str = "ariza",
    ) -> MurojaatResult:
        """POST /api/v1/murojatlar/voice-assistant — submit a citizen appeal."""
        tenant_hint = self._resolve_tenant_hint(tenant_id, tenant_slug)
        tenant_headers = self._require_tenant_headers(tenant_hint, "submit_murojaat")
        if not tenant_headers:
            return MurojaatResult(success=False, error="Missing tenant API key")
        try:
            import uuid

            now_iso = (
                datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.")
                + f"{datetime.now(timezone.utc).microsecond // 1000:03d}Z"
            )

            metadata: dict[str, Any] = {
                "source": "navai-voice-agent",
                "created_at": now_iso,
                "version": "1.0",
                "validated": True,
            }
            if call_id:
                metadata["call_id"] = call_id
            if caller_phone:
                metadata["caller_phone"] = caller_phone

            murojaat_qiluvchi: dict[str, Any] = {
                "ism_familiya": full_name,
                "yosh": age,
                "viloyat": region,
                "tuman": district,
                "lavozim": position or "Fuqaro",
            }
            if caller_phone:
                # Strip leading + for backend's expected format (998XXXXXXXXX).
                # Upstream we accept either +998... or 998...; normalize first
                # so we don't send garbage like "unknown" through lstrip.
                normalized = _normalize_phone(caller_phone)
                if normalized != "unknown":
                    murojaat_qiluvchi["telefon"] = normalized.lstrip("+")

            body = {
                "tenant_id": tenant_id,
                "murojaat_id": murojaat_id or str(uuid.uuid4())[:8],
                "sana": now_iso,
                "murojaat_qiluvchi": murojaat_qiluvchi,
                "murojaat": {
                    "turi": turi,
                    "mazmuni": content,
                },
                "metadata": metadata,
            }

            # openapi.json declares this route as bearerAuth. Use JWT when
            # configured; fall back to the default x-api-key header so
            # deployments where the server also accepts the internal api
            # key keep working.
            auth_headers = self._bearer_headers()
            if "Authorization" not in auth_headers:
                auth_headers = tenant_headers
            resp = await self._http.post(
                f"{self._base_url}/api/v1/murojatlar/voice-assistant",
                json=body,
                headers=auth_headers,
            )
            if resp.status_code in (200, 201):
                data = resp.json()
                inner = data.get("data", {}) if isinstance(data.get("data"), dict) else {}
                # The PATCH /internal/calls/:id body requires `murojat_id`
                # to be a 24-hex ObjectId, so prefer the Mongo _id/id over
                # the short user-visible murojaat_id.
                resolved = (
                    inner.get("_id")
                    or inner.get("id")
                    or data.get("_id")
                    or data.get("id")
                    or inner.get("murojaat_id")
                    or data.get("murojaat_id")
                )
                return MurojaatResult(success=True, murojaat_id=resolved)
            return MurojaatResult(
                success=False,
                error=f"HTTP {resp.status_code}: {resp.text}",
            )
        except Exception as exc:
            logger.exception("submit_murojaat failed")
            return MurojaatResult(success=False, error=str(exc))

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    async def check_health(self) -> bool:
        """Liveness probe against the real API, not the frontend SPA.

        `/health` on this deployment is caught by the frontend catch-all and
        returns the SPA HTML with status 200 even when the backend is down.
        Instead probe an API route that is guaranteed to return JSON — any
        JSON response (including 400/401/404) means the API server itself
        is reachable.
        """
        try:
            resp = await self._http.get(f"{self._base_url}/api/v1/internal/config")
            content_type = resp.headers.get("content-type", "")
            return "application/json" in content_type
        except Exception:
            logger.exception("check_health failed")
        return False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close the underlying httpx client."""
        await self._http.aclose()


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_instance: PlatformClient | None = None


def get_platform_client() -> PlatformClient:
    """Return (or create) the module-level singleton PlatformClient."""
    global _instance
    if _instance is None:
        _instance = PlatformClient()
    return _instance
