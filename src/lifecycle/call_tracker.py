"""
Call tracker: creates and updates call records via PlatformClient.
Also fetches caller history for returning caller context.
"""

from __future__ import annotations

import logging

from api.models import CallerHistory
from api.platform_client import get_platform_client

logger = logging.getLogger(__name__)


async def create_call(
    *,
    tenant_id: str,
    tenant_slug: str | None = None,
    caller_phone: str,
    agent_phone: str,
    call_sid: str,
    metadata: dict | None = None,
) -> str | None:
    """Create a call record. Returns call DB ID."""
    client = get_platform_client()
    return await client.create_call(
        tenant_id=tenant_id,
        tenant_slug=tenant_slug,
        call_sid=call_sid,
        caller_phone=caller_phone,
        agent_phone=agent_phone,
        metadata=metadata,
    )


async def update_call(
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
    metrics: dict | None = None,
    metadata: dict | None = None,
    murojat_id: str | None = None,
) -> bool:
    """Update a call record with final data. Empty/None fields are omitted."""
    client = get_platform_client()
    return await client.update_call(
        call_db_id=call_db_id,
        tenant_id=tenant_id,
        tenant_slug=tenant_slug,
        status=status,
        duration_seconds=duration_seconds,
        transcript=transcript,
        transcript_text=transcript_text or None,
        ai_summary=ai_summary or None,
        transfer_reason=transfer_reason or None,
        ended_at=ended_at or None,
        metrics=metrics,
        metadata=metadata,
        murojat_id=murojat_id or None,
    )


async def get_caller_history(
    *,
    phone: str,
    tenant_id: str,
    tenant_slug: str | None = None,
    exclude_call_id: str | None = None,
) -> CallerHistory:
    """Fetch caller's previous call history."""
    client = get_platform_client()
    return await client.get_caller_history(
        phone=phone,
        tenant_id=tenant_id,
        tenant_slug=tenant_slug,
        exclude_call_id=exclude_call_id,
    )
