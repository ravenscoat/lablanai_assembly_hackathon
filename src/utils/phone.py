"""
Phone number extraction from LiveKit SIP metadata.
Ported from navai-voice-agent/src/core/entrypoint_utils.py
"""

from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger(__name__)
_TRUE_VALUES = {"1", "true", "yes", "on"}


def _single_tenant_mode_enabled() -> bool:
    raw = os.getenv("SINGLE_TENANT_MODE", "false")
    return raw.strip().lower() in _TRUE_VALUES


def extract_called_phone(ctx) -> str | None:
    """
    Extract the called phone number from room/job metadata.

    Priority:
    1. Job metadata (dispatch rules)
    2. Room metadata
    3. AGENT_PHONE_NUMBER env var fallback (single-tenant mode only)

    NOTE: Room name encodes the CALLER phone (not the agent's), so we skip it.
    """
    try:
        # Method 1: Job/dispatch metadata
        if hasattr(ctx, "job") and ctx.job:
            dispatch_meta = None
            for attr in ("metadata", "dispatch_metadata", "agent_metadata"):
                val = getattr(ctx.job, attr, None)
                if val:
                    dispatch_meta = val
                    break

            if dispatch_meta:
                phone = _extract_phone_from_metadata(dispatch_meta)
                if phone:
                    logger.info(f"Extracted CALLED phone from job metadata: {phone}")
                    return phone

        # Method 2: Room metadata
        if hasattr(ctx, "room") and ctx.room and ctx.room.metadata:
            phone = _extract_phone_from_metadata(ctx.room.metadata)
            if phone:
                logger.info(f"Extracted phone from room metadata: {phone}")
                return phone

        # Method 3: Environment variable fallback (single-tenant mode only)
        if _single_tenant_mode_enabled():
            agent_phone = os.getenv("AGENT_PHONE_NUMBER")
            if agent_phone:
                logger.info(f"Using phone from AGENT_PHONE_NUMBER env: {agent_phone}")
                return agent_phone
        else:
            logger.info(
                "Skipping AGENT_PHONE_NUMBER fallback because SINGLE_TENANT_MODE is disabled"
            )

    except Exception as e:
        logger.error(f"Error extracting phone: {e}")

    return None


def extract_caller_phone(ctx) -> str | None:
    """Extract the caller's phone number from SIP participant attributes.

    Returns one of:
      - "+998..." (E.164 phone from PSTN caller)
      - "ext:NNN" (internal extension, 3-6 digits)
      - None (nothing recognizable)
    """
    try:
        if hasattr(ctx, "room") and ctx.room:
            for participant in ctx.room.remote_participants.values():
                if hasattr(participant, "attributes"):
                    phone = participant.attributes.get("sip.phoneNumber")
                    if phone:
                        return phone
        # Fallback: parse room name (SIP usually encodes caller in the room).
        # Formats observed:
        #   "+998932412054_7m4SYfXokdrR"  -> PSTN phone before underscore
        #   "701_tjTrKh8vLwjM"            -> internal extension before underscore
        if hasattr(ctx, "room") and ctx.room and ctx.room.name:
            name = ctx.room.name.lstrip("_")
            head = name.split("_", 1)[0]
            if head.startswith("+") and head[1:].isdigit():
                return head
            # Internal extensions: 3–6 digits, no leading "+"
            if head.isdigit() and 3 <= len(head) <= 6:
                return f"ext:{head}"
    except Exception as e:
        logger.debug(f"Could not extract caller phone: {e}")
    return None


def _extract_phone_from_metadata(metadata) -> str | None:
    """Parse metadata (str or dict) and extract phone number."""
    try:
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        if not isinstance(metadata, dict):
            return None

        # Unwrap nested metadata
        if "metadata" in metadata and isinstance(metadata["metadata"], dict):
            metadata = metadata["metadata"]

        for key in ("called_phone", "destination_phone", "agent_phone", "phone"):
            if key in metadata and metadata[key]:
                return metadata[key]
    except Exception:
        pass
    return None
