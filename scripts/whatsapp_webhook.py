"""
WhatsApp Cloud API → LiveKit webhook bridge.

Receives Meta's call-connect webhook (with the caller's SDP offer), then calls
`ConnectorService.accept_whatsapp_call` on LiveKit, which pre-accepts the call,
dispatches the hashim-girls-hostel-agent into a LiveKit room, and answers the
WhatsApp call once WebRTC is established.

Endpoints:
    GET  /webhook    — Meta webhook verification (returns hub.challenge)
    POST /webhook    — Meta event callbacks (calls, messages, status updates)
    GET  /healthz    — liveness probe

Required env (in repo .env):
    WHATSAPP_VERIFY_TOKEN     — any string you choose; paste the same in Meta UI
    WHATSAPP_APP_SECRET       — Meta app secret; used to verify X-Hub-Signature-256
    WHATSAPP_API_KEY          — Meta Cloud API access token (System User token)
    WHATSAPP_PHONE_NUMBER_ID  — the business phone number id from Meta
    WHATSAPP_CLOUD_API_VERSION — e.g. "23.0"
    LIVEKIT_URL / LIVEKIT_API_KEY / LIVEKIT_API_SECRET

Run:
    PYTHONPATH=src .venv/bin/python scripts/whatsapp_webhook.py
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import aiohttp
from aiohttp import web
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from livekit import api
from livekit.protocol.agent_dispatch import RoomAgentDispatch
from livekit.protocol.connector_whatsapp import AcceptWhatsAppCallRequest
from livekit.protocol.rtc import SessionDescription

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("wa-webhook")


VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN", "")
APP_SECRET = os.getenv("WHATSAPP_APP_SECRET", "")
WA_API_KEY = os.getenv("WHATSAPP_API_KEY", "")
WA_PHONE_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID", "")
WA_API_VERSION = os.getenv("WHATSAPP_CLOUD_API_VERSION", "23.0")
AGENT_NAME = os.getenv("AGENT_NAME", "hashim-girls-hostel-agent")
PORT = int(os.getenv("WHATSAPP_WEBHOOK_PORT", "8090"))

# Default to PK so Meta routes media through the right regional cluster.
DEFAULT_DESTINATION_COUNTRY = os.getenv("WHATSAPP_DESTINATION_COUNTRY", "PK")


def _verify_meta_signature(body: bytes, signature_header: str | None) -> bool:
    """Verify Meta's X-Hub-Signature-256 HMAC. Skipped when APP_SECRET is unset (dev)."""
    if not APP_SECRET:
        logger.warning(
            "WHATSAPP_APP_SECRET not set — skipping signature verification (dev only)."
        )
        return True
    if not signature_header or not signature_header.startswith("sha256="):
        return False
    expected = "sha256=" + hmac.new(APP_SECRET.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature_header)


async def _accept_call(payload_call: dict[str, Any]) -> str | None:
    """Hand the WhatsApp SDP offer to LiveKit and dispatch the hostel agent.

    Returns the LiveKit room name on success, None on failure.
    """
    call_id = payload_call.get("id")
    from_number = payload_call.get("from", "")
    session = payload_call.get("session") or {}
    sdp_text = session.get("sdp", "")
    sdp_type = session.get("sdp_type", "offer")

    if not call_id or not sdp_text:
        logger.error("malformed call payload (call_id=%s, has_sdp=%s)", call_id, bool(sdp_text))
        return None

    logger.info("accepting WhatsApp call %s from %s", call_id, from_number)

    lkapi = api.LiveKitAPI()
    try:
        req = AcceptWhatsAppCallRequest(
            whatsapp_phone_number_id=WA_PHONE_ID,
            whatsapp_api_key=WA_API_KEY,
            whatsapp_cloud_api_version=WA_API_VERSION,
            whatsapp_call_id=call_id,
            sdp=SessionDescription(type=sdp_type, sdp=sdp_text),
            agents=[RoomAgentDispatch(agent_name=AGENT_NAME)],
            participant_identity=f"wa:{from_number}",
            participant_name=from_number,
            participant_attributes={
                "wa.caller_phone": from_number,
                "wa.call_id": call_id,
            },
            destination_country=DEFAULT_DESTINATION_COUNTRY,
            wait_until_answered=False,
        )
        res = await lkapi.connector.accept_whatsapp_call(req)
        logger.info(
            "✅ call %s accepted; room=%s agent=%s caller=%s",
            call_id,
            res.room_name,
            AGENT_NAME,
            from_number,
        )
        return res.room_name
    except Exception as exc:
        logger.exception("accept_whatsapp_call failed for %s: %s", call_id, exc)
        return None
    finally:
        await lkapi.aclose()


async def _handle_calls_change(value: dict[str, Any]) -> None:
    """Process a single 'calls' change entry from Meta's webhook."""
    for call in value.get("calls", []) or []:
        event = call.get("event")
        call_id = call.get("id")
        logger.info("wa call event=%s id=%s", event, call_id)
        if event == "connect":
            await _accept_call(call)
        elif event in {"terminate", "reject"}:
            logger.info("call %s ended (event=%s)", call_id, event)
        # status/permission/etc. just get logged.


# --- HTTP handlers ---------------------------------------------------------


async def handle_verify(request: web.Request) -> web.Response:
    """Meta sends GET /webhook?hub.mode=subscribe&hub.verify_token=...&hub.challenge=..."""
    params = request.query
    if (
        params.get("hub.mode") == "subscribe"
        and params.get("hub.verify_token") == VERIFY_TOKEN
    ):
        challenge = params.get("hub.challenge", "")
        logger.info("✅ Meta verification OK")
        return web.Response(text=challenge)
    logger.warning("❌ Meta verification failed; token mismatch or missing params")
    return web.Response(status=403, text="verification failed")


async def handle_webhook(request: web.Request) -> web.Response:
    """Meta event delivery — must respond 200 quickly so Meta doesn't retry."""
    body = await request.read()
    if not _verify_meta_signature(body, request.headers.get("X-Hub-Signature-256")):
        logger.error("invalid HMAC signature; rejecting")
        return web.Response(status=401, text="invalid signature")

    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        logger.error("invalid JSON: %s", exc)
        return web.Response(status=400, text="bad json")

    if payload.get("object") != "whatsapp_business_account":
        return web.Response(text="ignored")

    for entry in payload.get("entry", []) or []:
        for change in entry.get("changes", []) or []:
            field = change.get("field")
            value = change.get("value") or {}
            if field == "calls":
                # Don't await here — Meta wants <5s response; we hand it to a task.
                asyncio.create_task(_handle_calls_change(value))
            else:
                logger.debug("non-calls webhook field=%s", field)

    return web.Response(text="ok")


async def handle_health(_: web.Request) -> web.Response:
    return web.Response(text=json.dumps({"status": "healthy"}))


def _check_env() -> None:
    missing = [k for k in ("WHATSAPP_VERIFY_TOKEN", "WHATSAPP_API_KEY", "WHATSAPP_PHONE_NUMBER_ID") if not os.getenv(k)]
    if missing:
        logger.warning(
            "Missing recommended env vars %s — webhook will run but calls won't connect.",
            missing,
        )
    for k in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"):
        if not os.getenv(k):
            raise SystemExit(f"required env var {k} is not set")


def main() -> None:
    _check_env()
    app = web.Application()
    app.router.add_get("/webhook", handle_verify)
    app.router.add_post("/webhook", handle_webhook)
    app.router.add_get("/healthz", handle_health)

    logger.info(
        "WhatsApp → LiveKit bridge listening on :%d (agent=%s, wa_phone_id=%s)",
        PORT,
        AGENT_NAME,
        WA_PHONE_ID or "<unset>",
    )
    web.run_app(app, host="0.0.0.0", port=PORT, print=lambda *_: None)


if __name__ == "__main__":
    main()
