"""LiveKit transport bridge for AssemblyAI's full Voice Agent API."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import signal
from typing import Any

import websockets
from dotenv import load_dotenv
from livekit import api, rtc

from config.registry import get_registry
from relaydesk import RelayDeskStore, create_store

from .session_config import build_session_update
from .tools import RELAYDESK_TOOLS, RelayDeskToolDispatcher

VOICE_AGENT_URL = "wss://agents.assemblyai.com/v1/ws"
SAMPLE_RATE = 24_000
CHANNELS = 1
logger = logging.getLogger("relaydesk.assemblyai")


def required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


def make_livekit_token(room_name: str, identity: str = "relaydesk-voice-agent") -> str:
    return (
        api.AccessToken(required_env("LIVEKIT_API_KEY"), required_env("LIVEKIT_API_SECRET"))
        .with_identity(identity)
        .with_name("RelayDesk AssemblyAI Agent")
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=room_name,
                can_publish=True,
                can_subscribe=True,
                can_publish_data=True,
            )
        )
        .to_jwt()
    )


class AssemblyAIBridge:
    def __init__(
        self,
        audio_source: rtc.AudioSource,
        session_event: dict[str, Any],
        dispatcher: RelayDeskToolDispatcher,
        session_id: str,
        store: RelayDeskStore,
    ) -> None:
        self.audio_source = audio_source
        self.session_event = session_event
        self.ws: Any = None
        self.ready = asyncio.Event()
        self.dispatcher = dispatcher
        self.session_id = session_id
        self.store = store
        self.pending_tool_results: list[dict[str, str]] = []

    async def run(self, microphone: rtc.RemoteAudioTrack) -> None:
        headers = {"Authorization": f"Bearer {required_env('ASSEMBLYAI_API_KEY')}"}
        async with websockets.connect(VOICE_AGENT_URL, additional_headers=headers) as ws:
            self.ws = ws
            await ws.send(json.dumps(self.session_event))
            await asyncio.gather(self._forward_audio(microphone), self._receive())

    async def _forward_audio(self, microphone: rtc.RemoteAudioTrack) -> None:
        stream = rtc.AudioStream.from_track(
            track=microphone,
            sample_rate=SAMPLE_RATE,
            num_channels=CHANNELS,
        )
        await self.ready.wait()
        async for event in stream:
            await self.ws.send(
                json.dumps(
                    {
                        "type": "input.audio",
                        "audio": base64.b64encode(bytes(event.frame.data)).decode("ascii"),
                    }
                )
            )

    async def _receive(self) -> None:
        async for raw in self.ws:
            event = json.loads(raw)
            event_type = event.get("type")
            if event_type == "session.ready":
                logger.info("AssemblyAI session ready: %s", event.get("session_id"))
                self.ready.set()
            elif event_type == "input.speech.started":
                self.audio_source.clear_queue()
                self.store.record_runtime_event(self.session_id, "speech_started", "caller", {})
            elif event_type == "transcript.user":
                logger.info("Caller: %s", event.get("text", ""))
                self.store.record_runtime_event(
                    self.session_id, "transcript", "caller", {"text": event.get("text", "")}
                )
            elif event_type == "transcript.agent":
                logger.info("Agent: %s", event.get("text", ""))
                self.store.record_runtime_event(
                    self.session_id, "transcript", "agent", {"text": event.get("text", "")}
                )
            elif event_type == "reply.audio":
                pcm = base64.b64decode(event["data"])
                await self.audio_source.capture_frame(
                    rtc.AudioFrame(
                        data=pcm,
                        sample_rate=SAMPLE_RATE,
                        num_channels=CHANNELS,
                        samples_per_channel=len(pcm) // 2,
                    )
                )
            elif event_type == "reply.done":
                if event.get("status") == "interrupted":
                    self.audio_source.clear_queue()
                    self.pending_tool_results.clear()
                    self.store.record_runtime_event(self.session_id, "interrupted", "system", {})
                else:
                    await self._flush_tool_results()
            elif event_type == "tool.call":
                arguments = event.get("arguments") or {}
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                result = self.dispatcher.execute(event.get("name", ""), arguments)
                logger.info("Tool %s result: %s", event.get("name"), result)
                self.pending_tool_results.append(
                    {"call_id": event["call_id"], "result": json.dumps(result)}
                )
                self.store.record_runtime_event(
                    self.session_id,
                    "tool_call",
                    event.get("name", "unknown"),
                    {"arguments": arguments, "result": result},
                    arguments.get("case_id"),
                )
            elif event_type == "session.error":
                self.store.record_runtime_event(
                    self.session_id, "error", "system", {"event": event}
                )
                raise RuntimeError(f"AssemblyAI session error: {event}")

    async def _flush_tool_results(self) -> None:
        pending, self.pending_tool_results = self.pending_tool_results, []
        for item in pending:
            await self.ws.send(json.dumps({"type": "tool.result", **item}))


async def run_bridge() -> None:
    tenant_slug = os.getenv("ASSEMBLYAI_TENANT_SLUG", "relaydesk")
    room_name = os.getenv("ASSEMBLYAI_ROOM_NAME", "relaydesk-demo")
    tenant = get_registry().resolve_by_slug(tenant_slug)
    if tenant is None:
        raise RuntimeError(f"Unknown tenant slug: {tenant_slug}")
    session_event = build_session_update(tenant, RELAYDESK_TOOLS)
    store = create_store()
    dispatcher = RelayDeskToolDispatcher(store)

    room = rtc.Room()
    source = rtc.AudioSource(sample_rate=SAMPLE_RATE, num_channels=CHANNELS)
    track = rtc.LocalAudioTrack.create_audio_track("relaydesk-agent", source)
    active_task: asyncio.Task[None] | None = None

    @room.on("track_subscribed")
    def on_track(track_in: rtc.Track, *_: object) -> None:
        nonlocal active_task
        if track_in.kind != rtc.TrackKind.KIND_AUDIO or active_task is not None:
            return
        bridge = AssemblyAIBridge(source, session_event, dispatcher, room_name, store)
        active_task = asyncio.create_task(bridge.run(track_in))  # type: ignore[arg-type]

    await room.connect(required_env("LIVEKIT_URL"), make_livekit_token(room_name))
    await room.local_participant.publish_track(
        track,
        rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
    )
    logger.info("RelayDesk waiting in LiveKit room %s", room_name)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:
            pass
    try:
        await stop.wait()
    finally:
        if active_task:
            active_task.cancel()
        await room.disconnect()
        await source.aclose()


def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_bridge())


if __name__ == "__main__":
    main()
