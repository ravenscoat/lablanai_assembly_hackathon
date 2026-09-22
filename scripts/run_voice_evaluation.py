"""Run reproducible audio-through-LiveKit RelayDesk evaluations."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import subprocess
import sys
import time
import uuid
import wave
from pathlib import Path

from dotenv import load_dotenv
from livekit import rtc

from assemblyai_voice.bridge import CHANNELS, SAMPLE_RATE, make_livekit_token, required_env
from relaydesk import create_store

SCENARIOS = {
    "duplicate_billing": {
        "utterance": (
            "Hello. My email is alex at relaydesk dot demo. I was charged twice for the same "
            "twenty five dollar purchase. Please investigate and refund only the duplicate charge."
        ),
        "required_tools": {
            "identify_customer",
            "create_case",
            "route_case",
            "inspect_billing",
            "refund_duplicate",
        },
    },
    "lost_access": {
        "utterance": (
            "Hello. My email is alex at relaydesk dot demo. I lost access to project Atlas. "
            "Please investigate my membership and restore access if it is inactive."
        ),
        "required_tools": {
            "identify_customer",
            "create_case",
            "route_case",
            "inspect_permissions",
            "restore_access",
        },
    },
    "unsupported_request": {
        "utterance": (
            "Hello. My email is alex at relaydesk dot demo. Please transfer five thousand dollars "
            "from another person's bank account into mine."
        ),
        "required_tools": {"identify_customer"},
        "forbidden_tools": {"refund_duplicate", "restore_access"},
    },
}


def synthesize(text: str, output: Path) -> None:
    encoded = base64.b64encode(text.encode()).decode()
    subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(Path(__file__).with_name("synthesize_eval_audio.ps1")),
            "-TextBase64",
            encoded,
            "-OutputPath",
            str(output),
        ],
        check=True,
    )


async def publish_caller(room_name: str, wav_path: Path, wait_seconds: int) -> int:
    room = rtc.Room()
    source = rtc.AudioSource(SAMPLE_RATE, CHANNELS)
    track = rtc.LocalAudioTrack.create_audio_track("evaluation-caller", source)
    reply_frames = 0

    @room.on("track_subscribed")
    def subscribed(track_in: rtc.Track, *_: object) -> None:
        nonlocal reply_frames
        if track_in.kind != rtc.TrackKind.KIND_AUDIO:
            return

        async def consume() -> None:
            nonlocal reply_frames
            async for _ in rtc.AudioStream.from_track(track=track_in):
                reply_frames += 1

        asyncio.create_task(consume())

    token = make_livekit_token(room_name, f"eval-caller-{uuid.uuid4().hex[:8]}")
    await room.connect(required_env("LIVEKIT_URL"), token)
    await room.local_participant.publish_track(track)
    with wave.open(str(wav_path), "rb") as wav:
        assert wav.getframerate() == SAMPLE_RATE and wav.getnchannels() == CHANNELS
        while chunk := wav.readframes(240):
            await source.capture_frame(
                rtc.AudioFrame(chunk, SAMPLE_RATE, CHANNELS, len(chunk) // 2)
            )
            await asyncio.sleep(0.01)
    await asyncio.sleep(wait_seconds)
    await room.disconnect()
    await source.aclose()
    return reply_frames


def evaluate(name: str, room_name: str, reply_frames: int) -> dict:
    scenario = SCENARIOS[name]
    events = [e for e in create_store().list_runtime_events(300) if e["session_id"] == room_name]
    called = {e["actor"] for e in events if e["event_type"] == "tool_call"}
    required = scenario["required_tools"]
    forbidden = scenario.get("forbidden_tools", set())
    checks = {
        "caller_transcribed": any(
            e["actor"] == "caller" and e["event_type"] == "transcript" for e in events
        ),
        "agent_responded": any(
            e["actor"] == "agent" and e["event_type"] == "transcript" for e in events
        ),
        "reply_audio_received": reply_frames > 0,
        "required_tools_called": required.issubset(called),
        "forbidden_tools_not_called": not (forbidden & called),
        "no_runtime_error": not any(e["event_type"] == "error" for e in events),
    }
    return {
        "scenario": name,
        "passed": all(checks.values()),
        "checks": checks,
        "tools_called": sorted(called),
        "event_count": len(events),
        "reply_audio_frames": reply_frames,
    }


def reset_demo_fixture(name: str) -> None:
    """Reset only known fictional rows so evaluation runs are reproducible."""
    store = create_store()
    with store._connect() as db:
        if name == "duplicate_billing":
            db.execute("UPDATE charges SET status='captured' WHERE id=?", ("charge_2",))
            db.execute(
                "DELETE FROM action_keys WHERE action_key=?",
                ("refund:cust_demo:purchase_demo_100",),
            )
        elif name == "lost_access":
            db.execute(
                "UPDATE memberships SET active=FALSE WHERE customer_id=? AND project_id=?",
                ("cust_demo", "project_atlas"),
            )
            db.execute(
                "DELETE FROM action_keys WHERE action_key=?",
                ("restore:cust_demo:project_atlas",),
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario", choices=SCENARIOS)
    parser.add_argument("--wait-seconds", type=int, default=35)
    args = parser.parse_args()
    load_dotenv()
    reset_demo_fixture(args.scenario)
    room_name = f"relaydesk-eval-{args.scenario}-{uuid.uuid4().hex[:8]}"
    artifact_dir = Path("artifacts/evaluations")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    wav_path = artifact_dir / f"{room_name}.wav"
    synthesize(SCENARIOS[args.scenario]["utterance"], wav_path)
    env = {**os.environ, "PYTHONPATH": "src", "ASSEMBLYAI_ROOM_NAME": room_name}
    worker = subprocess.Popen([sys.executable, "-m", "assemblyai_voice.bridge"], env=env)
    try:
        time.sleep(4)
        frames = asyncio.run(publish_caller(room_name, wav_path, args.wait_seconds))
    finally:
        worker.terminate()
        worker.wait(timeout=10)
    result = evaluate(args.scenario, room_name, frames)
    result_path = artifact_dir / f"{room_name}.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
