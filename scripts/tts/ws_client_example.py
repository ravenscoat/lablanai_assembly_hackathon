#!/usr/bin/env python3
"""Standalone WebSocket TTS client — connect, synthesize, measure latency.

This is the tool referenced by the team's WS doc. Use it to validate the dev
server and measure the real latency numbers (the whole point of moving off
Yandex). It speaks the same protocol as the in-agent provider
(``src/pipeline/providers/navai_ws_tts.py``) and uses aiohttp, the repo's
standard WS/HTTP client — no extra deps.

    export NAVAI_API_KEY=<your-key>
    PYTHONPATH=src python scripts/tts/ws_client_example.py \
        --url wss://<your-tts-gateway-host>/api/tts/synthesize/ws \
        --text "Sample streaming synthesis." \
        --voice <voice-id> --out out.wav

The gateway needs an API key (``X-API-Key`` header) — pass ``--api-key`` or set
``NAVAI_API_KEY``. (Local raw dev servers without auth still work key-less.)

Reports time-to-first-audio-frame (TTFB), total wall time, bytes, and the
server-reported ``first_packet_ms`` / ``audio_seconds`` from the ``done`` frame.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import wave

import aiohttp


def _to_ws_synthesize_url(raw: str) -> str:
    """Same normalization as the provider: bare host:port / http(s):// / full path."""
    raw = (raw or "").strip().rstrip("/")
    if raw.startswith("http://"):
        raw = "ws://" + raw[len("http://") :]
    elif raw.startswith("https://"):
        raw = "wss://" + raw[len("https://") :]
    elif not raw.startswith(("ws://", "wss://")):
        raw = "ws://" + raw
    if raw.endswith(("/synthesize/ws", "/ws/synthesize")):
        return raw
    return raw + "/api/tts/synthesize/ws"


async def synthesize(
    *,
    url: str,
    text: str,
    voice: str,
    mode: str | None,
    reference_text: str | None,
    out_path: str | None,
    timeout: float,
    api_key: str | None,
) -> int:
    ws_url = _to_ws_synthesize_url(url)
    payload: dict[str, str] = {"text": text, "voice_id": voice}
    if reference_text:
        payload["reference_text"] = reference_text
    if mode:
        payload["mode"] = mode

    print(f"→ connecting {ws_url}")
    print(f"→ voice={voice} mode={mode or '(default)'} chars={len(text)}")

    pcm = bytearray()
    sample_rate = 24000
    first_frame_at: float | None = None
    done_event: dict | None = None

    t0 = time.perf_counter()
    timeout_cfg = aiohttp.ClientTimeout(total=None, connect=timeout, sock_connect=timeout)
    # Overall ceiling so a server that streams without ever sending 'done' can't hang us.
    async with asyncio.timeout(max(120.0, timeout * 4)):
        async with aiohttp.ClientSession(timeout=timeout_cfg) as session:
            async with session.ws_connect(
                ws_url,
                headers={"X-API-Key": api_key} if api_key else {},
                timeout=aiohttp.ClientWSTimeout(ws_receive=timeout, ws_close=5),
                max_msg_size=0,
            ) as ws:
                connected_at = time.perf_counter()
                print(f"  connected in {(connected_at - t0) * 1000:.0f} ms")
                await ws.send_str(json.dumps(payload, ensure_ascii=False))

                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.BINARY:
                        if msg.data:
                            if first_frame_at is None:
                                first_frame_at = time.perf_counter()
                                print(
                                    f"  ★ first audio frame at {(first_frame_at - t0) * 1000:.0f} ms "
                                    f"(TTFB), {len(msg.data)} bytes"
                                )
                            pcm += msg.data
                    elif msg.type == aiohttp.WSMsgType.TEXT:
                        try:
                            event = json.loads(msg.data)
                        except (json.JSONDecodeError, ValueError):
                            print(f"  ! ignoring non-JSON control frame: {msg.data[:200]!r}")
                            continue
                        etype = event.get("event")
                        if etype == "start":
                            sample_rate = event.get("sample_rate", sample_rate)
                            print(f"  start: {event}")
                        elif etype == "done":
                            done_event = event
                            break
                        elif etype == "error":
                            print(f"  ✗ server error: {event.get('detail')}", file=sys.stderr)
                            return 2
                        else:
                            print(f"  ! unknown event: {event}")
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        print(f"  ✗ websocket error: {ws.exception()}", file=sys.stderr)
                        return 2
                    elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED):
                        break

    total_ms = (time.perf_counter() - t0) * 1000
    n = len(pcm)
    audio_seconds = n / 2 / sample_rate  # PCM16 mono
    print("\n=== summary ===")
    print(
        f"  TTFB (first frame): {((first_frame_at - t0) * 1000) if first_frame_at else float('nan'):.0f} ms"
    )
    print(f"  total wall time:    {total_ms:.0f} ms")
    print(f"  audio bytes:        {n} ({audio_seconds:.2f} s @ {sample_rate} Hz)")
    if done_event:
        print(f"  server first_packet_ms: {done_event.get('first_packet_ms')}")
        print(f"  server audio_seconds:   {done_event.get('audio_seconds')}")
        print(f"  server chunks:          {done_event.get('chunks')}")

    if not n:
        print("  ✗ no audio received", file=sys.stderr)
        return 3

    if out_path:
        with wave.open(out_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(bytes(pcm))
        print(f"  wrote {out_path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="WebSocket TTS client / latency probe")
    ap.add_argument(
        "--url",
        default=os.getenv("NAVAI_WS_TTS_URL", ""),
    )
    ap.add_argument(
        "--api-key",
        default=os.getenv("NAVAI_API_KEY"),
        help="gateway API key (X-API-Key); or set NAVAI_API_KEY",
    )
    ap.add_argument("--text", default="Salom, bu navai ovozida oqimli sintez.")
    ap.add_argument("--voice", default=os.getenv("NAVAI_WS_VOICE_ID", "navai"))
    ap.add_argument("--mode", default=os.getenv("NAVAI_WS_TTS_MODE") or None, help="local | grpc")
    ap.add_argument("--reference-text", default=None)
    ap.add_argument("--out", default=None, help="write synthesized audio to this WAV path")
    ap.add_argument("--timeout", type=float, default=15.0)
    args = ap.parse_args()

    try:
        return asyncio.run(
            synthesize(
                url=args.url,
                text=args.text,
                voice=args.voice,
                mode=args.mode,
                reference_text=args.reference_text,
                out_path=args.out,
                timeout=args.timeout,
                api_key=args.api_key,
            )
        )
    except (asyncio.TimeoutError, TimeoutError):
        print("✗ timed out waiting for synthesis", file=sys.stderr)
        return 1
    except aiohttp.ClientError as e:
        print(f"✗ connection failed: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
