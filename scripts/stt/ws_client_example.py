#!/usr/bin/env python3
"""Standalone WebSocket STT client — stream a WAV, print partials + final.

The STT counterpart to ``scripts/tts/ws_client_example.py``. Use it to validate the
dev STT server and measure real partial/final latency. It speaks the same protocol
as the in-agent provider (``src/pipeline/providers/navai_ws_stt.py``) and uses
aiohttp + numpy (already repo deps) — no soundfile/sounddevice needed.

    export NAVAI_API_KEY=<your-key>
    PYTHONPATH=src python scripts/stt/ws_client_example.py \
        --url wss://<your-stt-gateway-host>/api/stt/transcribe/live \
        --wav sample.wav --language ur

The gateway needs an API key (``X-API-Key`` header) — pass ``--api-key`` or set
``NAVAI_API_KEY``. (Local raw dev servers without auth still work key-less.)

Input is any 16-bit PCM WAV (mono/stereo, any sample rate) — it is downmixed to
mono and resampled to 16 kHz before streaming. Reports time-to-first-partial,
total wall time, partial count, and the final transcript.
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
import numpy as np

_TRANSCRIBE_PATH = "/api/stt/transcribe/live"
SAMPLE_RATE = 16000


def _to_ws_transcribe_url(raw: str) -> str:
    """Same normalization as the provider: bare host:port / http(s):// / full path."""
    raw = (raw or "").strip().rstrip("/")
    if raw.startswith("http://"):
        raw = "ws://" + raw[len("http://") :]
    elif raw.startswith("https://"):
        raw = "wss://" + raw[len("https://") :]
    elif not raw.startswith(("ws://", "wss://")):
        raw = "ws://" + raw
    if raw.endswith(("/transcribe/live", "/transcribe/transcribe")):
        return raw
    return raw + _TRANSCRIBE_PATH


def _load_wav_pcm16_16k_mono(path: str) -> bytes:
    """Read a 16-bit PCM WAV → mono, 16 kHz, PCM16 LE bytes."""
    with wave.open(path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    if sampwidth != 2:
        raise ValueError(f"expected 16-bit PCM WAV, got sample width {sampwidth} bytes")
    data = np.frombuffer(raw, dtype="<i2")
    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1).astype("<i2")
    if sr != SAMPLE_RATE and len(data) > 0:
        ratio = SAMPLE_RATE / sr
        new_len = int(len(data) * ratio)
        if new_len > 0:
            idx = np.linspace(0, len(data) - 1, new_len)
            data = np.interp(idx, np.arange(len(data)), data).astype("<i2")
    return data.tobytes()


async def transcribe(
    *,
    url: str,
    wav: str,
    language: str,
    timeout: float,
    realtime: bool,
    frame_ms: int,
    api_key: str | None,
) -> int:
    ws_url = _to_ws_transcribe_url(url)
    pcm = _load_wav_pcm16_16k_mono(wav)
    if not pcm:
        print("  ✗ input WAV decoded to no audio", file=sys.stderr)
        return 3
    audio_seconds = len(pcm) / 2 / SAMPLE_RATE
    frame_bytes = max(2, int(SAMPLE_RATE * frame_ms / 1000)) * 2  # PCM16

    print(f"→ connecting {ws_url}  (subprotocol stt.v1)")
    print(
        f"→ wav={wav}  {audio_seconds:.2f}s @ {SAMPLE_RATE} Hz mono  "
        f"lang={language}  pacing={'real-time' if realtime else 'fast'}"
    )

    first_partial_at: float | None = None
    n_partials = 0
    final_text: str | None = None

    t0 = time.perf_counter()
    timeout_cfg = aiohttp.ClientTimeout(total=None, connect=timeout, sock_connect=timeout)
    # Overall ceiling so a server that never sends 'done' can't hang us.
    async with asyncio.timeout(max(120.0, audio_seconds * 2 + timeout * 4)):
        async with aiohttp.ClientSession(timeout=timeout_cfg) as session:
            async with session.ws_connect(
                ws_url,
                protocols=("stt.v1",),
                headers={"X-API-Key": api_key} if api_key else {},
                timeout=aiohttp.ClientWSTimeout(ws_receive=timeout, ws_close=5),
                max_msg_size=0,
            ) as ws:
                print(f"  connected in {(time.perf_counter() - t0) * 1000:.0f} ms")
                await ws.send_str(
                    json.dumps(
                        {
                            "type": "start",
                            "sample_rate": SAMPLE_RATE,
                            "num_channels": 1,
                            "language": language,
                        }
                    )
                )

                async def _send() -> None:
                    for i in range(0, len(pcm), frame_bytes):
                        await ws.send_bytes(pcm[i : i + frame_bytes])
                        if realtime:
                            await asyncio.sleep(frame_ms / 1000)
                    await ws.send_str(json.dumps({"type": "end"}))

                send_task = asyncio.create_task(_send())
                try:
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            try:
                                ev = json.loads(msg.data)
                            except (json.JSONDecodeError, ValueError):
                                print(f"  ! ignoring non-JSON frame: {msg.data[:200]!r}")
                                continue
                            t = ev.get("type")
                            if t == "speech_start":
                                print("  · speech_start")
                            elif t == "interim":
                                n_partials += 1
                                if first_partial_at is None:
                                    first_partial_at = time.perf_counter()
                                    print(
                                        f"  ★ first partial at "
                                        f"{(first_partial_at - t0) * 1000:.0f} ms"
                                    )
                                print(f"  … {ev.get('text', '')}")
                            elif t == "final":
                                final_text = ev.get("text", "")
                                print(f"  ✓ FINAL: {final_text}")
                            elif t == "warn":
                                print(f"  ! warn: {ev.get('message')}")
                            elif t == "error":
                                print(f"  ✗ server error: {ev.get('message')}", file=sys.stderr)
                                return 2
                            elif t == "done":
                                break
                            else:
                                print(f"  ! unknown event: {ev}")
                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            print(f"  ✗ websocket error: {ws.exception()}", file=sys.stderr)
                            return 2
                        elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED):
                            break
                finally:
                    if not send_task.done():
                        send_task.cancel()
                    try:
                        await send_task
                    except (asyncio.CancelledError, aiohttp.ClientError):
                        pass

    total_ms = (time.perf_counter() - t0) * 1000
    print("\n=== summary ===")
    print(
        f"  time to first partial: "
        f"{((first_partial_at - t0) * 1000) if first_partial_at else float('nan'):.0f} ms"
    )
    print(f"  total wall time:       {total_ms:.0f} ms  (audio {audio_seconds:.2f}s)")
    print(f"  partials received:     {n_partials}")
    print(f"  final transcript:      {final_text!r}")
    if final_text is None:
        print("  ✗ no final transcript received", file=sys.stderr)
        return 3
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="WebSocket STT client / latency probe")
    ap.add_argument(
        "--url",
        default=os.getenv("NAVAI_WS_STT_URL", ""),
    )
    ap.add_argument(
        "--api-key",
        default=os.getenv("NAVAI_API_KEY"),
        help="gateway API key (X-API-Key); or set NAVAI_API_KEY",
    )
    ap.add_argument("--wav", required=True, help="16-bit PCM WAV to transcribe")
    ap.add_argument(
        "--language", default="uz", help="label only; decode is server-fixed (uz/ru/en)"
    )
    ap.add_argument("--timeout", type=float, default=15.0)
    ap.add_argument("--frame-ms", type=int, default=80, help="frame cadence in ms (~80 ms typical)")
    ap.add_argument(
        "--no-realtime",
        dest="realtime",
        action="store_false",
        help="blast audio as fast as possible instead of pacing in real time",
    )
    ap.set_defaults(realtime=True)
    args = ap.parse_args()

    try:
        return asyncio.run(
            transcribe(
                url=args.url,
                wav=args.wav,
                language=args.language,
                timeout=args.timeout,
                realtime=args.realtime,
                frame_ms=args.frame_ms,
                api_key=args.api_key,
            )
        )
    except FileNotFoundError:
        print(f"✗ WAV not found: {args.wav}", file=sys.stderr)
        return 1
    except (asyncio.TimeoutError, TimeoutError):
        print("✗ timed out waiting for transcription", file=sys.stderr)
        return 1
    except aiohttp.ClientError as e:
        print(f"✗ connection failed: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
