from __future__ import annotations

from typing import Any

from config.schema import TenantConfig


def build_session_update(
    config: TenantConfig, tools: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Translate a validated tenant into AssemblyAI's session.update event."""
    if not config.assemblyai.enabled:
        raise ValueError(f"AssemblyAI is not enabled for tenant {config.tenant.slug!r}")

    session = {
        "system_prompt": config.personality.system_prompt.strip(),
        "greeting": config.personality.greeting,
        "input": {
            "format": {"encoding": "audio/pcm"},
            "keyterms": config.assemblyai.keyterms,
            "turn_detection": {
                "vad_threshold": config.assemblyai.vad_threshold,
                "min_silence": config.assemblyai.min_silence_ms,
                "max_silence": config.assemblyai.max_silence_ms,
                "interrupt_response": config.assemblyai.interrupt_response,
            },
        },
        "output": {
            "voice": config.assemblyai.voice,
            "format": {"encoding": "audio/pcm"},
        },
    }
    if tools:
        session["tools"] = tools
    return {
        "type": "session.update",
        "session": session,
    }
