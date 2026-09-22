"""
Urdu STT (Speech-to-Text) provider — Speechmatics real-time backend.

Thin wrapper around `livekit.plugins.speechmatics.STT` so the existing
`urdu_stt` factory branch (`src/pipeline/voice_factory.py`) keeps a stable
class shape while we delegate the heavy lifting to Speechmatics' streaming
real-time API. Speechmatics natively supports Urdu (`ur`), so no extra
locale mapping is needed beyond stripping the `-PK` suffix.

Auth comes from `SPEECHMATICS_API_KEY` (read by the plugin itself) — see
`.env.example`. The plugin already exposes streaming + interim transcripts
which the agent uses for barge-in.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from livekit.plugins.speechmatics import STT as SpeechmaticsSTT

from observability.network_topology import register_service_route

logger = logging.getLogger(__name__)

_SPEECHMATICS_RT_DEFAULT = "wss://eu2.rt.speechmatics.com/v2"


def _resolve_language(language: str | None) -> str:
    """Normalize an incoming locale (e.g. ``ur-PK``) to a Speechmatics code (``ur``)."""
    if not language:
        return "ur"
    return language.split("-")[0].lower()


def _resolve_api_key() -> str | None:
    """Pick the first present Speechmatics API key from the env."""
    return (
        os.getenv("SPEECHMATICS_API_KEY")
        or os.getenv("SPEECHMATICS_STT_KEY")
        or os.getenv("URDU_STT_API_KEY")
    )


class UrduSTT(SpeechmaticsSTT):
    """Urdu STT backed by Speechmatics real-time transcription."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        language: str = "ur-PK",
        sample_rate: int = 16000,
        vad: Any = None,
        **_unused: Any,
    ) -> None:
        resolved_key = api_key or _resolve_api_key()
        if not resolved_key:
            raise RuntimeError(
                "Urdu STT requires SPEECHMATICS_API_KEY (or SPEECHMATICS_STT_KEY) — "
                "set it in your .env."
            )

        sm_language = _resolve_language(language)
        resolved_base_url = base_url or os.getenv("URDU_STT_URL") or _SPEECHMATICS_RT_DEFAULT

        super().__init__(
            api_key=resolved_key,
            base_url=resolved_base_url,
            language=sm_language,
            enable_partials=True,
            sample_rate=sample_rate,
        )

        # Telephony timing callbacks (set by main.py); kept for parity with
        # the other providers — Speechmatics already streams partials so we
        # rely on the session's tracker hooks for latency reporting.
        self._first_audio_signaled = False
        self._on_first_audio = None
        self._on_stt_duration = None
        self._vad = vad
        self._language_label = language

        register_service_route(
            "urdu_stt",
            resolved_base_url,
            provider="speechmatics",
            metadata={"language": sm_language, "session_locale": language},
        )
        logger.info(
            "UrduSTT (Speechmatics) initialized: url=%s, language=%s, sample_rate=%d",
            resolved_base_url,
            sm_language,
            sample_rate,
        )
