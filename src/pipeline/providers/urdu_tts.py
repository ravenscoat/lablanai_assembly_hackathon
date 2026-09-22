"""
Urdu TTS (Text-to-Speech) provider — Azure Speech backend.

Thin wrapper around `livekit.plugins.azure.TTS` so the existing `urdu_tts`
factory branch (`src/pipeline/voice_factory.py`) keeps a stable class shape
while we delegate synthesis to Azure Cognitive Services. Azure ships several
neural Urdu voices out of the box:

  * ur-PK-AsadNeural   (male, Pakistani Urdu)
  * ur-PK-UzmaNeural   (female, Pakistani Urdu) — default
  * ur-IN-GulNeural    (female, Indian Urdu)
  * ur-IN-SalmanNeural (male, Indian Urdu)

Auth comes from `AZURE_SPEECH_KEY` plus either `AZURE_SPEECH_REGION` or
`AZURE_SPEECH_ENDPOINT` (read by the plugin itself). Per-tenant voice
selection flows through `voice.tts_voice_id` / `voice.voices` in the YAML.
Speed multipliers map to SSML `prosody@rate`.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from livekit.plugins.azure import TTS as AzureTTS
from livekit.plugins.azure.tts import ProsodyConfig

from observability.network_topology import register_service_route

logger = logging.getLogger(__name__)

# Sensible defaults for Pakistani Urdu (the hostel demo target). Operators
# can override per-tenant via voice.tts_voice_id / voice.voices.
_DEFAULT_URDU_VOICE = "ur-PK-UzmaNeural"


def _resolve_voice(voice: str | None) -> str:
    """Pick a concrete Azure voice id, with sensible Urdu defaults."""
    if voice:
        return voice
    return os.getenv("URDU_VOICE_ID") or _DEFAULT_URDU_VOICE


def _resolve_speech_key() -> str | None:
    return (
        os.getenv("AZURE_SPEECH_KEY")
        or os.getenv("URDU_TTS_API_KEY")
    )


def _resolve_speech_region() -> str | None:
    return os.getenv("AZURE_SPEECH_REGION")


def _resolve_speech_endpoint() -> str | None:
    """Return an explicit synthesis endpoint, or None to let the plugin build it.

    The Azure plugin builds the correct URL from ``speech_region`` when no
    endpoint override is set:

        https://<region>.tts.speech.microsoft.com/cognitiveservices/v1

    We only honor ``URDU_TTS_URL`` here when it looks like a full synthesis URL
    (path contains ``cognitiveservices``). The Azure portal's "Endpoint" value
    (``https://<region>.api.cognitive.microsoft.com/``) is a resource endpoint,
    NOT the synthesis endpoint — passing it would point the plugin at the
    wrong host and silently produce empty audio.
    """
    explicit = os.getenv("URDU_TTS_URL", "").strip()
    if explicit and "cognitiveservices" in explicit:
        return explicit
    return None


def _build_prosody(speed: float) -> ProsodyConfig | None:
    """Translate a 0.5–2.0 speed multiplier into an Azure prosody@rate."""
    if not speed or speed == 1.0:
        return None
    rate = max(0.5, min(2.0, float(speed)))
    return ProsodyConfig(rate=rate)


class UrduTTS(AzureTTS):
    """Urdu TTS backed by Azure Cognitive Services neural voices."""

    def __init__(
        self,
        *,
        voice: str | None = None,
        language: str = "ur-PK",
        sample_rate: int = 24000,
        num_channels: int = 1,
        speed: float = 1.0,
        base_url: str | None = None,
        **_unused: Any,
    ) -> None:
        resolved_voice = _resolve_voice(voice)
        speech_key = _resolve_speech_key()
        speech_region = _resolve_speech_region()
        speech_endpoint = _resolve_speech_endpoint() if not base_url else base_url

        # The Azure plugin falls back to AZURE_SPEECH_ENDPOINT from the env when
        # we don't pass one. The Azure portal's "Endpoint" field is the resource
        # endpoint (https://<region>.api.cognitive.microsoft.com/), NOT the TTS
        # synthesis endpoint. If our resolver returned nothing but the env has
        # the wrong resource endpoint, we'd silently hit a 200-but-empty path.
        # Force the synthesis URL when we know the region.
        env_endpoint = os.getenv("AZURE_SPEECH_ENDPOINT", "")
        if not speech_endpoint and speech_region and (
            not env_endpoint or "cognitiveservices" not in env_endpoint
        ):
            speech_endpoint = (
                f"https://{speech_region}.tts.speech.microsoft.com/cognitiveservices/v1"
            )

        if not speech_key and not speech_endpoint:
            raise RuntimeError(
                "Urdu TTS requires AZURE_SPEECH_KEY plus AZURE_SPEECH_REGION (or "
                "a synthesis AZURE_SPEECH_ENDPOINT containing 'cognitiveservices') "
                "— set them in your .env."
            )

        prosody = _build_prosody(speed)

        kwargs: dict[str, Any] = {
            "voice": resolved_voice,
            "language": language,
            "sample_rate": sample_rate,
            "speech_key": speech_key,
            "speech_region": speech_region,
            "speech_endpoint": speech_endpoint,
        }
        if prosody is not None:
            kwargs["prosody"] = prosody

        super().__init__(**kwargs)

        # Telephony timing callback (set by main.py); kept for parity.
        self._on_tts_duration = None
        self._language_label = language
        self._speed = speed
        self._num_channels = num_channels

        topology_target = speech_endpoint or (
            f"https://{speech_region}.tts.speech.microsoft.com" if speech_region else "azure-tts"
        )
        register_service_route(
            "urdu_tts",
            topology_target,
            provider="azure_speech",
            metadata={
                "voice": resolved_voice,
                "language": language,
                "speed": speed,
            },
        )
        logger.info(
            "UrduTTS (Azure) initialized: endpoint=%s, voice=%s, language=%s, speed=%s",
            topology_target,
            resolved_voice,
            language,
            speed,
        )
