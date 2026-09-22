"""
Voice pipeline factory: creates STT, TTS, LLM, and VAD instances from TenantConfig.
Uses navai-shared for STT/TTS (Yandex SpeechKit v3) and LLM factory for language models.

NOTE: LiveKit plugins must be imported at module level (main thread) because
plugin registration raises RuntimeError if called from a worker thread.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from livekit.plugins import google as _google_plugin  # Must import on main thread

try:
    from livekit.plugins import openai as _openai_plugin  # noqa: F401
except ImportError:
    _openai_plugin = None

from livekit.plugins import silero

from config.schema import TenantConfig
from observability.network_topology import register_service_route

logger = logging.getLogger(__name__)

# Language code mapping
_LANG_MAP = {
    "uz": "uz-UZ",
    "ru": "ru-RU",
    "en": "en-US",
    "kk": "kk-KK",
    # Urdu (Pakistan). REQUIRED for an Urdu tenant — without this entry,
    # language="ur" falls back to uz-UZ for both STT and TTS locale.
    "ur": "ur-PK",
}


class VoiceFactory:
    """Creates voice pipeline components from tenant config."""

    @staticmethod
    def create_stt(config: TenantConfig) -> Any:
        """Create STT for the tenant's behavior.language (backward compat)."""
        return VoiceFactory.create_stt_for_language(config, config.behavior.language)

    @staticmethod
    def create_stt_for_language(config: TenantConfig, language: str, *, vad: Any = None) -> Any:
        """Create STT instance for a specific language code (e.g. 'uz' or 'ru').

        `vad` is the session's prewarmed Silero VAD; the streaming `navai_ws`
        provider reuses it for endpointing (other providers ignore it).
        """
        provider = config.voice.stt_provider_for(language).lower()
        locale = _LANG_MAP.get(language)
        if locale is None:
            logger.warning(f"Unknown language code {language!r}, falling back to uz-UZ")
            locale = "uz-UZ"

        if provider == "custom" and os.getenv("DISABLE_CUSTOM_STT", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        ):
            logger.warning(
                "DISABLE_CUSTOM_STT is set; routing tenant 'custom' STT to 'yandex' "
                "instead. Unset the env var once the custom STT host is healthy."
            )
            provider = "yandex"

        logger.info(f"Creating STT: provider={provider}, language={locale}")

        if provider == "yandex":
            from pipeline.providers.yandex_stt import YandexSTT

            return YandexSTT(language=locale)
        elif provider == "navai":
            from pipeline.providers.navai_stt import NavaiSTT

            return NavaiSTT(language=locale.split("-")[0])
        elif provider == "custom":
            from pipeline.providers.custom_stt import CustomSTT

            return CustomSTT(language=locale.split("-")[0])
        elif provider == "navai_ws":
            from pipeline.providers.navai_ws_stt import NavaiWSSTT

            # Streaming WS STT decodes a fixed (Uzbek) language server-side; the
            # language label is informational. Reuses the session's prewarmed VAD
            # for endpointing (falls back to a default Silero VAD if none passed).
            return NavaiWSSTT(language=locale.split("-")[0], vad=vad)
        elif provider == "urdu_stt":
            # TODO(urdu): plug your Urdu STT engine here.
            # UrduSTT is a STUB — see src/pipeline/providers/urdu_stt.py and the
            # README "Urdu integration seams" section. It currently raises
            # NotImplementedError until a real engine is wired.
            from pipeline.providers.urdu_stt import UrduSTT

            return UrduSTT(language=locale, vad=vad)
        else:
            from livekit.agents import inference

            return inference.STT(model="deepgram/nova-3-general")

    @staticmethod
    def create_greeting_stt_for_language_choice(
        config: TenantConfig,
        *,
        fallback_language: str,
    ) -> Any:
        """Create STT for first-turn language choice after greeting.

        This is intentionally separate from normal STT selection so only the
        greeting-phase choice turn can use a different provider.
        """
        provider = (config.voice.greeting_stt_provider or "").lower().strip()
        if not provider:
            return VoiceFactory.create_stt_for_language(config, fallback_language)

        if provider == "gemini":
            from pipeline.providers.gemini_live_stt import GeminiLiveSTT

            languages = config.voice.greeting_stt_languages or ["ru", "uz"]
            fallback_stt = VoiceFactory.create_stt_for_language(config, fallback_language)
            return GeminiLiveSTT(
                language_codes=languages,
                fallback_stt=fallback_stt,
                one_shot=True,
            )

        logger.warning(
            "Unknown greeting_stt_provider=%r; falling back to voice.stt_provider=%r",
            provider,
            config.voice.stt_provider,
        )
        return VoiceFactory.create_stt_for_language(config, fallback_language)

    @staticmethod
    def create_tts(config: TenantConfig) -> Any:
        """Create TTS for the tenant's behavior.language (backward compat)."""
        return VoiceFactory.create_tts_for_language(config, config.behavior.language)

    @staticmethod
    def create_tts_for_language(config: TenantConfig, language: str) -> Any:
        """Create TTS instance for a specific language, picking a voice via VoiceConfig.voice_for."""
        provider = config.voice.tts_provider.lower()
        voice_id = config.voice.voice_for(language)
        speed = config.voice.speed
        locale = _LANG_MAP.get(language)
        if locale is None:
            logger.warning(f"Unknown language code {language!r}, falling back to uz-UZ")
            locale = "uz-UZ"

        logger.info(
            f"Creating TTS: provider={provider}, voice={voice_id}, "
            f"speed={speed}, language={locale}"
        )

        if provider == "yandex":
            from pipeline.providers.yandex_tts import YandexTTS

            return YandexTTS(voice=voice_id, speed=speed, language=locale)
        elif provider == "navai":
            from pipeline.providers.navai_tts import NavaiTTS

            return NavaiTTS(voice=voice_id, speed=speed)
        elif provider == "navai_ws":
            from config.schema import VoiceConfig
            from pipeline.providers.navai_ws_tts import NavaiWSTTS

            # The global tts_voice_id defaults to a Yandex voice ("yulduz") that does
            # not exist on the NavAI WS server. Honor an explicit per-language voice
            # or a deliberately-set tts_voice_id; otherwise pass empty so the provider
            # defaults to its NavAI voice ("navai"). This lets a tenant flip
            # tts_provider -> navai_ws without also remembering to set a voice.
            default_voice_id = VoiceConfig.model_fields["tts_voice_id"].default
            ws_voice = config.voice.voices.get(language) or config.voice.tts_voice_id
            if ws_voice == default_voice_id:
                ws_voice = ""
            return NavaiWSTTS(voice=ws_voice, speed=speed)
        elif provider == "custom":
            from pipeline.providers.custom_tts import CustomTTS

            return CustomTTS(voice=voice_id, speed=speed)
        elif provider == "urdu_tts":
            # TODO(urdu): plug your Urdu TTS engine here.
            # UrduTTS is a STUB — see src/pipeline/providers/urdu_tts.py and the
            # README "Urdu integration seams" section. It currently raises
            # NotImplementedError until a real engine is wired.
            from pipeline.providers.urdu_tts import UrduTTS

            return UrduTTS(voice=voice_id, speed=speed, language=locale)
        else:
            from livekit.agents import inference

            return inference.TTS(
                model="cartesia/sonic-2",
                voice=voice_id,
                language=locale,
            )

    @staticmethod
    def create_llm(config: TenantConfig) -> Any:
        """Create LLM instance based on tenant config."""
        provider = config.llm.provider.lower()

        logger.info(
            f"Creating LLM: provider={provider}, "
            f"model={config.llm.model}, temp={config.llm.temperature}"
        )

        if provider == "gemini":
            return _create_gemini_llm(config)
        elif provider in ("gemini_api", "google_ai_studio"):
            return _create_gemini_api_llm(config)
        elif provider == "openai":
            return _create_openai_llm(config)
        elif provider == "lexantei":
            return _create_lexantei_llm(config)
        elif provider == "custom":
            return _create_custom_llm(config)
        else:
            logger.warning(f"Unknown LLM provider: {provider}, falling back to Gemini")
            return _create_gemini_llm(config)

    @staticmethod
    def load_vad(config: TenantConfig) -> silero.VAD:
        """Load Silero VAD with config-driven thresholds."""
        logger.info(
            f"Loading VAD: silence={config.vad.min_silence_duration}s, "
            f"threshold={config.vad.activation_threshold}"
        )
        return silero.VAD.load(
            min_silence_duration=config.vad.min_silence_duration,
            min_speech_duration=config.vad.min_speech_duration,
            activation_threshold=config.vad.activation_threshold,
            prefix_padding_duration=config.vad.prefix_padding_duration,
            max_buffered_speech=config.vad.max_buffered_speech,
        )


def _create_gemini_llm(config: TenantConfig) -> Any:
    """Create Google Gemini LLM instance via Vertex AI (ADC from attached SA)."""
    from utils.vertex_region import PROJECT, resolve_location, vertex_endpoint

    location = resolve_location(config.llm.model)
    register_service_route(
        "gemini_llm",
        vertex_endpoint(location),
        provider="vertex-ai",
        metadata={"model": config.llm.model, "location": location},
        notes="LiveKit Google plugin requests for LLM generation (Vertex)",
    )

    # Lower temperature for reliable function calling
    temperature = min(config.llm.temperature, 0.3)

    return _google_plugin.LLM(
        model=config.llm.model,
        temperature=temperature,
        vertexai=True,
        project=PROJECT,
        location=location,
    )


def _create_gemini_api_llm(config: TenantConfig) -> Any:
    """Create Google Gemini LLM via the Google AI Studio API key path.

    Uses ``GOOGLE_API_KEY`` (or ``GEMINI_API_KEY``) instead of Vertex ADC, so the
    agent can run without a GCP service account / project. Set the tenant's
    ``llm.provider`` to ``gemini_api`` to enable.
    """
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "gemini_api provider requires GOOGLE_API_KEY (or GEMINI_API_KEY) in the env."
        )

    register_service_route(
        "gemini_llm",
        "https://generativelanguage.googleapis.com",
        provider="google_ai_studio",
        metadata={"model": config.llm.model},
        notes="LiveKit Google plugin requests for LLM generation (AI Studio key)",
    )

    temperature = min(config.llm.temperature, 0.3)

    return _google_plugin.LLM(
        model=config.llm.model,
        temperature=temperature,
        vertexai=False,
        api_key=api_key,
    )


def _create_openai_llm(config: TenantConfig) -> Any:
    """Create OpenAI-compatible LLM instance."""
    from livekit.plugins import openai

    base_url = os.getenv("OPENAI_BASE_URL")
    kwargs: dict[str, Any] = {
        "model": config.llm.model,
        "temperature": config.llm.temperature,
    }
    if base_url:
        kwargs["base_url"] = base_url

    return openai.LLM(**kwargs)


def _create_custom_llm(config: TenantConfig) -> Any:
    """Create custom LLM instance using local vLLM server."""
    from pipeline.providers.custom_llm import CustomLLM

    return CustomLLM(
        model=config.llm.model,
        temperature=config.llm.temperature,
        max_completion_tokens=config.llm.max_tokens,
    )


def _create_lexantei_llm(config: TenantConfig) -> Any:
    """Create Lexantei LLM instance for Uzbek language."""
    try:
        from llm.lexantei_api import LexanteiAPI
        from llm.lexantei_llm import LexanteiLLM

        api_url = os.getenv("LEXANTEI_API_BASE_URL", "https://api.lexantei.com/api/v1")
        user_id = os.getenv("LEXANTEI_USER_ID", "default_user")

        return LexanteiLLM(api_client=LexanteiAPI(base_url=api_url), user_id=user_id)
    except ImportError:
        logger.warning("Lexantei not available, falling back to Gemini")
        return _create_gemini_llm(config)
