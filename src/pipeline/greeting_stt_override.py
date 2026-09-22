"""Helpers for first-turn greeting STT override selection.

A tenant can opt into a dedicated STT provider for ONLY the first
post-greeting language-choice turn by setting `voice.greeting_stt_provider`
in its YAML (see VoiceFactory.create_greeting_stt_for_language_choice). This
is useful when a single-shot, language-detecting STT (e.g. Gemini Live) gives
a cleaner first-turn language pick than the per-language STT used for the rest
of the call.
"""

from __future__ import annotations

from config.schema import TenantConfig


def should_use_greeting_stt_override(
    config: TenantConfig,
    preferred_language: str | None,
) -> bool:
    """True for the first-time multilingual language-choice turn when the
    tenant has configured a `voice.greeting_stt_provider`.

    Config-driven (not tied to any specific tenant slug): any multilingual
    tenant that sets `greeting_stt_provider` opts in.
    """
    return (
        config.languages.is_multilingual
        and preferred_language is None
        and (config.voice.greeting_stt_provider or "").strip() != ""
    )


# Backwards-compatible alias for existing call sites/tests.
should_use_paynet_greeting_stt_override = should_use_greeting_stt_override
