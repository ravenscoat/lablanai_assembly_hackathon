from __future__ import annotations

from types import ModuleType
from unittest.mock import MagicMock, patch

from config.schema import LanguagesConfig, TenantConfig, TenantIdentity, VoiceConfig
from pipeline.greeting_stt_override import should_use_greeting_stt_override
from pipeline.voice_factory import VoiceFactory


def _config(
    *,
    slug: str = "example-tenant",
    multilingual: bool = True,
    greeting_provider: str | None = "gemini",
    stt_provider: str = "yandex",
) -> TenantConfig:
    langs = (
        LanguagesConfig(default="ru", available=["ru", "uz"]) if multilingual else LanguagesConfig()
    )
    voice = VoiceConfig(
        stt_provider=stt_provider,
        greeting_stt_provider=greeting_provider,
        greeting_stt_languages=["ru", "uz"],
    )
    return TenantConfig(
        tenant=TenantIdentity(id="t1", slug=slug),
        languages=langs,
        voice=voice,
    )


def test_should_use_greeting_stt_override_true_for_first_time_multilingual():
    cfg = _config()
    assert should_use_greeting_stt_override(cfg, preferred_language=None) is True


def test_should_use_greeting_stt_override_false_for_returning_caller():
    cfg = _config()
    assert should_use_greeting_stt_override(cfg, preferred_language="ru") is False


def test_should_use_greeting_stt_override_is_slug_agnostic():
    # Config-driven (not tied to any specific tenant slug): any multilingual
    # tenant that configured greeting_stt_provider opts in.
    cfg = _config(slug="another-tenant")
    assert should_use_greeting_stt_override(cfg, preferred_language=None) is True


def test_should_use_greeting_stt_override_false_without_greeting_provider():
    cfg = _config(greeting_provider=None)
    assert should_use_greeting_stt_override(cfg, preferred_language=None) is False


def test_should_use_greeting_stt_override_false_for_single_language():
    cfg = _config(multilingual=False)
    assert should_use_greeting_stt_override(cfg, preferred_language=None) is False


def test_create_greeting_stt_uses_gemini_for_choice_and_fallback_after():
    cfg = _config(greeting_provider="gemini", stt_provider="custom")
    fake_gemini_cls = MagicMock()
    fake_default_stt = object()
    fake_module = ModuleType("pipeline.providers.gemini_live_stt")
    setattr(fake_module, "GeminiLiveSTT", fake_gemini_cls)

    with (
        patch.dict("sys.modules", {"pipeline.providers.gemini_live_stt": fake_module}),
        patch.object(
            VoiceFactory, "create_stt_for_language", return_value=fake_default_stt
        ) as create_stt,
    ):
        VoiceFactory.create_greeting_stt_for_language_choice(
            cfg,
            fallback_language="ru",
        )

    create_stt.assert_called_once_with(cfg, "ru")
    fake_gemini_cls.assert_called_once()
    assert fake_gemini_cls.call_args.kwargs["language_codes"] == ["ru", "uz"]
    assert fake_gemini_cls.call_args.kwargs["fallback_stt"] is fake_default_stt
    assert fake_gemini_cls.call_args.kwargs["one_shot"] is True


def test_create_greeting_stt_falls_back_for_unknown_provider():
    cfg = _config(greeting_provider="unknown", stt_provider="custom")
    with patch.object(
        VoiceFactory, "create_stt_for_language", return_value="fallback"
    ) as create_stt:
        result = VoiceFactory.create_greeting_stt_for_language_choice(
            cfg,
            fallback_language="ru",
        )
    assert result == "fallback"
    create_stt.assert_called_once_with(cfg, "ru")
