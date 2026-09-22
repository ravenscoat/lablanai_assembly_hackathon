"""Tests for per-language STT/TTS helpers on VoiceFactory."""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

from config.schema import TenantConfig, TenantIdentity, VoiceConfig
from pipeline.voice_factory import VoiceFactory


def _base_config(**voice_kwargs):
    return TenantConfig(
        tenant=TenantIdentity(id="t1"),
        voice=VoiceConfig(**voice_kwargs),
    )


def _make_provider_mock(class_name: str) -> tuple[ModuleType, MagicMock]:
    """Return a (module_mock, class_mock) pair for a provider module."""
    cls_mock = MagicMock()
    mod_mock = ModuleType(class_name)
    setattr(mod_mock, class_name, cls_mock)
    return mod_mock, cls_mock


def test_create_stt_for_language_passes_correct_locale():
    mod, cls_mock = _make_provider_mock("YandexSTT")
    with patch.dict(sys.modules, {"pipeline.providers.yandex_stt": mod}):
        config = _base_config(stt_provider="yandex")
        VoiceFactory.create_stt_for_language(config, "ru")
    cls_mock.assert_called_once()
    assert cls_mock.call_args.kwargs["language"] == "ru-RU"


def test_create_stt_for_language_uses_stt_providers_mapping():
    yandex_mod, yandex_cls = _make_provider_mock("YandexSTT")
    custom_mod, custom_cls = _make_provider_mock("CustomSTT")
    with patch.dict(
        sys.modules,
        {
            "pipeline.providers.yandex_stt": yandex_mod,
            "pipeline.providers.custom_stt": custom_mod,
        },
    ):
        config = _base_config(
            stt_provider="custom",
            stt_providers={"ru": "yandex", "uz": "custom"},
        )
        VoiceFactory.create_stt_for_language(config, "ru")
        VoiceFactory.create_stt_for_language(config, "uz")
    assert yandex_cls.call_count == 1
    assert yandex_cls.call_args.kwargs["language"] == "ru-RU"
    assert custom_cls.call_count == 1
    assert custom_cls.call_args.kwargs["language"] == "uz"


def test_create_tts_for_language_uses_voices_mapping():
    mod, cls_mock = _make_provider_mock("YandexTTS")
    with patch.dict(sys.modules, {"pipeline.providers.yandex_tts": mod}):
        config = _base_config(
            tts_provider="yandex",
            voices={"ru": "yulduz_ru", "uz": "yulduz"},
        )
        VoiceFactory.create_tts_for_language(config, "ru")
    cls_mock.assert_called_once()
    assert cls_mock.call_args.kwargs["voice"] == "yulduz_ru"
    assert cls_mock.call_args.kwargs["language"] == "ru-RU"


def test_create_tts_for_language_falls_back_to_tts_voice_id():
    mod, cls_mock = _make_provider_mock("YandexTTS")
    with patch.dict(sys.modules, {"pipeline.providers.yandex_tts": mod}):
        config = _base_config(
            tts_provider="yandex",
            tts_voice_id="yulduz",
            voices={},
        )
        VoiceFactory.create_tts_for_language(config, "uz")
    assert cls_mock.call_args.kwargs["voice"] == "yulduz"


def test_create_stt_delegates_to_language_helper():
    """Backward compat: create_stt uses behavior.language."""
    mod, cls_mock = _make_provider_mock("YandexSTT")
    with patch.dict(sys.modules, {"pipeline.providers.yandex_stt": mod}):
        config = TenantConfig(
            tenant=TenantIdentity(id="t1"),
            voice=VoiceConfig(stt_provider="yandex"),
        )
        config.behavior.language = "ru"
        VoiceFactory.create_stt(config)
        assert cls_mock.call_args.kwargs["language"] == "ru-RU"


def test_disable_custom_stt_env_var_routes_custom_to_yandex(monkeypatch):
    yandex_mod, yandex_cls = _make_provider_mock("YandexSTT")
    custom_mod, custom_cls = _make_provider_mock("CustomSTT")
    monkeypatch.setenv("DISABLE_CUSTOM_STT", "1")
    with patch.dict(
        sys.modules,
        {
            "pipeline.providers.yandex_stt": yandex_mod,
            "pipeline.providers.custom_stt": custom_mod,
        },
    ):
        config = _base_config(stt_provider="custom")
        VoiceFactory.create_stt_for_language(config, "uz")
    assert yandex_cls.call_count == 1
    assert custom_cls.call_count == 0


def test_disable_custom_stt_unset_keeps_custom_provider():
    yandex_mod, yandex_cls = _make_provider_mock("YandexSTT")
    custom_mod, custom_cls = _make_provider_mock("CustomSTT")
    with patch.dict(
        sys.modules,
        {
            "pipeline.providers.yandex_stt": yandex_mod,
            "pipeline.providers.custom_stt": custom_mod,
        },
    ):
        config = _base_config(stt_provider="custom")
        VoiceFactory.create_stt_for_language(config, "uz")
    assert custom_cls.call_count == 1
    assert yandex_cls.call_count == 0
