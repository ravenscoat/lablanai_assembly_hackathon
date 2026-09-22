"""Tests for LanguagesConfig + VoiceConfig.voices."""

from __future__ import annotations

from config.schema import LanguagesConfig, TenantConfig, TenantIdentity, VoiceConfig


def test_languages_config_defaults():
    """Single-language default matches old behavior."""
    cfg = LanguagesConfig()
    assert cfg.default == "uz"
    assert cfg.available == ["uz"]
    assert cfg.ask_on_start is False


def test_languages_config_multilingual():
    cfg = LanguagesConfig(default="ru", available=["ru", "uz"], ask_on_start=True)
    assert cfg.is_multilingual is True
    assert set(cfg.available) == {"ru", "uz"}


def test_languages_is_multilingual_false_for_single_lang():
    assert LanguagesConfig(available=["uz"]).is_multilingual is False


def test_voice_config_voices_mapping():
    voice = VoiceConfig(voices={"ru": "yulduz_ru", "uz": "yulduz"})
    assert voice.voice_for("ru") == "yulduz_ru"
    assert voice.voice_for("uz") == "yulduz"


def test_voice_config_greeting_stt_defaults():
    voice = VoiceConfig()
    assert voice.greeting_stt_provider is None
    assert voice.greeting_stt_languages == []


def test_voice_config_stt_provider_for_uses_per_language_override():
    voice = VoiceConfig(
        stt_provider="custom",
        stt_providers={"ru": "yandex"},
    )
    assert voice.stt_provider_for("ru") == "yandex"
    assert voice.stt_provider_for("uz") == "custom"


def test_voice_config_voice_for_falls_back_to_default_voice_id():
    """Unknown language falls back to the singular tts_voice_id."""
    voice = VoiceConfig(tts_voice_id="yulduz", voices={})
    assert voice.voice_for("ru") == "yulduz"


def test_tenant_config_has_languages_default():
    cfg = TenantConfig(tenant=TenantIdentity(id="t1"))
    assert cfg.languages.default == "uz"
    assert cfg.languages.is_multilingual is False


def test_personality_greetings_dict_default_empty():
    from config.schema import PersonalityConfig

    p = PersonalityConfig()
    assert p.greetings == {}
    # Backward-compat: single greeting still works.
    assert p.greeting.startswith("Assalomu")


def test_personality_greeting_for_picks_language_specific():
    from config.schema import PersonalityConfig

    p = PersonalityConfig(
        greeting="bilingual fallback",
        greetings={"ru": "Здравствуйте!", "uz": "Assalomu alaykum!"},
    )
    assert p.greeting_for("ru") == "Здравствуйте!"
    assert p.greeting_for("uz") == "Assalomu alaykum!"
    assert p.greeting_for("en") == "bilingual fallback"  # fallback


def test_field_config_prompt_for_prefers_per_language():
    from config.schema import FieldConfig

    fc = FieldConfig(
        name="content",
        prompt="fallback uz",
        prompts={"uz": "o'zbekcha", "ru": "по-русски"},
    )
    assert fc.prompt_for("ru") == "по-русски"
    assert fc.prompt_for("uz") == "o'zbekcha"
    assert fc.prompt_for("en") == "fallback uz"  # fallback to monolingual `prompt`


def test_field_config_prompt_for_monolingual_backcompat():
    """Youth-Agency-style config with only `prompt:` still resolves."""
    from config.schema import FieldConfig

    fc = FieldConfig(name="age", prompt="Yoshingiz?", validation="14-30")
    assert fc.prompt_for("uz") == "Yoshingiz?"
    assert fc.prompt_for("ru") == "Yoshingiz?"  # no prompts dict → fallback
    assert fc.validation == "14-30"


def test_sub_agent_config_messages_default_empty():
    from config.schema import SubAgentConfig

    sc = SubAgentConfig(type="appeal")
    assert sc.messages == {}


def test_sub_agent_config_messages_per_language():
    from config.schema import SubAgentConfig

    sc = SubAgentConfig(
        type="appeal",
        messages={
            "uz": {"start": "Boshlaymiz", "success": "Qabul qilindi"},
            "ru": {"start": "Начнём", "success": "Принято"},
        },
    )
    assert sc.messages["ru"]["success"] == "Принято"
