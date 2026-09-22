from assemblyai_voice.session_config import build_session_update
from config.schema import TenantConfig


def test_builds_full_pipeline_session_from_tenant() -> None:
    config = TenantConfig.model_validate(
        {
            "tenant": {"id": "1", "slug": "relaydesk"},
            "personality": {"system_prompt": "Resolve verified cases.", "greeting": "Hello"},
            "assemblyai": {
                "enabled": True,
                "voice": "ivy",
                "keyterms": ["RelayDesk", "invoice"],
                "min_silence_ms": 700,
            },
        }
    )

    event = build_session_update(config)

    assert event["type"] == "session.update"
    session = event["session"]
    assert session["system_prompt"] == "Resolve verified cases."
    assert session["output"]["voice"] == "ivy"
    assert session["input"]["keyterms"] == ["RelayDesk", "invoice"]
    assert session["input"]["turn_detection"]["min_silence"] == 700


def test_rejects_tenant_without_assemblyai_enabled() -> None:
    config = TenantConfig.model_validate({"tenant": {"id": "1", "slug": "legacy"}})

    try:
        build_session_update(config)
    except ValueError as exc:
        assert "not enabled" in str(exc)
    else:
        raise AssertionError("expected AssemblyAI-disabled tenant to be rejected")
