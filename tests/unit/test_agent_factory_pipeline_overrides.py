from __future__ import annotations

from unittest.mock import MagicMock, patch

from agents.factory import AgentFactory
from config.schema import TenantConfig, TenantIdentity


def test_create_agent_respects_explicit_pipeline_overrides():
    config = TenantConfig(
        tenant=TenantIdentity(id="t1", slug="paynet"),
    )
    tool_registry = MagicMock()
    tool_registry.resolve.return_value = None
    factory = AgentFactory(tool_registry=tool_registry)

    supplied_stt = object()
    supplied_tts = object()
    supplied_llm = object()
    created_agent = MagicMock()

    with (
        patch("agents.tenant_agent.TenantAgent", return_value=created_agent) as tenant_cls,
        patch("pipeline.voice_factory.VoiceFactory.create_stt_for_language") as create_stt,
        patch("pipeline.voice_factory.VoiceFactory.create_tts_for_language") as create_tts,
        patch("pipeline.voice_factory.VoiceFactory.create_llm") as create_llm,
    ):
        result = factory.create_agent(
            config,
            stt=supplied_stt,
            tts=supplied_tts,
            llm=supplied_llm,
        )

    assert result is created_agent
    kwargs = tenant_cls.call_args.kwargs
    assert kwargs["stt"] is supplied_stt
    assert kwargs["tts"] is supplied_tts
    assert kwargs["llm"] is supplied_llm
    create_stt.assert_not_called()
    create_tts.assert_not_called()
    create_llm.assert_not_called()
