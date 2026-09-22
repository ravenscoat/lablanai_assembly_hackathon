"""End-to-end flow: probe → summarize → submit with the new goals path."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from api.models import MurojaatResult
from config.schema import TenantConfig


def _ya_config_with_goals():
    return TenantConfig.model_validate(
        {
            "tenant": {"id": "ya-int", "slug": "yoshlar", "name": "YA"},
            "languages": {"default": "uz", "available": ["uz"]},
            "sub_agents": {
                "appeal": {
                    "type": "appeal",
                    "instructions": "Sen murojaat yig'uvchisan.",
                    "fields": [
                        {"name": "content", "prompt": "?"},
                        {"name": "full_name", "prompt": "Ism-familiya?"},
                        {"name": "age", "prompt": "Yosh?", "validation": "14-30"},
                        {"name": "region", "prompt": "Viloyat?"},
                        {"name": "district", "prompt": "Tuman?"},
                        {"name": "position", "prompt": "Lavozim?"},
                    ],
                    "goals": [
                        {"key": "problem_specifics", "description": {"uz": "x"}},
                        {"key": "desired_outcome", "description": {"uz": "y"}},
                    ],
                }
            },
            "transfer": {"enabled": True, "office_hours_start": 9, "office_hours_end": 18},
        }
    )


@pytest.mark.asyncio
async def test_summary_lands_as_content_on_submission():
    """Regression: when goals are declared, the summary argument becomes the
    content kwarg sent to submit_murojaat (not the caller's literal quote)."""
    from agents.factory import build_appeal_handoff_from_agent
    from agents.sub_agents.appeal import AppealAgent

    cfg = _ya_config_with_goals()

    # Build an AppealAgent via the production helper to exercise the real
    # construction path (probing block, state forwarding).
    class _FakeMainAgent:
        language = "uz"
        call_db_id = "db-int-1"
        caller_phone = "+923900000000"
        _platform_client = None
        chat_ctx = None

    main = _FakeMainAgent()
    mock_client = AsyncMock()
    mock_client.submit_murojaat.return_value = MurojaatResult(success=True, murojaat_id="int-1")
    main._platform_client = mock_client

    result = build_appeal_handoff_from_agent(main, cfg)
    assert result is not None
    sub, _ = result
    assert isinstance(sub, AppealAgent)
    # content was stripped from fields under goals:
    field_names = {f.name for f in sub._fields}  # noqa: SLF001
    assert "content" not in field_names
    assert "full_name" in field_names

    # Stub _create_main_agent so the success path doesn't try to build a real
    # TenantAgent (would need YANDEX_API_KEY / VoiceFactory wiring). Same trick
    # used in Task 6 unit tests for confirm_and_submit.
    sub._create_main_agent = lambda: MagicMock()  # noqa: SLF001

    # Simulate: LLM has gathered demographics and now calls confirm_and_submit
    # with a contextual summary. Manually populate _data the way the real
    # setter tools would, then call confirm_and_submit's wrapped coroutine.
    sub._data.update(  # noqa: SLF001
        {
            "full_name": "Ali Valiyev",
            "age": 22,
            "region": "Toshkent",
            "district": "Chilonzor",
            "position": "Talaba",
        }
    )

    summary = (
        "Chaqiruvchi (22 yosh, Toshkent) biznes uchun kichik kredit olishga "
        "harakat qilgan; Falon bankka topshirgan, kredit tarixi yetarli emas "
        "deb rad etilgan; SME dasturlari bo'yicha yordam kutmoqda."
    )
    tool = AppealAgent.confirm_and_submit
    fn = getattr(tool, "fnc", None) or getattr(tool, "_fnc", None) or tool
    await fn(sub, context=None, summary=summary)

    # _submit was called; content passed to the platform client IS the summary.
    mock_client.submit_murojaat.assert_called_once()
    kwargs = mock_client.submit_murojaat.call_args.kwargs
    # Critical regression guard: the LLM-crafted summary (not a literal quote)
    # lands as murojaat content. A length check adds a teeth-in sanity signal —
    # if the summary branch regresses to fallback-on-empty-_data["content"],
    # kwargs["content"] would be "" and this would fail loudly.
    assert kwargs["content"] == summary
    assert len(kwargs["content"]) > 50, "summary should be a real synthesized paragraph, not empty"
