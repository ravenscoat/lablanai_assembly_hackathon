"""
NAV-214: Tests for search_knowledge_base tool hardening against fabrication.

We test the inner behaviour by directly exercising the factory's logic via a
stub kb_manager + a minimal RunContext-like object. The function_tool wrapper
is a thin LiveKit decorator — we reach the underlying callable through its
`__wrapped__` attribute (LiveKit's function_tool exposes the original coroutine
this way), or by calling the returned tool directly if it is invocable.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest

from config.schema import (
    KBConfidenceThresholds,
    KnowledgeBaseConfig,
    PersonalityConfig,
    TenantConfig,
    TenantIdentity,
)
from knowledge.rag_engine import RAGResult
from tools.platform.search_kb import create_search_kb_tool


@dataclass
class _StubUserdata:
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


class _StubSession:
    def __init__(self):
        self.userdata = _StubUserdata()


class _StubContext:
    def __init__(self):
        self.session = _StubSession()


class _StubKBManager:
    def __init__(self, result: RAGResult | None):
        self._result = result
        self.calls: list[dict[str, Any]] = []

    async def search(self, *, tenant_id: str, query: str, collection: str, top_k: int = 3):
        self.calls.append(
            {"tenant_id": tenant_id, "query": query, "collection": collection, "top_k": top_k}
        )
        return self._result


def _make_config(
    *,
    high: float = 0.78,
    medium: float = 0.72,
    oos: list[str] | None = None,
    oos_response: str = "OUT_OF_SCOPE_RESPONSE",
    fallback: str = "FALLBACK_RESPONSE",
) -> TenantConfig:
    cfg = TenantConfig(
        tenant=TenantIdentity(id="t_test", name="Test"),
        personality=PersonalityConfig(fallback=fallback),
        knowledge_base=KnowledgeBaseConfig(
            enabled=True,
            collection="test_collection",
            confidence_thresholds=KBConfidenceThresholds(high=high, medium=medium),
            out_of_scope_keywords=oos or [],
            out_of_scope_response=oos_response,
            surface_top_k=3,
        ),
    )
    return cfg


def _invoke(tool, context, query: str) -> str:
    """
    LiveKit's @function_tool decorator wraps an async coroutine. The underlying
    callable is exposed differently across versions; try the common attrs.
    """
    target = (
        getattr(tool, "__wrapped__", None)
        or getattr(tool, "fnc", None)
        or getattr(tool, "callable", None)
        or tool
    )
    coro = target(context, query)
    return asyncio.get_event_loop().run_until_complete(coro)


@pytest.mark.asyncio
async def test_out_of_scope_keyword_short_circuits_before_kb():
    cfg = _make_config(oos=["Sog'liqni saqlash", "intellektual mulk"])
    kb = _StubKBManager(
        result=RAGResult(answer="WRONG-ANSWER", confidence=0.95, chunks=[], chunk_scores=[])
    )
    tool = create_search_kb_tool(cfg, kb_manager=kb)
    ctx = _StubContext()

    target = getattr(tool, "__wrapped__", None) or getattr(tool, "fnc", None) or tool
    result = await target(ctx, "Sog'liqni saqlash vazirligi telefon raqami nima?")

    assert result == "OUT_OF_SCOPE_RESPONSE"
    assert kb.calls == [], "KB must NOT be queried when query matches out-of-scope keyword"
    assert ctx.session.userdata.tool_calls[-1]["decision"] == "out_of_scope"


@pytest.mark.asyncio
async def test_out_of_scope_case_insensitive_and_word_boundary():
    cfg = _make_config(oos=["intellektual mulk", "1003"])
    kb = _StubKBManager(result=None)
    tool = create_search_kb_tool(cfg, kb_manager=kb)
    target = getattr(tool, "__wrapped__", None) or getattr(tool, "fnc", None) or tool

    # Case-insensitive
    r1 = await target(_StubContext(), "INTELLEKTUAL MULK departamenti")
    assert r1 == "OUT_OF_SCOPE_RESPONSE"

    # Number with word boundary — "1003" alone matches, "10003" does not.
    r2 = await target(_StubContext(), "1003 raqami nima?")
    assert r2 == "OUT_OF_SCOPE_RESPONSE"
    r3 = await target(_StubContext(), "10030 raqami nima?")
    assert r3 != "OUT_OF_SCOPE_RESPONSE"


@pytest.mark.asyncio
async def test_below_medium_threshold_returns_fallback_not_invented_answer():
    cfg = _make_config(high=0.78, medium=0.72)
    kb = _StubKBManager(
        result=RAGResult(
            answer="MAYBE-WRONG-1093",
            confidence=0.68,  # below medium=0.72
            chunks=["MAYBE-WRONG-1093"],
            chunk_scores=[0.68],
        )
    )
    tool = create_search_kb_tool(cfg, kb_manager=kb)
    target = getattr(tool, "__wrapped__", None) or getattr(tool, "fnc", None) or tool
    ctx = _StubContext()

    result = await target(ctx, "kredit haqida")

    assert result == "FALLBACK_RESPONSE"
    assert ctx.session.userdata.tool_calls[-1]["decision"] == "below_medium"


@pytest.mark.asyncio
async def test_high_confidence_returns_top_k_block_with_query_echo():
    cfg = _make_config(high=0.78, medium=0.72)
    kb = _StubKBManager(
        result=RAGResult(
            answer="Yoshlar daftari arizasi davlat xizmatlari markazida topshiriladi.",
            confidence=0.88,
            chunks=[
                "Yoshlar daftari arizasi davlat xizmatlari markazida topshiriladi.",
                "Yoshlar daftariga 14-30 yosh oralig'idagilar kiritiladi.",
                "Ariza 3-13 ish kunida ko'rib chiqiladi.",
            ],
            chunk_scores=[0.88, 0.86, 0.83],
        )
    )
    tool = create_search_kb_tool(cfg, kb_manager=kb)
    target = getattr(tool, "__wrapped__", None) or getattr(tool, "fnc", None) or tool
    ctx = _StubContext()

    result = await target(ctx, "Yoshlar daftariga qanday yoziladi?")

    assert "[KB result for query:" in result
    assert "Yoshlar daftariga qanday yoziladi?" in result
    assert "Confidence: HIGH" in result
    assert "davlat xizmatlari markazida" in result
    # Alt chunks must be surfaced
    assert "Alt #2" in result and "Alt #3" in result
    rec = ctx.session.userdata.tool_calls[-1]
    assert rec["decision"] == "hit"
    assert rec["band"] == "HIGH"
    assert rec["top_score"] == pytest.approx(0.88)


@pytest.mark.asyncio
async def test_medium_confidence_includes_anti_fabrication_warning():
    cfg = _make_config(high=0.78, medium=0.72)
    kb = _StubKBManager(
        result=RAGResult(
            answer="Some marginal match",
            confidence=0.74,  # between medium and high
            chunks=["Some marginal match"],
            chunk_scores=[0.74],
        )
    )
    tool = create_search_kb_tool(cfg, kb_manager=kb)
    target = getattr(tool, "__wrapped__", None) or getattr(tool, "fnc", None) or tool
    ctx = _StubContext()

    result = await target(ctx, "Ambig query")
    assert "Confidence: MEDIUM" in result
    assert "Do NOT invent" in result
    assert ctx.session.userdata.tool_calls[-1]["band"] == "MEDIUM"


@pytest.mark.asyncio
async def test_no_match_returns_fallback():
    cfg = _make_config()
    kb = _StubKBManager(result=RAGResult(answer="", confidence=0.0, chunks=[], chunk_scores=[]))
    tool = create_search_kb_tool(cfg, kb_manager=kb)
    target = getattr(tool, "__wrapped__", None) or getattr(tool, "fnc", None) or tool
    ctx = _StubContext()

    result = await target(ctx, "anything")
    assert result == "FALLBACK_RESPONSE"
    assert ctx.session.userdata.tool_calls[-1]["decision"] == "no_match"


@pytest.mark.asyncio
async def test_kb_search_exception_does_not_propagate():
    cfg = _make_config()

    class _BoomKB:
        async def search(self, **kwargs):
            raise RuntimeError("qdrant unreachable")

    tool = create_search_kb_tool(cfg, kb_manager=_BoomKB())
    target = getattr(tool, "__wrapped__", None) or getattr(tool, "fnc", None) or tool
    ctx = _StubContext()

    result = await target(ctx, "anything")
    assert result == "FALLBACK_RESPONSE"
    assert ctx.session.userdata.tool_calls[-1]["decision"] == "error"


@pytest.mark.asyncio
async def test_telemetry_records_latency_and_query():
    cfg = _make_config()
    kb = _StubKBManager(
        result=RAGResult(answer="ans", confidence=0.85, chunks=["ans"], chunk_scores=[0.85])
    )
    tool = create_search_kb_tool(cfg, kb_manager=kb)
    target = getattr(tool, "__wrapped__", None) or getattr(tool, "fnc", None) or tool
    ctx = _StubContext()

    await target(ctx, "  test query  ")
    rec = ctx.session.userdata.tool_calls[-1]
    assert rec["query"] == "test query"
    assert "latency_ms" in rec and rec["latency_ms"] >= 0


# --- keyword-match boost (rescue terse in-scope queries) ---------------------


def _kw_tool(confidence: float, keywords: list[str], *, medium: float = 0.72):
    cfg = _make_config(high=0.78, medium=medium)
    kb = _StubKBManager(
        result=RAGResult(
            answer="UzKombinator — O'zbekistonning birinchi startup akseleratori.",
            confidence=confidence,
            chunks=["UzKombinator — ...startup akseleratori."],
            chunk_scores=[confidence],
            keywords=keywords,
        )
    )
    tool = create_search_kb_tool(cfg, kb_manager=kb)
    target = getattr(tool, "__wrapped__", None) or getattr(tool, "fnc", None) or tool
    return target


@pytest.mark.asyncio
async def test_keyword_boost_rescues_terse_in_scope_query():
    # 0.704 < medium 0.72, but the query exactly matches a curated alias.
    target = _kw_tool(0.704, ["uzkombinator", "uz kombinator", "akselerator"])
    ctx = _StubContext()
    result = await target(ctx, "uz kombinator")

    assert "[KB result for query:" in result  # answered, not fallback
    assert "Confidence: MEDIUM" in result  # boosted hits stay MEDIUM (caveat applies)
    assert "Do NOT invent" in result
    rec = ctx.session.userdata.tool_calls[-1]
    assert rec["decision"] == "keyword_boost"
    assert rec["matched_keyword"] == "uz kombinator"
    assert rec["band"] == "MEDIUM"


@pytest.mark.asyncio
async def test_keyword_boost_handles_stt_truncation():
    # Real observed LLM search query when STT dropped the trailing 'r':
    # "uz kombinato" — a prefix substring of the curated alias "uz kombinator".
    target = _kw_tool(0.668, ["uz kombinator"])
    ctx = _StubContext()
    result = await target(ctx, "uz kombinato")

    assert "[KB result for query:" in result
    assert ctx.session.userdata.tool_calls[-1]["decision"] == "keyword_boost"


@pytest.mark.asyncio
async def test_keyword_boost_skipped_when_far_below_floor():
    # 0.50 is below the boost floor (medium - 0.10 = 0.62) → never rescued,
    # even though the alias matches. Guards against boosting a wrong match.
    target = _kw_tool(0.50, ["uz kombinator"])
    ctx = _StubContext()
    result = await target(ctx, "uz kombinator")

    assert result == "FALLBACK_RESPONSE"
    assert ctx.session.userdata.tool_calls[-1]["decision"] == "below_medium"


@pytest.mark.asyncio
async def test_keyword_boost_skipped_without_overlap():
    # In-band score (0.68) but query shares no alias → stays below_medium,
    # so a generic cross-agency near-miss is NOT boosted (NAV-214 preserved).
    target = _kw_tool(0.68, ["startup", "akselerator", "demo day"])
    ctx = _StubContext()
    result = await target(ctx, "kredit stavkasi qancha")

    assert result == "FALLBACK_RESPONSE"
    assert ctx.session.userdata.tool_calls[-1]["decision"] == "below_medium"


@pytest.mark.asyncio
async def test_keyword_boost_ignores_too_short_alias():
    # A 2-char alias ("uz") must not boost — would match nearly everything.
    target = _kw_tool(0.68, ["uz"])
    ctx = _StubContext()
    result = await target(ctx, "uz haqida ma'lumot ber")

    assert result == "FALLBACK_RESPONSE"
    assert ctx.session.userdata.tool_calls[-1]["decision"] == "below_medium"
