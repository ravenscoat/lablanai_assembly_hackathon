"""
search_knowledge_base tool: queries tenant's knowledge base for answers.

NAV-214: Hardened to prevent fabricated answers.

Changes vs the previous prompt-only design:
  1. Pre-search out-of-scope keyword gate — if the user's query mentions
     an agency or topic the tenant is NOT responsible for (e.g. Health
     Ministry, Anti-corruption Agency, Intellectual Property Dept for the
     Youth Agency tenant), short-circuit BEFORE hitting Qdrant. This is
     the single biggest fabrication driver: gemini-embedding-001 returns
     spuriously high scores (0.65-0.85) for cross-agency phone queries,
     causing the LLM to read back wrong numbers as if they were correct.

  2. Top-K chunk surfacing — return the top-3 chunks (with their scores)
     to the LLM, not just the top-1 answer. Lets the LLM detect a
     wrong-match by cross-referencing multiple chunks.

  3. Original query echo — include the user's original query in the tool
     output, so the LLM can verify "is this answer actually about what
     was asked?" before reading it back.

  4. Per-call telemetry — emit a structured log line for every call with
     query, scores, decision path, and a `tool_call` record attached to
     the RunContext userdata so the platform can persist it to
     calls.metrics.tool_calls[] for the >95% KB-search dashboard metric.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from livekit.agents import RunContext, function_tool

from config.schema import TenantConfig

logger = logging.getLogger(__name__)


def _build_oos_pattern(keywords: list[str]) -> re.Pattern[str] | None:
    """Compile a single case-insensitive OR-pattern from the keyword list."""
    if not keywords:
        return None
    parts = [re.escape(k.strip()) for k in keywords if k and k.strip()]
    if not parts:
        return None
    return re.compile(r"\b(?:" + "|".join(parts) + r")\b", re.IGNORECASE)


# --- Keyword-match boost (rescue terse in-scope queries) ---------------------
# gemini-embedding-001 scores a short brand/alias query (e.g. "uz kombinator")
# just below the medium floor even when it's a genuine in-scope hit, because a
# 2-word query embeds weakly against a ~100-word entry. When the query exactly
# matches an operator-curated alias in the TOP entry's `keywords`, that low
# score is an embedding artifact, not a wrong match — so we promote it to a
# MEDIUM hit. This separates in-scope (keyword overlap) from cross-agency (no
# overlap, and already gated by the out-of-scope step), so NAV-214 holds
# WITHOUT lowering the global threshold.
#
# Only scores within KEYWORD_BOOST_MARGIN of medium are rescued — a genuinely
# wrong match scoring far below the floor is never boosted, even on overlap.
KEYWORD_BOOST_MARGIN = 0.10
# Ignore trivially short aliases ("uz") so a single common token can't boost.
MIN_KEYWORD_CHARS = 5


def _matched_keyword(query: str, keywords: list[str] | None) -> str | None:
    """Return the curated keyword the query matches, else None.

    Bidirectional substring on whitespace-normalized, lowercased text, so it
    catches an exact alias ("uz kombinator"), an STT truncation
    ("uz kombinato" ⊂ "uz kombinator"), and the alias inside a longer query
    ("uz kombinator haqida ...").
    """
    if not keywords:
        return None
    q = re.sub(r"\s+", " ", (query or "").lower()).strip()
    if len(q) < MIN_KEYWORD_CHARS:
        return None
    for kw in keywords:
        k = re.sub(r"\s+", " ", (kw or "").lower()).strip()
        if len(k) < MIN_KEYWORD_CHARS:
            continue
        if k in q or q in k:
            return kw
    return None


def _record_tool_call(context: RunContext, record: dict[str, Any]) -> None:
    """
    Attach a tool_call record to the session userdata so the platform can
    flush it to calls.metrics.tool_calls[]. Best-effort; never raises.
    """
    try:
        userdata = getattr(context.session, "userdata", None)
        if userdata is None:
            return
        bucket = getattr(userdata, "tool_calls", None)
        if bucket is None:
            try:
                userdata.tool_calls = []
                bucket = userdata.tool_calls
            except Exception:
                return
        bucket.append(record)
    except Exception:
        # Telemetry must never break the agent.
        pass


def create_search_kb_tool(config: TenantConfig, kb_manager: Any = None, **kwargs):
    """Factory: creates a search_knowledge_base function_tool bound to tenant's KB."""

    kb_cfg = config.knowledge_base
    thresholds = kb_cfg.confidence_thresholds
    oos_pattern = _build_oos_pattern(kb_cfg.out_of_scope_keywords)
    oos_response = (kb_cfg.out_of_scope_response or config.personality.fallback).strip()
    surface_top_k = max(1, int(kb_cfg.surface_top_k or 3))
    tenant_id = config.tenant.id
    collection = kb_cfg.collection

    @function_tool(name="search_knowledge_base")
    async def search_knowledge_base(context: RunContext, query: str) -> str:
        """Bilimlar bazasidan savol javobini qidirish. Search the knowledge base for an answer.

        Args:
            query: The user's question to search for in the knowledge base.
        """
        started = time.monotonic()
        clean_query = (query or "").strip()

        # 1. Out-of-scope keyword gate — short-circuit before any embedding call.
        if oos_pattern is not None and clean_query and oos_pattern.search(clean_query):
            match = oos_pattern.search(clean_query)
            matched_term = match.group(0) if match else ""
            logger.info(
                "search_kb out_of_scope tenant=%s query=%r matched=%r",
                tenant_id,
                clean_query[:120],
                matched_term,
            )
            _record_tool_call(
                context,
                {
                    "tool": "search_knowledge_base",
                    "query": clean_query,
                    "decision": "out_of_scope",
                    "matched_term": matched_term,
                    "latency_ms": int((time.monotonic() - started) * 1000),
                },
            )
            return oos_response

        # 2. Empty / missing query — return fallback (never invent).
        if not clean_query:
            _record_tool_call(
                context,
                {
                    "tool": "search_knowledge_base",
                    "query": "",
                    "decision": "empty_query",
                    "latency_ms": int((time.monotonic() - started) * 1000),
                },
            )
            return config.personality.fallback

        # 3. Missing kb_manager — fail safe.
        if not kb_manager:
            logger.warning("search_kb no kb_manager tenant=%s", tenant_id)
            _record_tool_call(
                context,
                {
                    "tool": "search_knowledge_base",
                    "query": clean_query,
                    "decision": "no_kb_manager",
                    "latency_ms": int((time.monotonic() - started) * 1000),
                },
            )
            return config.personality.fallback

        # 4. Actual KB search.
        try:
            result = await kb_manager.search(
                tenant_id=tenant_id,
                query=clean_query,
                collection=collection,
                top_k=surface_top_k,
            )
        except TypeError:
            # KBManager.search may not yet accept top_k in older deployments.
            result = await kb_manager.search(
                tenant_id=tenant_id,
                query=clean_query,
                collection=collection,
            )
        except Exception as e:
            logger.error(
                "KB search error tenant=%s query=%r err=%s", tenant_id, clean_query[:80], e
            )
            _record_tool_call(
                context,
                {
                    "tool": "search_knowledge_base",
                    "query": clean_query,
                    "decision": "error",
                    "error": str(e)[:200],
                    "latency_ms": int((time.monotonic() - started) * 1000),
                },
            )
            return config.personality.fallback

        # 5. No result at all — return fallback, do NOT invent.
        if not result or not result.answer:
            logger.info("search_kb no_match tenant=%s query=%r", tenant_id, clean_query[:120])
            _record_tool_call(
                context,
                {
                    "tool": "search_knowledge_base",
                    "query": clean_query,
                    "decision": "no_match",
                    "latency_ms": int((time.monotonic() - started) * 1000),
                },
            )
            return config.personality.fallback

        confidence = float(result.confidence or 0.0)
        chunks = list(result.chunks or [])
        chunk_scores = list(result.chunk_scores or [])

        # 6. Below medium — fabrication risk too high; escalate, UNLESS the query
        #    matches an operator-curated keyword of the matched entry (a terse
        #    in-scope query that embeds just under the floor — see
        #    _matched_keyword). Cross-agency queries have no youth-keyword overlap
        #    and are already gated in step 1, so NAV-214 holds.
        boosted_kw = None
        if confidence < thresholds.medium:
            if confidence >= thresholds.medium - KEYWORD_BOOST_MARGIN:
                boosted_kw = _matched_keyword(clean_query, result.keywords)
            if not boosted_kw:
                logger.info(
                    "search_kb below_threshold tenant=%s query=%r score=%.3f medium=%.3f",
                    tenant_id,
                    clean_query[:120],
                    confidence,
                    thresholds.medium,
                )
                _record_tool_call(
                    context,
                    {
                        "tool": "search_knowledge_base",
                        "query": clean_query,
                        "decision": "below_medium",
                        "top_score": confidence,
                        "latency_ms": int((time.monotonic() - started) * 1000),
                    },
                )
                return config.personality.fallback
            logger.info(
                "search_kb keyword_boost tenant=%s query=%r score=%.3f medium=%.3f matched_kw=%r",
                tenant_id,
                clean_query[:120],
                confidence,
                thresholds.medium,
                boosted_kw,
            )

        # 7. Build cross-reference block — top-K chunks with their scores so
        #    the LLM can detect wrong-matches before reading anything back.
        confidence_band = "HIGH" if confidence >= thresholds.high else "MEDIUM"
        lines = [
            f"[KB result for query: {clean_query!r}]",
            f"[Confidence: {confidence_band} ({confidence:.2f})]",
            "",
            result.answer.strip(),
        ]
        extra_chunks = []
        for idx, (chunk, score) in enumerate(
            zip(chunks[1:surface_top_k], chunk_scores[1:surface_top_k]), start=2
        ):
            if chunk and chunk.strip() and chunk.strip() != result.answer.strip():
                extra_chunks.append(f"[Alt #{idx} (score {score:.2f})]: {chunk.strip()}")
        if extra_chunks:
            lines.append("")
            lines.extend(extra_chunks)

        if confidence_band == "MEDIUM":
            lines.append("")
            lines.append(
                "[IMPORTANT]: Confidence is MEDIUM. Only use facts that are "
                "explicitly in the chunks above. If the chunks don't directly "
                "answer the user's question, say "
                "\"bu haqida aniq ma'lumotim yo'q\" and offer to escalate. "
                "Do NOT invent phone numbers, hours, or amounts."
            )

        logger.info(
            "search_kb hit tenant=%s query=%r score=%.3f band=%s",
            tenant_id,
            clean_query[:120],
            confidence,
            confidence_band,
        )
        record = {
            "tool": "search_knowledge_base",
            "query": clean_query,
            "decision": "keyword_boost" if boosted_kw else "hit",
            "top_score": confidence,
            "band": confidence_band,
            "n_chunks": len(extra_chunks) + 1,
            "latency_ms": int((time.monotonic() - started) * 1000),
        }
        if boosted_kw:
            record["matched_keyword"] = boosted_kw
        _record_tool_call(context, record)
        return "\n".join(lines)

    return search_knowledge_base
