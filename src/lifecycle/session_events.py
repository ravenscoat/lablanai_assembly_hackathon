"""
Session event handlers: metrics, transcript, state tracking, telephony latency.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def setup_session_events(
    session: Any,
    agent: Any,
    config: Any,
) -> None:
    """
    Register event handlers on AgentSession.

    Handles:
    - agent_state_changed: speaking state + direct e2e timing
    - user_state_changed: silence timer + mark user stopped speaking
    - conversation_item_added: SDK metrics (bonus)
    """
    telephony_tracker = getattr(agent, "_telephony_tracker", None)

    @session.on("agent_state_changed")
    def on_agent_state(event):
        # LiveKit dispatches AgentStateChangedEvent objects; tests and future
        # call sites may pass a raw string. Tolerate both to avoid silently
        # dropping silence/latency state on AttributeError.
        state = getattr(event, "new_state", event)
        current_agent = getattr(session, "current_agent", None) or agent

        # Ensure monitor exists for the active agent after handoffs.
        try:
            if hasattr(current_agent, "_start_silence_monitor"):
                current_agent._start_silence_monitor()
        except Exception:
            pass

        # Silence monitoring
        try:
            if hasattr(current_agent, "_error_handler") and current_agent._error_handler:
                # Treat both speaking and active processing as "agent busy".
                # Silence should only count when BOTH sides are idle.
                busy_states = {"speaking", "thinking", "processing", "tool_calling"}
                is_busy = state in busy_states
                handler = current_agent._error_handler

                # Cancel any pending "mark idle" task once we become busy again.
                idle_task = getattr(current_agent, "_silence_idle_task", None)
                if idle_task is not None and not idle_task.done() and is_busy:
                    idle_task.cancel()

                if is_busy:
                    handler.set_agent_speaking(True)
                else:
                    # Some SDK state transitions can briefly report non-busy
                    # before final audio playout completes. Wait for full session
                    # inactivity before opening the silence window.
                    import asyncio

                    async def _mark_idle_after_playout() -> None:
                        try:
                            if hasattr(session, "wait_for_inactive"):
                                await asyncio.wait_for(session.wait_for_inactive(), timeout=30.0)
                            else:
                                await asyncio.sleep(0.2)
                            handler.set_agent_speaking(False)
                        except asyncio.CancelledError:
                            return
                        except Exception:
                            # Best-effort fallback: don't block forever.
                            handler.set_agent_speaking(False)

                    new_task = asyncio.create_task(_mark_idle_after_playout())
                    setattr(current_agent, "_silence_idle_task", new_task)
        except Exception:
            pass

        # Direct e2e timing: agent started speaking = turn complete
        if telephony_tracker and state == "speaking":
            try:
                telephony_tracker.mark_agent_started_speaking()
            except Exception:
                pass

    @session.on("user_state_changed")
    def on_user_state(event):
        state = getattr(event, "new_state", event)
        current_agent = getattr(session, "current_agent", None) or agent

        # Ensure monitor exists for the active agent after handoffs.
        try:
            if hasattr(current_agent, "_start_silence_monitor"):
                current_agent._start_silence_monitor()
        except Exception:
            pass

        # Silence timer reset
        try:
            if hasattr(current_agent, "_error_handler") and current_agent._error_handler:
                current_agent._error_handler.set_user_speaking(state == "speaking")
                # Reset on any explicit user-activity state, not only
                # "speaking". Some pipelines transition through non-idle
                # states while still buffering/transcribing the same utterance.
                # Limiting reset to only "speaking" can trigger near-race
                # silence prompts while caller audio is still in-flight.
                if state and state != "idle":
                    current_agent._error_handler.reset_silence_timer()
        except Exception:
            pass

        # Direct e2e timing: user stopped speaking = turn start
        if telephony_tracker and state != "speaking":
            try:
                telephony_tracker.mark_user_stopped_speaking()
            except Exception:
                pass

    # Transcript capture + SDK metrics.
    # `on_user_turn_completed` in TenantAgent only records user messages, so
    # agent replies were missing from the transcript. Subscribe to
    # `conversation_item_added` which fires for both roles.
    try:
        import time as _time

        from livekit.agents import ConversationItemAddedEvent
        from livekit.agents.llm import ChatMessage

        @session.on("conversation_item_added")
        def on_conversation_item_added(ev: ConversationItemAddedEvent):
            try:
                if not isinstance(ev.item, ChatMessage):
                    return

                # 1) Append to conversation_history so the final PATCH
                #    /calls/:id transcript contains both sides.
                text = ""
                content = getattr(ev.item, "text_content", None)
                if callable(content):
                    text = content() or ""
                elif isinstance(content, str):
                    text = content
                else:
                    raw = getattr(ev.item, "content", None)
                    if isinstance(raw, list):
                        text = " ".join(str(p) for p in raw if p)
                    elif isinstance(raw, str):
                        text = raw

                text = (text or "").strip()
                role = ev.item.role or "unknown"
                if role == "assistant":
                    role = "agent"

                if text and role in ("user", "agent") and hasattr(agent, "conversation_history"):
                    history = agent.conversation_history
                    # on_user_turn_completed already appends the user side; skip
                    # duplicates so we don't inflate the transcript.
                    last = history[-1] if history else None
                    is_dup = (
                        last is not None
                        and last.get("role") == role
                        and last.get("content") == text
                    )
                    if not is_dup:
                        history.append(
                            {
                                "role": role,
                                "content": text,
                                "timestamp": _time.time(),
                            }
                        )

                # 2) Telephony SDK metrics (unchanged).
                if telephony_tracker is None:
                    return

                sdk_metrics = getattr(ev.item, "metrics", None)
                if sdk_metrics is None:
                    return

                if hasattr(sdk_metrics, "items"):
                    metrics_dict = dict(sdk_metrics)
                elif hasattr(sdk_metrics, "__dict__"):
                    metrics_dict = vars(sdk_metrics)
                else:
                    metrics_dict = {}

                if metrics_dict:
                    telephony_tracker.record_sdk_metrics(
                        role=ev.item.role,
                        sdk_metrics=metrics_dict,
                    )
            except Exception as e:
                logger.debug("conversation_item_added handler: %s", e)

    except ImportError:
        logger.debug("ConversationItemAddedEvent not available in this SDK version")
