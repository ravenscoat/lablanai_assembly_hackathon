"""
P1: Sliding context window — prevents context overflow on long calls.
Keeps last N turns in full detail, compresses older turns into a summary.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Turn:
    """A single conversation turn."""

    role: str  # "user" or "assistant"
    content: str
    turn_number: int
    topic: str = ""


class ContextWindow:
    """
    Manages conversation context with a sliding window.

    Strategy:
    - Keep last `window_size` turns in full detail
    - Summarize older turns into a compact summary
    - Inject summary + recent turns into LLM context each turn

    This prevents context overflow on 10+ minute calls while
    preserving recent conversation flow.
    """

    def __init__(self, window_size: int = 5, max_summary_length: int = 300):
        self._window_size = window_size
        self._max_summary_length = max_summary_length
        self._all_turns: list[Turn] = []
        self._summary: str = ""
        self._turn_counter: int = 0

    @property
    def turn_count(self) -> int:
        return self._turn_counter

    @property
    def summary(self) -> str:
        return self._summary

    def add_turn(self, role: str, content: str, topic: str = "") -> None:
        """Record a new turn."""
        self._turn_counter += 1
        self._all_turns.append(
            Turn(
                role=role,
                content=content,
                turn_number=self._turn_counter,
                topic=topic,
            )
        )

        # Compress when we exceed window
        if len(self._all_turns) > self._window_size + 2:
            self._compress_old_turns()

    def get_context_messages(self) -> list[dict]:
        """
        Get context messages for LLM injection.

        Returns a list of messages:
        1. Summary of older turns (if any)
        2. Recent turns in full
        """
        messages = []

        # Add summary of compressed turns
        if self._summary:
            messages.append(
                {
                    "role": "system",
                    "content": f"[Oldingi suhbat xulosasi] {self._summary}",
                }
            )

        return messages

    def get_recent_turns(self) -> list[Turn]:
        """Get the recent turns within the window."""
        return self._all_turns[-self._window_size :]

    def _compress_old_turns(self) -> None:
        """
        Compress turns outside the window into a summary.
        Uses extractive summarization (no LLM needed).
        """
        # Turns to compress (everything except last window_size)
        old_turns = self._all_turns[: -self._window_size]
        recent_turns = self._all_turns[-self._window_size :]

        if not old_turns:
            return

        # Build summary from old turns
        summary_parts = []

        # Extract topics discussed
        topics = list(dict.fromkeys(t.topic for t in old_turns if t.topic))
        if topics:
            summary_parts.append(f"Muhokama qilingan mavzular: {', '.join(topics)}")

        # Extract key user questions (skip short fillers)
        user_questions = [
            t.content[:80] for t in old_turns if t.role == "user" and len(t.content) > 10
        ]
        if user_questions:
            # Keep last 3 meaningful questions
            recent_qs = user_questions[-3:]
            summary_parts.append(
                "Foydalanuvchi so'ragan savollar: " + "; ".join(f'"{q}"' for q in recent_qs)
            )

        # Extract key agent actions (tool calls, transfers).
        # TODO(urdu): these keyword heuristics + summary phrases are Uzbek. Add
        # 'ur'/'en' variants (or make them config-driven) for an Urdu tenant.
        agent_actions = []
        for t in old_turns:
            if t.role == "assistant":
                if "murojaat" in t.content.lower():
                    agent_actions.append("murojaat jarayoni boshlangan")
                if "operator" in t.content.lower():
                    agent_actions.append("operatorga yo'naltirish taklif qilingan")
        if agent_actions:
            summary_parts.append(f"Bajarilgan amallar: {', '.join(set(agent_actions))}")

        # Combine and truncate
        self._summary = ". ".join(summary_parts)
        if len(self._summary) > self._max_summary_length:
            self._summary = self._summary[: self._max_summary_length] + "..."

        # Keep only recent turns
        self._all_turns = recent_turns

        logger.debug(
            f"Context compressed: {len(old_turns)} old turns -> summary "
            f"({len(self._summary)} chars), keeping {len(recent_turns)} recent"
        )

    def get_stats(self) -> dict:
        """Get context window statistics."""
        return {
            "total_turns": self._turn_counter,
            "active_turns": len(self._all_turns),
            "has_summary": bool(self._summary),
            "summary_length": len(self._summary),
        }
