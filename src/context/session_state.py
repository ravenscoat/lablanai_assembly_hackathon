"""
P4: Session state manager — injects call state into LLM context.
Gives the LLM awareness of call duration, topics discussed,
and what has happened so far in this session.
"""

from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)


class SessionStateManager:
    """
    Tracks and formats session state for LLM context injection.
    Gives the LLM situational awareness about the ongoing call.
    """

    def __init__(
        self,
        tenant_name: str = "",
        language: str = "uz",
    ):
        self._tenant_name = tenant_name
        self._language = language
        self._call_start: float = time.time()
        self._turn_count: int = 0
        self._topics_discussed: list[str] = []
        self._tools_used: list[str] = []
        self._kb_queries: int = 0
        self._kb_hits: int = 0
        self._transfer_offered: bool = False
        self._sub_agent_active: str | None = None

    def record_turn(self) -> None:
        self._turn_count += 1

    def record_topic(self, topic: str) -> None:
        if topic and topic not in self._topics_discussed:
            self._topics_discussed.append(topic)

    def record_tool_use(self, tool_name: str) -> None:
        self._tools_used.append(tool_name)

    def record_kb_query(self, hit: bool) -> None:
        self._kb_queries += 1
        if hit:
            self._kb_hits += 1

    def record_transfer_offered(self) -> None:
        self._transfer_offered = True

    def set_sub_agent(self, name: str | None) -> None:
        self._sub_agent_active = name

    def get_state_summary(self) -> str:
        """
        Build a concise state summary for LLM injection.
        Updated each turn to give LLM situational awareness.
        """
        duration = int(time.time() - self._call_start)
        minutes = duration // 60
        seconds = duration % 60

        parts = []

        # Call duration
        if minutes > 0:
            parts.append(f"Qo'ng'iroq davomiyligi: {minutes} daqiqa {seconds} soniya")
        else:
            parts.append(f"Qo'ng'iroq davomiyligi: {seconds} soniya")

        # Turn count
        parts.append(f"Navbat: {self._turn_count}")

        # Topics discussed
        if self._topics_discussed:
            topics = ", ".join(self._topics_discussed[-5:])  # last 5
            parts.append(f"Muhokama qilingan mavzular: {topics}")

        # KB usage
        if self._kb_queries > 0:
            hit_rate = (self._kb_hits / self._kb_queries * 100) if self._kb_queries else 0
            parts.append(f"Bilimlar bazasi: {self._kb_queries} so'rov, {hit_rate:.0f}% topildi")

        # Active sub-agent
        if self._sub_agent_active:
            parts.append(f"Faol sub-agent: {self._sub_agent_active}")

        # Transfer status
        if self._transfer_offered:
            parts.append("Operatorga yo'naltirish taklif qilingan")

        # Long call warning
        if duration > 300:  # 5 minutes
            parts.append("DIQQAT: Uzoq qo'ng'iroq — xulosaga keling")

        return " | ".join(parts)

    def should_suggest_wrap_up(self) -> bool:
        """Check if the call is running long and should wrap up."""
        duration = time.time() - self._call_start
        return duration > 480  # 8 minutes
