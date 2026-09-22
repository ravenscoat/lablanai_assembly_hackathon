"""
FlowEngine: provides flow context to the LLM.
NOT a state machine — the LLM drives the conversation naturally.
Flows are guidance documents, not rigid step sequences.

The engine only:
1. Tracks which flow is active
2. Injects the full flow document as LLM context
3. Lets the LLM handle user responses naturally
"""

from __future__ import annotations

import logging
from pathlib import Path

from .models import FlowAdvanceResult, FlowDocument, FlowState
from .parser import FlowParser

logger = logging.getLogger(__name__)


class FlowEngine:
    """
    Provides flow context to the LLM.
    No keyword matching, no step tracking, no state machine.
    The LLM reads the flow document and follows it naturally.
    """

    def __init__(self, tenant_id: str, flow_dir: Path | str, **kwargs):
        self._tenant_id = tenant_id
        self._flow_dir = Path(flow_dir) if isinstance(flow_dir, str) else flow_dir
        self._flows: dict[str, FlowDocument] = {}
        self._state: FlowState = FlowState.INACTIVE
        self._active_flow: FlowDocument | None = None

    def load_all_flows(self) -> int:
        """Parse all .flow.md files at startup."""
        self._flows = FlowParser.parse_all(self._flow_dir)
        logger.info(
            f"Loaded {len(self._flows)} flows for tenant {self._tenant_id}: "
            f"{list(self._flows.keys())}"
        )
        return len(self._flows)

    def get_flow(self, flow_id: str) -> FlowDocument | None:
        return self._flows.get(flow_id)

    # --- State ---

    def is_active(self) -> bool:
        return self._state == FlowState.ACTIVE and self._active_flow is not None

    @property
    def active_flow_id(self) -> str | None:
        return self._active_flow.id if self._active_flow else None

    # --- Lifecycle ---

    def activate(self, flow: FlowDocument) -> FlowAdvanceResult:
        """Activate a flow. Returns the flow content for the LLM."""
        self._active_flow = flow
        self._state = FlowState.ACTIVE
        logger.info(f"Flow activated: {flow.id} (type={flow.type})")

        # Deterministic flows return fixed text
        # All flows are guided — return the full flow content for LLM
        return FlowAdvanceResult(
            status="advanced",
            step_context=self._build_full_flow_context(flow),
        )

    def deactivate(self) -> None:
        """End the current flow."""
        if self._active_flow:
            logger.info(f"Flow deactivated: {self._active_flow.id}")
        self._active_flow = None
        self._state = FlowState.INACTIVE

    # --- Context injection ---

    def get_context_for_turn(self) -> list[dict[str, str]]:
        """Inject the full flow document as LLM context."""
        if not self.is_active() or not self._active_flow:
            return []

        return [
            {
                "role": "system",
                "content": self._build_full_flow_context(self._active_flow),
            }
        ]

    def _build_full_flow_context(self, flow: FlowDocument) -> str:
        """Build concise flow checklist for the LLM."""
        parts = []
        parts.append(f"[FAOL JARAYON: {flow.title}]")
        parts.append(f"{flow.description}")
        parts.append("")

        # Full step prompts (LLM needs exact wording, URLs, names)
        if flow.steps:
            parts.append("BOSQICHLAR:")
            for i, (step_id, step) in enumerate(flow.steps.items(), 1):
                parts.append(f"  {i}. [{step_id}]: {step.prompt.strip()}")
            parts.append("")

        # TODO(urdu): this flow-step instruction block is in Uzbek — add an
        # 'ur'/'en' variant for an Urdu tenant (framework prompt fragment seam).
        parts.append(
            "QOIDALAR:\n"
            "- Suhbat tarixiga qarab qaysi bosqichda ekanligingizni aniqlang\n"
            "- Bajarilgan bosqichlarni QAYTARMANG\n"
            "- Keyingi bosqichga tabiiy o'ting\n"
            "- Foydalanuvchi allaqachon aytgan narsani qayta so'ramang\n"
            "- Portal, sayt va tashkilot NOMLARINI AYNAN shu holda ayting\n"
            "- start_flow va search_knowledge_base toollarini CHAQIRMANG"
        )

        return "\n".join(parts)
