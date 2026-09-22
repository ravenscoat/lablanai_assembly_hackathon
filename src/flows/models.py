"""
Flow-as-Knowledge data models.
Defines the structure for flow documents parsed from .flow.md files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class FlowState(Enum):
    INACTIVE = "inactive"
    ACTIVE = "active"
    COMPLETED = "completed"


@dataclass
class FlowOption:
    """A selectable option within a flow step."""

    label: str
    keywords: list[str] = field(default_factory=list)
    next_step: str | None = None
    redirect_flow: str | None = None
    action: str | None = None  # "end_flow", "transfer_to_operator"


@dataclass
class FlowValidation:
    """Validation rules for a collect step."""

    type: str = ""  # "range", "options", "non_empty"
    min: int | None = None
    max: int | None = None
    allowed: list[str] | None = None
    fail_message: str = ""
    fail_action: str | None = None  # "end_flow", "retry", "transfer"


@dataclass
class FlowStep:
    """A single step in a flow conversation."""

    id: str
    prompt: str = ""
    type: str = "choice"  # "choice", "collect", "informational", "terminal"
    options: list[FlowOption] = field(default_factory=list)
    collect: str | None = None  # field name to collect from user
    validation: FlowValidation | None = None
    next_step: str | None = None
    action: str | None = None  # "end_flow", "transfer_to_operator"
    on_no_match: str | None = None


@dataclass
class FlowDocument:
    """A complete flow parsed from a .flow.md file."""

    id: str
    type: str  # "deterministic", "guided", "complex"
    title: str = ""
    description: str = ""
    trigger_keywords: list[str] = field(default_factory=list)
    activation_phrases: list[str] = field(default_factory=list)
    faq_phrases: list[str] = field(default_factory=list)
    initial_step: str = ""
    steps: dict[str, FlowStep] = field(default_factory=dict)
    redirect_targets: list[str] = field(default_factory=list)

    # Deterministic flows only
    deterministic_response: str | None = None
    deterministic_action: str | None = None  # "transfer_to_operator"

    # Raw source
    source_path: str = ""


@dataclass
class FlowAdvanceResult:
    """Result of processing user input against the current flow step."""

    status: str  # "advanced", "repeated", "no_match", "completed", "redirect", "deterministic"
    next_step_id: str | None = None
    response_text: str | None = None
    skip_llm: bool = False
    redirect_flow_id: str | None = None
    action: str | None = None  # "end_flow", "transfer_to_operator"
    collected_field: str | None = None
    collected_value: Any | None = None
    step_context: str | None = None  # context to inject into LLM
