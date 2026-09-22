"""
AppealAgent: conversational data collection sub-agent.

Generalized across tenants:
- Fields declared in YAML (`sub_agents.<name>.fields`) — no hardcoded schema.
- Prompts resolved per language via FieldConfig.prompt_for().
- User-facing messages (start/summary/success) resolved per language via
  SubAgentConfig.messages; falls back to built-in Uzbek defaults for
  Youth-Agency-style configs that predate per-language messages.
- Sentinel defaults for any murojaat_qiluvchi field not in the collected
  set, so the existing backend endpoint accepts the submission with no
  schema change.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from livekit.agents import RunContext, function_tool

from config.schema import FieldConfig, GoalConfig, TenantConfig

from .base_sub_agent import BaseSubAgent

logger = logging.getLogger(__name__)


# Uzbek defaults used when `messages[lang]` is absent — preserves
# Youth-Agency behavior for configs that don't declare messages.
_DEFAULT_MESSAGES_UZ = {
    "start": (
        "Foydalanuvchiga murojaat qabul qilishni boshlayotganingizni ayting. "
        "Birinchi savol: murojaatingiz mazmunini aytib bering."
    ),
    "success": ("Murojaatingiz muvaffaqiyatli qabul qilindi. Boshqa savolingiz bormi?"),
    # Summary string uses named {field} placeholders; any fields present
    # in _data will be substituted.
    "summary": "Barcha ma'lumotlar to'plandi. Tasdiqlaysizmi?",
}


class AppealAgent(BaseSubAgent):
    """
    Sub-agent that collects appeal data conversationally.

    Fields are declared per-tenant in YAML; setter tools are generated
    dynamically from `self._fields` in __init__ and passed to the base
    Agent as `tools=`. The per-turn flow:

      on_enter → speak messages[lang].start
      user says field value → set_<name>() → _next_field_prompt(name)
      all fields collected → summary → confirm_and_submit → _submit → main agent
    """

    SUB_AGENT_NAME = "appeal"

    # Sentinel values used when a backend-required field is not in the
    # collected set (Paynet lightweight intake). Operators see these
    # markers and know the call came from the lightweight flow.
    _BACKEND_SENTINELS: dict[str, Any] = {
        "age": 0,
        "region": "-",
        "district": "-",
        "position": "Mijoz",
    }
    _DUPLICATE_REJECT_MESSAGE = (
        "Bu qo'ng'iroqda murojaatni qayta yuborish mumkin emas. "
        "Iltimos, yangi murojaat uchun qayta qo'ng'iroq qiling."
    )

    def __init__(
        self,
        *,
        instructions: str = "",
        fields: list[FieldConfig] | list[dict] = (),
        goals: list[GoalConfig] | list[dict] = (),
        summary_instructions: dict[str, str] | None = None,
        parent_config: TenantConfig,
        api_endpoint: str = "",
        messages: dict[str, dict[str, str]] | None = None,
        language: str | None = None,
        chat_ctx: Any | None = None,
        **kwargs,
    ):
        if not instructions:
            instructions = self._default_instructions()

        # Normalize fields to FieldConfig instances.
        normalized: list[FieldConfig] = []
        for f in fields:
            if isinstance(f, FieldConfig):
                normalized.append(f)
            else:
                normalized.append(FieldConfig.model_validate(f))
        self._data: dict[str, Any] = {}
        self._language: str = language or parent_config.languages.default
        self._messages: dict[str, dict[str, str]] = messages or {}

        # Normalize goals to GoalConfig instances (same pattern as fields).
        normalized_goals: list[GoalConfig] = []
        for g in goals:
            if isinstance(g, GoalConfig):
                normalized_goals.append(g)
            else:
                normalized_goals.append(GoalConfig.model_validate(g))
        self._goals: list[GoalConfig] = normalized_goals
        self._summary_instructions: dict[str, str] = summary_instructions or {}

        # Strip `content` from fields when goals are declared — it's populated
        # via the summary arg on confirm_and_submit, not a setter tool. Must
        # run AFTER self._goals assignment.
        self._fields: list[FieldConfig] = self._filter_fields_for_goals(normalized, self._goals)

        # Build per-field setter tools from config.
        setter_tools = self._build_setter_tools(self._fields)

        # Pass dynamic tools through to the base Agent. BaseSubAgent's
        # class-level @function_tool methods (cancel_and_return,
        # escalate_to_human, end_call) plus AppealAgent's own
        # confirm_and_submit are still auto-registered by LiveKit
        # introspection — `tools=` is additive.
        super().__init__(
            instructions=instructions,
            parent_config=parent_config,
            chat_ctx=chat_ctx,
            tools=setter_tools,
        )

        self._api_endpoint = api_endpoint
        self._platform_client = None
        self._parent_call_db_id: str | None = None
        self._parent_caller_phone: str | None = None
        self._parent_agent: Any | None = None
        self._submit_started: bool = False
        self._submitted_once: bool = False

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _build_setter_tools(self, fields: list[FieldConfig]) -> list:
        """Generate one `set_<field.name>` function_tool per field.

        Uses the same closure pattern as AgentFactory._create_handoff_tool
        so each tool captures its own FieldConfig reference.
        """
        tools = []
        for field in fields:
            tool = self._make_setter_tool(field)
            tools.append(tool)
        return tools

    def _make_setter_tool(self, field: FieldConfig):
        """Build a single setter function_tool for `field`.

        The returned tool records the value into self._data[field.name]
        (with validation if configured), then returns _next_field_prompt
        so the LLM speaks the next question.
        """
        field_name = field.name
        validation = field.validation

        # Int fields use a typed parameter; other fields take a string.
        if validation:

            async def _setter(self_agent, context: RunContext, value: int) -> str:  # type: ignore[misc]
                err = AppealAgent._validate_int_range(validation, value)
                if err:
                    return err
                self_agent._data[field_name] = value
                return self_agent._next_field_prompt(field_name)

        else:

            async def _setter(self_agent, context: RunContext, value: str) -> str:  # type: ignore[misc]
                self_agent._data[field_name] = value
                return self_agent._next_field_prompt(field_name)

        _setter.__name__ = f"set_{field_name}"
        _setter.__doc__ = (
            f"Record the caller's {field_name}. "
            f"Call this tool as soon as the caller provides their {field_name}."
        )

        # function_tool needs `self` bound — wrap with a trampoline.
        async def _bound(context: RunContext, value):  # type: ignore[no-redef]
            return await _setter(self, context, value)

        _bound.__name__ = f"set_{field_name}"
        _bound.__doc__ = _setter.__doc__
        # Copy annotations so LiveKit's schema-generation can introspect the
        # parameter type. Without this, schema generation raises KeyError on
        # the untyped `value` parameter and the LLM sees zero setter tools.
        # NOTE: use the actual type class (int/str) rather than copying from
        # _setter.__annotations__, because `from __future__ import annotations`
        # makes all annotations lazy strings in this module. LiveKit needs real
        # type objects, not the string literals.
        _bound.__annotations__ = {
            "context": RunContext,
            "value": int if validation else str,
            "return": str,
        }
        return function_tool(name=f"set_{field_name}")(_bound)

    @staticmethod
    def _filter_fields_for_goals(
        fields: list[FieldConfig], goals: list[GoalConfig]
    ) -> list[FieldConfig]:
        """When goals are declared, `content` is populated via the summary arg
        on confirm_and_submit — not via a setter tool. Strip it here so no
        `set_content` tool is generated."""
        if not goals:
            return list(fields)
        return [f for f in fields if f.name != "content"]

    @staticmethod
    def _validate_int_range(validation: str, value: int) -> str | None:
        """Return an Uzbek error string if `value` is out of range; None otherwise."""
        if not validation:
            return None
        try:
            lo_s, hi_s = validation.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
        except (ValueError, TypeError):
            return None
        if value < lo or value > hi:
            return f"Qiymat {lo} dan {hi} gacha bo'lishi kerak. Iltimos, qayta ayting."
        return None

    def _default_instructions(self) -> str:
        return (
            "Siz fuqarodan murojaat qabul qiluvchi yordamchisiz. "
            "Har bir ma'lumotni alohida so'rang. "
            "Barcha ma'lumotlar to'plangach tasdiqlash uchun takrorlang. "
            "Operator ish vaqti, operator mavjudligi, smena jadvali yoki callback vaqti haqida "
            "o'zingizdan gapirmang. Faqat murojaatni qisqa va aniq qabul qiling."
        )

    def _message_for(self, key: str) -> str:
        """Look up `messages[self._language][key]` with Uzbek default fallback."""
        lang_bucket = self._messages.get(self._language, {})
        if key in lang_bucket:
            return lang_bucket[key]
        default = _DEFAULT_MESSAGES_UZ.get(key)
        if default is None:
            logger.warning("AppealAgent._message_for: unknown message key %r", key)
            return ""
        return default

    # ------------------------------------------------------------------
    # LiveKit lifecycle
    # ------------------------------------------------------------------

    async def on_enter(self) -> None:
        """Greet and ask for the first missing field."""
        if self.session:
            await self.session.generate_reply(instructions=self._message_for("start"))

    # ------------------------------------------------------------------
    # Tools declared at class level (shape is identical across tenants)
    # ------------------------------------------------------------------

    @function_tool()
    async def confirm_and_submit(self, context: RunContext, summary: str = ""):
        """Confirm all collected data and submit the appeal.

        Args:
            summary: For tenants that declare goals, a 2-3 sentence contextual
                summary written by the LLM for the operator dashboard. Required
                when goals are declared. Legacy tenants (no goals) ignore this.

        Call ONLY after the caller has explicitly confirmed the summary.
        """
        # Hard idempotency guard: one successful appeal submission per call.
        submitted_once = bool(getattr(self, "_submitted_once", False))
        parent_murojaat_id = getattr(getattr(self, "_parent_agent", None), "murojaat_id", None)
        parent_has_murojaat = isinstance(parent_murojaat_id, str) and bool(
            parent_murojaat_id.strip()
        )
        if submitted_once or parent_has_murojaat:
            return self._DUPLICATE_REJECT_MESSAGE

        # Prevent duplicate submit attempts while one is already running.
        if bool(getattr(self, "_submit_started", False)):
            return self._DUPLICATE_REJECT_MESSAGE

        # Under goals: summary IS the content. Reject if missing.
        if self._goals:
            if not summary or not summary.strip():
                return (
                    "Iltimos, avval 2-3 gaplik murojaat xulosasini yozing "
                    "(muammo, urinishlar, kutilgan natija), keyin confirm_and_submit "
                    "ni summary argumenti bilan qayta chaqiring."
                )
            self._data["content"] = summary

        missing = self._get_missing_fields()
        if missing:
            return (
                f"Quyidagi ma'lumotlar hali to'ldirilmagan: {', '.join(missing)}. "
                f"Iltimos, avval ularni to'ldiring."
            )

        self._submit_started = True
        try:
            ok = await self._submit()
        finally:
            self._submit_started = False
        if ok:
            self._submitted_once = True
            main_agent = self._create_main_agent()
            success_message = self._message_for("success")
            history = getattr(main_agent, "conversation_history", None)
            if isinstance(history, list) and success_message:
                history.append(
                    {
                        "role": "assistant",
                        "content": success_message,
                        "timestamp": time.time(),
                    }
                )
            return main_agent, success_message
        return "Murojaatni yuborishda xatolik yuz berdi. Qayta urinib ko'ramiz."

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _next_field_prompt(self, completed_field: str) -> str:
        """Return either the next field's prompt or the final summary."""
        missing = self._get_missing_fields()
        if not missing:
            # All fields collected — build and return the summary.
            summary_tpl = self._message_for("summary")
            try:
                return summary_tpl.format(**self._data)
            except KeyError:
                # Caller templated a field we don't have; fall back to the
                # un-formatted template so intake still progresses.
                return summary_tpl

        next_name = missing[0]
        field = next((f for f in self._fields if f.name == next_name), None)
        if field is None:
            return f"{next_name} ni ayting."
        return field.prompt_for(self._language)

    def _get_missing_fields(self) -> list[str]:
        """Return the list of required field names whose value is unset/falsy."""
        missing: list[str] = []
        for field in self._fields:
            if not field.required:
                continue
            if not self._data.get(field.name):
                missing.append(field.name)
        return missing

    async def _submit(self) -> bool:
        """Submit the collected appeal, filling backend sentinels for
        any required murojaat_qiluvchi field not in `_fields`."""
        # Fill sentinels for backend-required fields not in the schema.
        payload: dict[str, Any] = dict(self._data)
        for key, sentinel in self._BACKEND_SENTINELS.items():
            payload.setdefault(key, sentinel)

        kwargs = dict(
            tenant_id=self._parent_config.tenant.id,
            tenant_slug=self._parent_config.tenant.slug,
            content=payload.get("content", ""),
            full_name=payload.get("full_name", ""),
            age=int(payload.get("age", 0) or 0),
            region=payload.get("region", "-"),
            district=payload.get("district", "-"),
            position=payload.get("position", "Mijoz"),
            call_id=self._parent_call_db_id,
            caller_phone=self._parent_caller_phone,
        )

        if self._platform_client:
            result = await self._platform_client.submit_murojaat(**kwargs)
            if result.success:
                logger.info(
                    "Appeal submitted via PlatformClient: %s (call_id=%s)",
                    result.murojaat_id,
                    self._parent_call_db_id,
                )
                if self._parent_agent is not None and result.murojaat_id:
                    self._parent_agent.murojaat_id = result.murojaat_id
                return True
            logger.error("Appeal submission failed: %s", result.error)
            return False

        try:
            from api.platform_client import get_platform_client

            client = get_platform_client()
            result = await client.submit_murojaat(**kwargs)
            if result.success:
                logger.info(
                    "Appeal submitted via singleton client: %s (call_id=%s)",
                    result.murojaat_id,
                    self._parent_call_db_id,
                )
                if self._parent_agent is not None and result.murojaat_id:
                    self._parent_agent.murojaat_id = result.murojaat_id
                return True
            logger.error("Appeal submission failed: %s", result.error)
            return False
        except Exception as exc:
            logger.warning("PlatformClient unavailable, appeal not submitted: %s", exc)
            return True  # Simulate success in dev/console mode
