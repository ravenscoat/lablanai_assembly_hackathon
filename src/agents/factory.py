"""
AgentFactory: creates TenantAgent instances from TenantConfig.
Resolves tools from ToolRegistry, builds sub-agent handoff tools,
and constructs the fully-wired TenantAgent.
"""

from __future__ import annotations

import logging
from typing import Any

from livekit.agents import Agent, RunContext, function_tool

from config.schema import SubAgentConfig, TenantConfig
from tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

# Maps sub_agent type names to their classes
_SUB_AGENT_TYPES: dict[str, type] = {}


def _load_sub_agent_types() -> None:
    """Lazily load sub-agent type mappings."""
    global _SUB_AGENT_TYPES
    if _SUB_AGENT_TYPES:
        return
    try:
        from agents.sub_agents.appeal import AppealAgent

        _SUB_AGENT_TYPES["appeal"] = AppealAgent
    except ImportError:
        pass


_DEFAULT_SUMMARY_GUIDANCE_UZ = (
    "Summary uchun 2-3 gap yozing. Operator uchun ma'lumotga boy, uchinchi shaxs "
    "tomonidan yozilgandek bo'lsin. Misol: 'Chaqiruvchi biznes kengaytirish uchun "
    "kredit olishga harakat qilmoqda; Falon bankka topshirgan, rad etilgan; "
    "SME dasturlari bo'yicha yordam kerak.'"
)


def _build_probing_block(sub_config: "SubAgentConfig", language: str) -> str | None:
    """Return the probing-block system-prompt suffix to append to AppealAgent's
    instructions, or None if the tenant has no goals (legacy behavior).

    The block lists each goal's language-resolved description and tells the
    LLM to weave follow-up questions naturally, then call confirm_and_submit
    with a contextual summary.
    """
    goals = getattr(sub_config, "goals", None) or []
    if not goals:
        return None

    lines = [
        "",
        "## PROBING GOALS",
        "Before calling confirm_and_submit, learn the following from the caller "
        "through natural follow-up questions. Do NOT list them as a form — weave "
        "them into conversation.",
    ]
    for g in goals:
        hint = g.description.get(language, "") or next(iter(g.description.values()), "")
        marker = "(required)" if g.required else "(optional)"
        lines.append(f"- {g.key} {marker}: {hint}")

    summary_guidance = sub_config.summary_instructions.get(language) or _DEFAULT_SUMMARY_GUIDANCE_UZ
    lines.extend(
        [
            "",
            "## SUMMARY GUIDELINES",
            summary_guidance,
            "",
            "Once all required goals are covered, call confirm_and_submit with "
            "the `summary` argument set to your contextual summary.",
        ]
    )
    return "\n".join(lines)


def build_appeal_handoff_from_agent(
    current_agent,
    tenant_config: "TenantConfig",
    *,
    override_message: str | None = None,
    extra_instructions: str | None = None,
    sub_cls: type | None = None,
    sub_config: "SubAgentConfig | None" = None,
) -> tuple[object, str] | None:
    """Construct an AppealAgent from a live main agent for immediate handoff.

    Mirrors the state-forwarding done inside _create_handoff_tool's closure,
    so both the LLM-driven `transfer_to_appeal` path and the deterministic
    out-of-hours path in escalate_to_human produce the same sub-agent shape.

    Note: currently supports only the "appeal" sub-agent type. If a tenant
    ever declares additional sub-agent types, this helper needs to accept a
    sub_name parameter.

    The `sub_cls` and `sub_config` overrides exist so callers (and tests)
    can inject an alternate class / config instead of looking them up from
    the registered AppealAgent and tenant_config.sub_agents["appeal"];
    production code (e.g. the out-of-hours escalate path) should leave them
    as None.

    Returns (appeal_agent, message) or None if the tenant has no appeal
    sub-agent declared.
    """
    _load_sub_agent_types()
    appeal_cfg = sub_config or tenant_config.sub_agents.get("appeal")
    if not appeal_cfg or appeal_cfg.type != "appeal":
        return None
    cls = sub_cls or _SUB_AGENT_TYPES.get("appeal")
    if cls is None:
        return None

    agent_lang = getattr(current_agent, "language", None) or tenant_config.languages.default
    probing = _build_probing_block(appeal_cfg, agent_lang)

    base_instructions = appeal_cfg.instructions
    if extra_instructions:
        base_instructions = f"{base_instructions.rstrip()}\n\n{extra_instructions.strip()}"
    if probing:
        base_instructions = f"{base_instructions}\n{probing}"

    sub = cls(
        instructions=base_instructions,
        fields=appeal_cfg.fields,
        goals=appeal_cfg.goals,
        summary_instructions=appeal_cfg.summary_instructions,
        parent_config=tenant_config,
        api_endpoint=appeal_cfg.api_endpoint,
        messages=appeal_cfg.messages,
        language=agent_lang,
        chat_ctx=current_agent.chat_ctx if hasattr(current_agent, "chat_ctx") else None,
    )

    # State forwarding — identical to _create_handoff_tool's closure.
    if hasattr(current_agent, "_platform_client"):
        sub._platform_client = current_agent._platform_client
    if hasattr(current_agent, "call_db_id"):
        sub._parent_call_db_id = current_agent.call_db_id
    if hasattr(current_agent, "caller_phone"):
        sub._parent_caller_phone = current_agent.caller_phone
    if hasattr(current_agent, "_job_context"):
        sub._parent_job_context = current_agent._job_context
    sub._parent_agent = current_agent

    msg = (
        override_message
        or appeal_cfg.transfer_messages.get(agent_lang)
        or appeal_cfg.transfer_message
        or "Murojaat qabul qilishga yo'naltirmoqdaman..."
    )
    return sub, msg


class AgentFactory:
    """
    Creates TenantAgent instances from TenantConfig.

    The factory:
    1. Builds instructions from config
    2. Resolves platform + tenant tools from ToolRegistry
    3. Builds sub-agent handoff tools from config.sub_agents
    4. Constructs TenantAgent with all components
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        kb_manager: Any | None = None,
    ):
        self._tool_registry = tool_registry
        self._kb_manager = kb_manager

    def create_agent(
        self,
        config: TenantConfig,
        *,
        chat_ctx: Any | None = None,
        job_context: Any | None = None,
        platform_client: Any | None = None,
        current_language: str | None = None,
        is_post_handoff: bool = False,
        stt: Any | None = None,
        tts: Any | None = None,
        llm: Any | None = None,
    ) -> Agent:
        """
        Create a fully-wired TenantAgent from config.

        Args:
            config: Tenant configuration
            chat_ctx: Optional chat context (for handoff continuity)
            job_context: LiveKit JobContext
            current_language: Effective language for this agent instance. When
                provided, the caller has already chosen a language (post-handoff
                or returning caller with last_language on file).
            is_post_handoff: True when this agent is being created as the
                return target from a sub-agent (e.g. AppealAgent handing
                control back). When True, on_enter will NOT trigger an LLM
                reply — the sub-agent's return message is already playing
                and the next turn should wait for the user, preventing the
                LLM from hallucinating another sub-agent handoff.
        """
        from agents.tenant_agent import TenantAgent
        from resilience.error_recovery import ErrorRecoveryHandler

        effective_lang = current_language or config.languages.default
        # If current_language was explicitly passed, the caller has already
        # chosen (post-handoff, or returning caller with last_language on file).
        skip_prompt = current_language is not None

        # 1. Build instructions
        instructions = self._build_instructions(
            config,
            language=effective_lang,
            skip_language_prompt=skip_prompt,
        )

        # 2. Build error handler
        error_handler = ErrorRecoveryHandler(config)

        # 3. Resolve and add platform + tenant tools
        additional_tools = self._resolve_tools(config)

        # 4. Build sub-agent handoff tools
        handoff_tools = self._build_handoff_tools(config)

        # Per-language pipeline components
        from pipeline.voice_factory import VoiceFactory

        stt_inst = stt or VoiceFactory.create_stt_for_language(config, effective_lang)
        tts_inst = tts or VoiceFactory.create_tts_for_language(config, effective_lang)
        llm_inst = llm or VoiceFactory.create_llm(config)

        # 5. Create agent with all tools. STT/TTS/LLM are forwarded to the
        # base Agent's __init__ so they land in its `_stt`/`_tts`/`_llm`
        # slots before AgentActivity binding. Critical for multilingual
        # handoff — retrofitting these after construction doesn't rebind
        # the activity's cached STT reference.
        agent = TenantAgent(
            config=config,
            instructions=instructions,
            extra_tools=additional_tools + handoff_tools,
            chat_ctx=chat_ctx,
            job_context=job_context,
            kb_manager=self._kb_manager,
            error_handler=error_handler,
            current_language=effective_lang,
            is_post_handoff=is_post_handoff,
            stt=stt_inst,
            tts=tts_inst,
            llm=llm_inst,
        )

        agent._platform_client = platform_client

        # 6. Initialize FlowEngine if flows enabled
        flow_count = 0
        if config.flows.enabled and config.flows.flow_dir:
            from pathlib import Path

            from flows.engine import FlowEngine

            flow_engine = FlowEngine(
                tenant_id=config.tenant.id,
                flow_dir=Path(config.flows.flow_dir),
            )
            flow_count = flow_engine.load_all_flows()
            agent._flow_engine = flow_engine

        logger.info(
            f"Created agent for {config.tenant.slug}: "
            f"{len(additional_tools)} resolved + "
            f"{len(handoff_tools)} handoffs + "
            f"{flow_count} flows, "
            f"lang={effective_lang}"
        )

        return agent

    def _build_instructions(
        self,
        config: TenantConfig,
        *,
        language: str | None = None,
        skip_language_prompt: bool = False,
    ) -> str:
        """Build full agent instructions from config.

        Args:
            language: effective language for this agent instance (default or
                caller-selected). Drives the response-format tail.
            skip_language_prompt: when True, omit the MULTILINGUAL POLICY block.
                Caller must already have chosen a language — post-handoff path,
                or returning caller whose language is on file.
        """
        lang = language or config.languages.default
        parts = []

        # System prompt
        if config.personality.system_prompt:
            parts.append(config.personality.system_prompt.strip())

        # Transfer policy — two-stage flow: request_escalation → reflection
        # prompt → confirm_escalation. Lenient reasons (out_of_scope,
        # psychological_confirmed) skip reflection. A 3-strike deterministic
        # ceiling forces the transfer through if the LLM keeps requesting
        # without confirming.
        if config.transfer.enabled:
            phrases = ", ".join(f'"{p}"' for p in config.transfer.trigger_phrases)
            # TODO(urdu): this transfer/escalation policy block is hard-coded in
            # Uzbek. Add an 'ur' (and 'en') branch with Urdu policy text here —
            # e.g. select the fragment by `lang` like _csat_policy_instructions
            # below. The free-text personality.system_prompt/greeting can be
            # Urdu immediately; only this framework fragment needs translation.
            parts.append(
                f"\n## OPERATORGA YO'NALTIRISH:\n"
                f"Foydalanuvchi {phrases} desa yoki siz javob bera olmasangiz:\n"
                f"1. Avval request_escalation(reason_for_transfer=...) toolini chaqiring.\n"
                f"2. Strict sabablar (kb_miss, user_dissatisfied) uchun tool sizga "
                f"INTERNAL CHECK matnini qaytaradi. Bu matn FAQAT siz uchun — uni "
                f"foydalanuvchiga aytmang, ovoz chiqarib o'qimang, undan iqtibos "
                f"keltirmang.\n"
                f"3. Agar urinib ko'rgan bo'lsangiz va foydalanuvchi baribir "
                f'operator istasa — DARHOL confirm_escalation(reasoning="...") '
                f"toolini chaqiring. Tool chaqirishdan oldin yoki keyin "
                f'"men qidirdim", "men urinib ko\'rdim", "keling tekshirib '
                f'ko\'ray", "sizga ma\'lumot bergan edim" kabi gaplarni AYTMANG. '
                f"Reflection o'z-o'zingiz uchun, ovoziga chiqarmaslik kerak. "
                f'Faqat "Sizni operatorga ulayman, kuting" deyishingiz mumkin.\n'
                f"4. Agar urinib ko'rmagan bo'lsangiz — avval KB qidiring yoki "
                f"boshqa yo'l taklif qiling, keyin yangidan request_escalation "
                f"chaqiring.\n"
                f"Lenient sabablar (out_of_scope, psychological_confirmed) uchun "
                f"reflection o'tkazib yuboriladi — request_escalation darhol "
                f"o'tkazadi."
            )
            # Guardian — make the LLM try to help before escalating (NAV-150).
            guardian = config.transfer.guardian
            if guardian and guardian.operator_instructions:
                parts.append(
                    "\n## OPERATOR TRANSFER GUARDIAN:\n" + guardian.operator_instructions.strip()
                )
            if guardian and guardian.psychological_instructions:
                parts.append(
                    "\n## PSYCHOLOGICAL TRANSFER GUARDIAN:\n"
                    + guardian.psychological_instructions.strip()
                )

        # Flow + KB policy (flows take priority over KB)
        has_flows = False
        if config.flows.enabled and config.flows.flow_dir:
            # Use cached flows if available (avoid double parsing)
            if hasattr(self, "_cached_flows") and self._cached_flows:
                flows = self._cached_flows
            else:
                from pathlib import Path

                from flows.parser import FlowParser

                flows = FlowParser.parse_all(Path(config.flows.flow_dir))
                self._cached_flows = flows

            if flows:
                has_flows = True
                # TODO(urdu): this flow tool-selection policy block is hard-coded
                # in Uzbek. Add an 'ur' (and 'en') language branch with Urdu
                # policy text here, keyed off `lang`.
                flow_lines = ["\n## MUHIM: TOOL TANLASH TARTIBI"]
                flow_lines.append(
                    "Foydalanuvchi quyidagi mavzulardan biri haqida gapirganda, "
                    "DARHOL start_flow toolini chaqiring. Tasdiqlash SO'RAMANG. "
                    "search_knowledge_base NI ISHLATMANG."
                )
                flow_lines.append("\nMavjud jarayonlar:")
                for flow in flows.values():
                    flow_lines.append(f"- {flow.id}: {flow.title}")
                flow_lines.append(
                    "\nQOIDALAR:"
                    "\n1. Foydalanuvchi jarayon mavzusi haqida so'raganda â†’ DARHOL start_flow chaqiring"
                    "\n2. Foydalanuvchidan tasdiq SO'RAMANG ('boshlashni xohlaysizmi?' DEMANG)"
                    "\n3. start_flow qaytargan matnni foydalanuvchiga tabiiy tarzda ayting"
                    "\n4. Agar jarayon faol bo'lsa, faqat [Flow step] ko'rsatmalariga amal qiling"
                    "\n5. search_knowledge_base faqat jarayonlarga tegishli BO'LMAGAN savollar uchun ishlating"
                    "\n6. HECH QACHON tool nomlarini ovoz chiqarib aytmang (start_flow, search_knowledge_base va h.k.)"
                    "\n7. Foydalanuvchiga texnik atamalar yoki tool nomlari ko'rsatmang"
                )
                parts.append("\n".join(flow_lines))

        if config.knowledge_base.enabled:
            # TODO(urdu): this KB-usage policy block is hard-coded in Uzbek.
            # Add an 'ur' (and 'en') language branch with Urdu policy text here,
            # keyed off `lang`.
            if has_flows:
                parts.append(
                    "\n## BILIMLAR BAZASI:\n"
                    "Faqat jarayonlarga tegishli BO'LMAGAN savollar uchun "
                    "search_knowledge_base toolini ishlating. "
                    "Bilimlar bazasida topilmagan ma'lumotni o'ylab topmang."
                )
            else:
                parts.append(
                    "\n## BILIMLAR BAZASI:\n"
                    "Savolga javob berish uchun search_knowledge_base toolini ishlating. "
                    "Bilimlar bazasida topilmagan ma'lumotni o'ylab topmang."
                )

        # CSAT policy (NAV-156) — universal: every call collects a 1-5 rating
        # before end_call. Skipped automatically when the LLM transfers to a
        # human (escalation tools fire instead of end_call, so the rule never
        # triggers); refusals / non-numeric answers / hangups are graceful
        # nulls because the tool is never called and shutdown omits csat from
        # metadata.
        parts.append(self._csat_policy_instructions(lang))

        # Language-selection preamble for multilingual tenants — only for
        # callers who have not yet chosen a language.
        if config.languages.is_multilingual and not skip_language_prompt:
            # TODO(urdu): the MULTILINGUAL / LANGUAGE POLICY blocks below are
            # hard-coded in Russian/Uzbek. Add Urdu ('ur') phrasing for a
            # bilingual Urdu tenant. `langs_human` includes an 'ur' label.
            langs_human = {"uz": "o'zbek", "ru": "русский", "ur": "اردو", "en": "English"}
            listed = ", ".join(langs_human.get(c, c) for c in config.languages.available)
            parts.insert(
                0,
                "## MULTILINGUAL POLICY (first-time caller):\n"
                "**The system has ALREADY played the bilingual opening greeting** "
                "(recorded audio, asking the caller which language to use). "
                f"Supported languages: {listed}. Do NOT greet again or repeat the "
                "question — trust that the caller heard it.\n\n"
                "When the caller responds:\n"
                "- If they say a language name (even a single word like "
                '"русский" / "рус" / "o\'zbek" / "uzbek" / "узбекский"), '
                "call the `select_language` tool with the ISO code ('ru' or 'uz'). "
                "Do NOT call any other tool before select_language fires.\n"
                "- If the response is unclear / off-topic, re-ask **ONLY in the "
                "default language (Russian)** with one short sentence: "
                '"На каком языке вам удобнее — на русском или узбекском?" '
                "Do NOT repeat the full bilingual greeting.\n\n"
                "**Do NOT output mixed bilingual replies in one turn after the opening greeting.** "
                "Use one language per reply (Russian until select_language is called).\n\n"
                "**After select_language fires, do NOT acknowledge the switch again.** "
                "The system already plays confirmation and a single "
                '"how can I help" prompt in the selected language. '
                "Your next spoken response should answer the caller's next question.\n",
            )
        elif config.languages.is_multilingual and skip_language_prompt:
            # Returning caller / post-handoff: still tell the LLM the caller
            # can switch if they ask.
            parts.insert(
                0,
                "## LANGUAGE POLICY:\n"
                f"Respond in {lang!r}. If the caller asks to switch language "
                '(e.g. "Ð´Ð°Ð²Ð°Ð¹ Ð½Ð° ÑƒÐ·Ð±ÐµÐºÑÐºÐ¾Ð¼" / "ruschaga o\'tamiz"), call the '
                "`select_language` tool with the requested ISO code.\n\n"
                "**After select_language fires, STAY SILENT.** The system plays a "
                "pre-rendered confirmation in the correct voice — do NOT generate "
                'any acknowledgment like "OK, I\'ll continue in Uzbek". Your next '
                "spoken response should be the answer to the caller's *next* "
                "question, NOT an acknowledgment of the switch.\n",
            )

        # Response format — localized to the effective language.
        parts.append(self._response_format_instructions(lang))

        return "\n\n".join(parts)

    @staticmethod
    def _csat_policy_instructions(language: str) -> str:
        """Return the CSAT-collection rule in the given language (NAV-156).

        Unknown codes fall back to Uzbek.
        """
        if language == "ur":
            # TODO(urdu): add the CSAT-collection rule in Urdu here. For now we
            # fall back to the Uzbek default so behavior stays intact. Replace
            # this branch with a proper Urdu fragment when wiring an Urdu tenant.
            pass
        if language == "ru":
            return (
                "\n## ПРАВИЛО CSAT:\n"
                "Перед вызовом end_call ВСЕГДА просите пользователя оценить разговор: "
                '"Оцените, пожалуйста, наш разговор от 1 до 5."\n'
                "- Если пользователь называет число (1, 2, 3, 4 или 5) → вызовите "
                "collect_csat(rating=N), затем end_call.\n"
                '- Если пользователь отказывается ("не нужно", "тороплюсь") → '
                "пропустите CSAT и сразу вызывайте end_call.\n"
                '- Если ответ не число ("хорошо", "нормально") → уточните один раз; '
                "если число так и не названо, пропустите CSAT.\n"
                "- При переводе на оператора (request_escalation, confirm_escalation, "
                "transfer_to_*) CSAT НЕ спрашивайте — оценка собирается только перед end_call.\n"
                "Никогда не придумывайте число сами."
            )
        return (
            "\n## CSAT QOIDASI:\n"
            "end_call dan oldin DOIMO foydalanuvchidan suhbatni baholashni so'rang: "
            '"Suhbatimizni 1 dan 5 gacha baholang."\n'
            "- Foydalanuvchi raqam (1, 2, 3, 4 yoki 5) aytsa → collect_csat(rating=N) "
            "chaqiring, keyin end_call.\n"
            '- Rad etsa ("kerak emas", "shoshib turibman") → CSAT o\'tkazib yuboring, '
            "darhol end_call.\n"
            '- Raqam emas javob bersa ("yaxshi", "soz") → 1 marta aniqlashga harakat '
            "qiling; baribir raqam aytmasa, o'tkazib yuboring.\n"
            "- Operatorga ulanish (request_escalation, confirm_escalation, transfer_to_*) "
            "holatlarida CSAT SO'RAMANG — baholash faqat end_call dan oldin.\n"
            "Hech qachon o'zingiz raqam o'ylab topmang."
        )

    @staticmethod
    def _response_format_instructions(language: str) -> str:
        """Return the response-format appendix in the given language.

        Unknown codes fall back to Uzbek so existing single-language tenants
        keep their current behavior.
        """
        if language == "ur":
            # TODO(urdu): add the response-format guidance in Urdu here. For now
            # we fall back to the Uzbek default so behavior stays intact.
            pass
        if language == "ru":
            return (
                "\n## ФОРМАТ ОТВЕТА:\n"
                "- Ответы должны быть короткими и адаптированными для голосового общения\n"
                "- Не используйте emoji, *, #, -\n"
                "- Произносите числа словами\n"
                "- Минимальная пунктуация"
            )
        # Default: Uzbek
        return (
            "\n## JAVOB FORMATI:\n"
            "- Javoblar qisqa va ovozli kommunikatsiya uchun moslashtirilgan bo'lsin\n"
            "- Emoji, *, #, - belgilarini ishlatmang\n"
            "- Raqamlarni so'z bilan ayting\n"
            "- Punktuatsiya minimal bo'lsin"
        )

    def _resolve_tools(self, config: TenantConfig) -> list:
        """Resolve tool names to actual function_tool instances."""
        tools = []

        tool_names = list(config.tools.platform + config.tools.tenant)
        if config.languages.is_multilingual and "select_language" not in tool_names:
            tool_names.append("select_language")
        # request_escalation and confirm_escalation are a pair — either both
        # are exposed or neither. If YAML lists one but not the other, attach
        # the missing one automatically so the two-stage flow can complete.
        if "request_escalation" in tool_names and "confirm_escalation" not in tool_names:
            tool_names.append("confirm_escalation")
        # collect_csat is universal (NAV-156): every tenant collects a 1-5
        # rating before end_call. Auto-included so tenant YAMLs stay clean.
        if "collect_csat" not in tool_names:
            tool_names.append("collect_csat")

        for tool_name in tool_names:
            factory = self._tool_registry.resolve(tool_name)
            if factory:
                try:
                    tool = factory(config, kb_manager=self._kb_manager)
                    if (
                        tool is not None
                    ):  # Skip noop factories (e.g. select_language on single-lang)
                        tools.append(tool)
                except Exception as e:
                    logger.error(f"Failed to create tool {tool_name}: {e}")

        return tools

    def _build_handoff_tools(self, config: TenantConfig) -> list:
        """Build function_tool wrappers for sub-agent handoffs."""
        _load_sub_agent_types()
        tools = []

        for name, sub_config in config.sub_agents.items():
            agent_cls = _SUB_AGENT_TYPES.get(sub_config.type)
            if not agent_cls:
                logger.warning(f"Unknown sub-agent type: {sub_config.type} for {name}")
                continue

            tool = self._create_handoff_tool(name, sub_config, agent_cls, config)
            if tool:
                tools.append(tool)

        return tools

    def _create_handoff_tool(
        self,
        name: str,
        sub_config: SubAgentConfig,
        agent_cls: type,
        parent_config: TenantConfig,
    ):
        """Create a function_tool that performs agent handoff."""
        tool_name = f"transfer_to_{name}"
        # Prefer the explicit `description` (WHEN to call it) over
        # `instructions` (which is the sub-agent's own system prompt — it
        # tells the sub-agent what to do, not the main agent when to call).
        # Fall back to a short per-type default so the LLM gets *some*
        # trigger guidance instead of a vague "transfer to foo" line.
        _DEFAULT_DOCS = {
            "appeal": (
                "Call ONLY when the user wants to file a formal appeal/complaint "
                "(shikoyat/murojaat) — e.g. corruption, legal violation, or a "
                "concrete issue the knowledge base cannot resolve. Do NOT call "
                "for general questions, operator requests, psychological support, "
                "or after an appeal has already been submitted in this call."
            ),
        }
        doc = (
            sub_config.description.strip()
            or _DEFAULT_DOCS.get(sub_config.type)
            or f"Transfer to {name} sub-agent."
        )

        # Closure captures — cfg and cls are threaded through to the helper
        # so the LLM-driven handoff uses the exact SubAgentConfig/class this
        # tool was built for, independent of tenant_config.sub_agents lookup.
        pcfg = parent_config
        cfg = sub_config
        cls = agent_cls

        async def _do_handoff(context: RunContext):
            agent = context.session.current_agent
            existing_murojaat_id = getattr(agent, "murojaat_id", None)
            has_existing_murojaat = isinstance(existing_murojaat_id, str) and bool(
                existing_murojaat_id.strip()
            )
            if has_existing_murojaat:
                return cls._DUPLICATE_REJECT_MESSAGE
            extra = (
                "APPEAL RESPONSE POLICY:\n"
                "Keep appeal flow strictly focused on collecting and submitting the appeal.\n"
                "Do NOT mention operator availability, transfer status, office hours, shift schedule, "
                "callback timing promises, or internal system behavior.\n"
                "Do NOT add extra explanations beyond the required next question or summary confirmation.\n"
                "Ask for confirmation once, then call confirm_and_submit once.\n"
                "After success is spoken, do not re-confirm and do not submit again in this call.\n"
                "\n"
                "PHONE NUMBER POLICY:\n"
                "Only address the phone number if the caller is offering to dictate, "
                "spell out, or confirm their own number. In that case, reassure them: "
                '"raqamingiz bizda bor, tashvishlanmang" (we have your number, do not worry). '
                "Do not bring up, repeat, or read back the phone number in any other context."
            )

            result = build_appeal_handoff_from_agent(
                agent,
                pcfg,
                extra_instructions=extra,
                sub_cls=cls,
                sub_config=cfg,
            )
            if result is None:
                return f"Murojaat yordamchisi sozlanmagan: {name}"
            return result

        _do_handoff.__name__ = tool_name
        _do_handoff.__doc__ = doc

        return function_tool(name=tool_name)(_do_handoff)


# Singleton factory
_factory: AgentFactory | None = None


def get_factory() -> AgentFactory:
    """Get singleton AgentFactory instance."""
    global _factory
    if _factory is None:
        from tools.registry import get_tool_registry

        _factory = AgentFactory(tool_registry=get_tool_registry())
    return _factory


def set_factory(factory: AgentFactory) -> None:
    """Set the singleton AgentFactory (for initialization with KB manager)."""
    global _factory
    _factory = factory
