"""
Pydantic models for tenant configuration.
Defines the complete schema for YAML-based tenant configs.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class TenantIdentity(BaseModel):
    id: str
    slug: str = ""
    name: str = "Voice Agent"
    phone_numbers: list[str] = []


class PersonalityConfig(BaseModel):
    system_prompt: str = ""
    greeting: str = "Assalomu alaykum! Sizga qanday yordam bera olaman?"
    # Controls interruption behavior ONLY for opening greeting playback.
    # Other speech keeps using existing behavior defaults.
    greeting_allow_interruptions: bool = True
    # Per-language monolingual greetings used for returning callers whose
    # language is already known. Keys are ISO codes ("ru", "uz").
    greetings: dict[str, str] = Field(default_factory=dict)
    # Per-language text for the FIRST-TIME bilingual opening. Each half is
    # rendered by scripts/generate_greeting_audio.py to a WAV file using the
    # language-specific TTS voice; at runtime on_enter plays those files
    # sequentially so Russian text gets read by the Russian voice and Uzbek
    # text by the Uzbek voice. Keys are ISO codes.
    greeting_parts: dict[str, str] = Field(default_factory=dict)
    # Per-language pre-rendered audio file paths (WAV, 16 kHz mono PCM),
    # produced from greeting_parts by scripts/generate_greeting_audio.py.
    # When all keys present in languages.available have a readable file, the
    # agent plays them in languages.available order and skips TTS for the
    # opening. Missing files -> falls back to `greeting` via TTS.
    greeting_audio: dict[str, str] = Field(default_factory=dict)
    # Language-switch confirmation text + pre-rendered audio per language.
    # When select_language fires, the TARGET language's TTS isn't active on
    # the session yet — LiveKit's activity swap is async and gated on the
    # current speech completing. A naive post-tool LLM reply gets spoken by
    # the OLD voice (yulduz_ru reading Uzbek text with a Russian accent). We
    # pre-render the confirmation so the tool can play the correct voice
    # IMMEDIATELY before returning the handoff.
    language_switch_text: dict[str, str] = Field(default_factory=dict)
    language_switch_audio: dict[str, str] = Field(default_factory=dict)
    fallback: str = "Kechirasiz, bu savolga javob bera olmayapman."
    goodbye: str = "Xayr! Yana savollaringiz bo'lsa, qo'ng'iroq qiling."

    def greeting_for(self, language: str) -> str:
        """Return the monolingual greeting for `language`, falling back to `greeting`."""
        return self.greetings.get(language) or self.greeting

    def greeting_part_for(self, language: str) -> str:
        """Return the first-time bilingual-opening part for `language`, or ''."""
        return self.greeting_parts.get(language, "")

    def greeting_audio_for(self, language: str) -> str:
        """Return the pre-rendered audio path for `language`'s opening, or ''."""
        return self.greeting_audio.get(language, "")

    def language_switch_text_for(self, language: str) -> str:
        """Return the confirmation text for a select_language handoff to `language`."""
        return self.language_switch_text.get(language, "")

    def language_switch_audio_for(self, language: str) -> str:
        """Return the pre-rendered confirmation audio path for `language`, or ''."""
        return self.language_switch_audio.get(language, "")


class VoiceConfig(BaseModel):
    stt_provider: str = "yandex"
    # Per-language STT provider overrides, used when languages.available > 1.
    # Example: {"ru": "yandex", "uz": "custom"}.
    stt_providers: dict[str, str] = Field(default_factory=dict)
    # Optional STT override used only for the first post-greeting language-choice turn.
    greeting_stt_provider: str | None = None
    # Allowed language codes for greeting-phase STT detection (e.g. ["ru", "uz"]).
    greeting_stt_languages: list[str] = Field(default_factory=list)
    tts_provider: str = "yandex"
    tts_voice_id: str = "yulduz"
    speed: float = 1.0
    pitch: float = 1.0
    # Per-language voice_id overrides, used when languages.available > 1.
    voices: dict[str, str] = Field(default_factory=dict)

    def voice_for(self, language: str) -> str:
        """Return the TTS voice id for `language`, falling back to tts_voice_id."""
        return self.voices.get(language, self.tts_voice_id)

    def stt_provider_for(self, language: str) -> str:
        """Return STT provider for `language`, falling back to global stt_provider."""
        return self.stt_providers.get(language, self.stt_provider)


class LLMConfig(BaseModel):
    provider: str = "gemini"
    model: str = "gemini-3.5-flash"
    temperature: float = 0.2
    max_tokens: int = 2048


class VADConfig(BaseModel):
    min_silence_duration: float = 0.8
    min_speech_duration: float = 0.12
    activation_threshold: float = 0.45
    prefix_padding_duration: float = 0.4
    max_buffered_speech: float = 60.0


class LanguagesConfig(BaseModel):
    """Multilingual tenant configuration.

    When `available` has more than one entry, the agent starts in `default`
    and exposes a `select_language` tool so the LLM can switch. When only
    one entry, behaves identically to the legacy single-language path.
    """

    default: str = "uz"
    available: list[str] = Field(default_factory=lambda: ["uz"])
    ask_on_start: bool = False

    @property
    def is_multilingual(self) -> bool:
        return len(self.available) > 1


class BehaviorConfig(BaseModel):
    language: str = "uz"
    max_call_duration_seconds: int = 600
    preemptive_generation: bool = False
    allow_interruptions: bool = True


class AssemblyAIAgentConfig(BaseModel):
    """AssemblyAI Voice Agent API configuration for a tenant."""

    enabled: bool = False
    voice: str = "ivy"
    keyterms: list[str] = Field(default_factory=list)
    vad_threshold: float = 0.5
    min_silence_ms: int = 600
    max_silence_ms: int = 1500
    interrupt_response: bool = True


class GuardianConfig(BaseModel):
    """Tenant-authored instructions that make the LLM try to help before
    escalating to a human operator. Either field null ⇒ that guardian path
    is not rendered into the system prompt.
    """

    operator_instructions: str | None = None
    psychological_instructions: str | None = None


class TransferConfig(BaseModel):
    enabled: bool = False
    transfer_mode: str = "web"  # web | sip
    transfer_number: str = ""
    trigger_phrases: list[str] = Field(
        default_factory=lambda: ["operator", "jonli odam", "odam bilan gaplashaman"]
    )
    transfer_message: str = "Sizni operatorga ulayman. Iltimos, kuting."
    office_hours_start: int = 9
    office_hours_end: int = 18
    office_days: list[int] = Field(default_factory=lambda: [0, 1, 2, 3, 4, 5])  # Mon-Sat
    # IANA timezone used for the out-of-hours office-hours clock check
    # (tools/platform/escalate.py). Defaults to Asia/Karachi for Pakistan;
    # override per-tenant for other regions.
    timezone: str = "Asia/Karachi"
    # Per-language message used when escalate_to_human fires outside
    # office hours. Empty dict falls back to a built-in Uzbek default.
    out_of_hours_message: dict[str, str] = Field(default_factory=dict)
    guardian: GuardianConfig | None = None


class KBConfidenceThresholds(BaseModel):
    high: float = 0.75
    medium: float = 0.70


class KnowledgeBaseConfig(BaseModel):
    enabled: bool = False
    provider: str = "qdrant"
    collection: str = ""
    embedding_model: str = "gemini-embedding-001"
    confidence_thresholds: KBConfidenceThresholds = Field(default_factory=KBConfidenceThresholds)
    kb_paths: list[str] = []
    # NAV-214: Phrases that, when present in a user query, mean the question is
    # for an agency/topic outside this tenant's scope. The search_kb tool will
    # short-circuit and return a hard out-of-scope response instead of
    # surfacing a spuriously high-scoring embedding match.
    out_of_scope_keywords: list[str] = Field(default_factory=list)
    # NAV-214: Response template returned when an out_of_scope keyword matches.
    # Empty string falls back to personality.fallback.
    out_of_scope_response: str = ""
    # NAV-214: How many chunks to surface to the LLM in the search result.
    # Default 3 — gives the LLM enough material to cross-reference and
    # detect a wrong-match without overwhelming the prompt.
    surface_top_k: int = 3


class ResponsesConfig(BaseModel):
    silence_prompt: str = "Eshitiapsizmi? Iltimos, javob bering."
    silence_goodbye: str = "Aloqa uzildi shekilli. Xayr!"
    silence_prompt_by_language: dict[str, str] = Field(default_factory=dict)
    silence_goodbye_by_language: dict[str, str] = Field(default_factory=dict)
    fallback: str = "Kechirasiz, bu savolga javob bera olmayapman."
    goodbye: str = "Xayr! Yana savollaringiz bo'lsa, qo'ng'iroq qiling."
    tool_wait: str = "Iltimos, bir oz kutib turing."

    def silence_prompt_for(self, language: str) -> str:
        """Return silence prompt for language, falling back to default prompt."""
        return self.silence_prompt_by_language.get(language, self.silence_prompt)

    def silence_goodbye_for(self, language: str) -> str:
        """Return silence goodbye for language, falling back to default goodbye."""
        return self.silence_goodbye_by_language.get(language, self.silence_goodbye)


class SilenceConfig(BaseModel):
    prompt_timeout: float = 15.0
    goodbye_timeout: float = 30.0
    monitor_interval: float = 2.0


class ToolsConfig(BaseModel):
    platform: list[str] = Field(
        default_factory=lambda: [
            "search_knowledge_base",
            "request_escalation",
            # confirm_escalation is auto-paired by AgentFactory when
            # request_escalation is present.
            "get_current_time",
            "end_call",
        ]
    )
    tenant: list[str] = []


class FieldConfig(BaseModel):
    name: str
    # Monolingual fallback. Youth Agency configs set only this.
    prompt: str = ""
    # Per-language prompts keyed by ISO code ("uz", "ru"). Used for
    # multilingual tenants; falls back to `prompt` when the caller's
    # language is not present.
    prompts: dict[str, str] = Field(default_factory=dict)
    validation: str = ""  # e.g. "14-30"; empty = no range check
    required: bool = True

    def prompt_for(self, language: str) -> str:
        """Return the prompt for `language`, falling back to the monolingual `prompt`."""
        return self.prompts.get(language) or self.prompt


class GoalConfig(BaseModel):
    """A semantic probing goal the LLM must cover during appeal intake.

    `key` identifies the goal for logging/telemetry. `description[lang]`
    is a short hint the LLM uses to phrase a natural follow-up question
    (e.g. "Konkret muammo tafsiloti — qaysi xizmat va qanday xato").
    """

    key: str
    description: dict[str, str] = Field(default_factory=dict)
    required: bool = True


class SubAgentConfig(BaseModel):
    type: str  # currently "appeal"; new types wired in AgentFactory._SUB_AGENT_TYPES
    # Sub-agent's own system prompt (used once control is handed off).
    instructions: str = ""
    # Docstring seen by the MAIN agent's LLM on the handoff tool. Describe
    # WHEN to call it, not what the sub-agent does. Falls back to a short
    # generic description if empty.
    description: str = ""
    fields: list[FieldConfig] = []
    api_endpoint: str = ""
    transfer_message: str = ""
    # Per-language transfer messages. When populated, takes precedence over
    # `transfer_message` and is resolved at handoff time using the main
    # agent's runtime language (falls back to `transfer_message` if the
    # caller's language is not present in the dict).
    transfer_messages: dict[str, str] = Field(default_factory=dict)
    # Per-language user-facing messages used by the sub-agent. Expected
    # keys per language: {"start", "summary", "success"}. Empty dict falls
    # back to the sub-agent's built-in (Uzbek) defaults for backward compat.
    messages: dict[str, dict[str, str]] = Field(default_factory=dict)
    # Semantic probing goals the LLM must cover before calling
    # confirm_and_submit. Empty list = legacy behavior (LLM takes the
    # literal caller quote as content).
    goals: list[GoalConfig] = []
    # Per-language override for the summary-generation guidance appended
    # to the AppealAgent's system prompt. Empty dict = use the built-in
    # default guidance.
    summary_instructions: dict[str, str] = Field(default_factory=dict)


class FlowConfig(BaseModel):
    enabled: bool = False
    flow_dir: str = ""
    activation_threshold: float = 0.6
    max_steps_per_flow: int = 20
    allow_cross_flow: bool = True
    deterministic_bypass_llm: bool = True


class ObservabilityConfig(BaseModel):
    metrics_port: int = 9090
    health_port: int = 8082


class ExperimentBlock(BaseModel):
    """Marks a tenant YAML as a variant in an A/B experiment.

    All variants in the same experiment must share tenant identity fields
    (id, slug, name, phone_numbers); cohort validation in the loader enforces
    this. See docs/superpowers/specs/2026-04-25-ab-testing-design.md.
    """

    id: str = Field(..., min_length=1, max_length=64, pattern=r"^[a-z0-9_-]+$")
    variant: str = Field(..., min_length=1, max_length=32, pattern=r"^[a-z0-9_-]+$")
    weight: float = Field(..., gt=0)


class TenantConfig(BaseModel):
    """Complete validated tenant configuration."""

    tenant: TenantIdentity
    personality: PersonalityConfig = Field(default_factory=PersonalityConfig)
    voice: VoiceConfig = Field(default_factory=VoiceConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    vad: VADConfig = Field(default_factory=VADConfig)
    behavior: BehaviorConfig = Field(default_factory=BehaviorConfig)
    assemblyai: AssemblyAIAgentConfig = Field(default_factory=AssemblyAIAgentConfig)
    transfer: TransferConfig = Field(default_factory=TransferConfig)
    knowledge_base: KnowledgeBaseConfig = Field(default_factory=KnowledgeBaseConfig)
    responses: ResponsesConfig = Field(default_factory=ResponsesConfig)
    silence: SilenceConfig = Field(default_factory=SilenceConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    sub_agents: dict[str, SubAgentConfig] = {}
    flows: FlowConfig = Field(default_factory=FlowConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    languages: LanguagesConfig = Field(default_factory=LanguagesConfig)
    experiment: ExperimentBlock | None = None
    # Platform-level KB repopulate flags consumed by scripts/populate_qdrant.py
    # --auto on container startup. Keys are knowledge/<dir>/ directory names.
    # Lives on TenantConfig because _defaults.yaml deep-merges into each tenant
    # config; declaring it here documents the field instead of relying on
    # Pydantic's silent extra-field drop.
    repopulate_kb: dict[str, bool] = Field(default_factory=dict)

    @property
    def all_tool_names(self) -> list[str]:
        return self.tools.platform + self.tools.tenant
