"""
TenantAgent: config-driven voice agent.

Per-turn pipeline:
  1. Sliding window ? compress old turns
  2. Flow check ? inject checklist if active
  3. Session state + conversation summary ? inject context
  4. LLM decides everything (KB search, flows, tools)
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import timedelta
from typing import TYPE_CHECKING, Any

from livekit.agents import Agent

from config.schema import TenantConfig
from context.session_state import SessionStateManager
from context.window import ContextWindow

if TYPE_CHECKING:
    from flows.engine import FlowEngine
    from resilience.error_recovery import ErrorRecoveryHandler
    from resilience.silence_monitor import SilenceMonitor

logger = logging.getLogger(__name__)

_UZBEK_WEEKDAYS = (
    "Dushanba",
    "Seshanba",
    "Chorshanba",
    "Payshanba",
    "Juma",
    "Shanba",
    "Yakshanba",
)


class TenantAgent(Agent):
    """
    Config-driven voice agent. LLM drives everything:
    KB search, flow activation, topic handling, coreference.
    """

    def __init__(
        self,
        *,
        config: TenantConfig,
        instructions: str,
        extra_tools: list | None = None,
        chat_ctx: Any | None = None,
        job_context: Any | None = None,
        kb_manager: Any | None = None,
        error_handler: ErrorRecoveryHandler | None = None,
        current_language: str | None = None,
        is_post_handoff: bool = False,
        stt: Any | None = None,
        tts: Any | None = None,
        llm: Any | None = None,
    ):
        # Forward stt/tts/llm to the base Agent so they're stored in its
        # `_stt`/`_tts`/`_llm` slots BEFORE any AgentActivity binding happens.
        # Critical for multilingual handoff: if we only set these as attributes
        # after construction, LiveKit's activity caches the session-level pipeline
        # and the new agent's language-specific STT/TTS never get used.
        super_kwargs: dict[str, Any] = {
            "instructions": instructions,
            "chat_ctx": chat_ctx,
            "tools": extra_tools,
        }
        if stt is not None:
            super_kwargs["stt"] = stt
        if tts is not None:
            super_kwargs["tts"] = tts
        if llm is not None:
            super_kwargs["llm"] = llm
        super().__init__(**super_kwargs)
        self.config = config
        self._job_context = job_context
        self._kb_manager = kb_manager
        self._error_handler = error_handler

        # Flow engine (set by AgentFactory if flows enabled)
        self._flow_engine: FlowEngine | None = None

        # Context management
        self._context_window = ContextWindow(window_size=5)
        self._session_state = SessionStateManager(
            tenant_name=config.tenant.name,
            language=config.behavior.language,
        )

        # Runtime-selected language (may differ from config.languages.default
        # after a select_language handoff).
        self.language: str = current_language or config.languages.default
        if self._error_handler is not None:
            self._error_handler.set_language(self.language)

        # True when this agent is the return target from a sub-agent handoff
        # (e.g. AppealAgent.confirm_and_submit returning to main). When True,
        # on_enter must NOT call generate_reply - the sub-agent has already
        # spoken a closing message, and triggering the LLM without a user
        # prompt leads it to hallucinate another sub-agent handoff.
        self._is_post_handoff: bool = is_post_handoff

        # Call state
        self.call_id: str = str(uuid.uuid4())
        self.call_db_id: str | None = None
        self.caller_phone: str = "unknown"
        self.agent_phone: str = "unknown"
        self.call_start_time: float = time.time()
        self.conversation_history: list[dict] = []
        self.transferred: bool = False
        self.handoff_completed: bool = False
        self._transfer_rejection_count: int = 0
        self._pending_transfer_metadata: dict = {}
        self._turns_since_last_sync: int = 0
        self._history_injected: bool = False

        # Platform API state
        self.transfer_reason: str = ""
        self.ai_summary: str = ""
        self.agent_identity: str = ""
        self._platform_client: Any | None = None
        self._caller_history: Any | None = None
        self._silence_monitor: SilenceMonitor | None = None
        # One-shot prompt spoken on next on_enter (used by language handoff).
        self._pending_post_handoff_prompt: str | None = None
        # Set by sub-agents (e.g. AppealAgent) after they create a record
        # so the final PATCH /calls/:id can link the call to the murojaat.
        self.murojaat_id: str | None = None

    async def on_enter(self) -> None:
        self.call_start_time = time.time()
        self._session_state._call_start = self.call_start_time
        # Start monitor on every agent activation (including handoffs).
        # `_start_silence_monitor` is idempotent, so main startup remains safe.
        self._start_silence_monitor()

        logger.info(
            f"Agent active: tenant={self.config.tenant.slug}, "
            f"call_id={self.call_id}, language={self.language}"
        )

        # Sub-agent handoff return path: the sub-agent's closing message
        # (e.g. "Murojaatingiz muvaffaqiyatli qabul qilindi...") is already
        # being spoken. Do NOT trigger generate_reply - without a fresh
        # user turn, the LLM sees the historical tool calls in chat_ctx
        # and hallucinates another transfer_to_<name> call, creating a
        # duplicate-submission loop (observed bug).
        if self._is_post_handoff:
            logger.info("on_enter: post-handoff from sub-agent - waiting for user")
            self._arm_silence_timer()
            return

        pending_prompt = self._pending_post_handoff_prompt
        if pending_prompt:
            if self._recent_assistant_said(pending_prompt):
                logger.info("on_enter: pending post-handoff help prompt already present in history")
            else:
                logger.info("on_enter: speaking pending post-handoff help prompt")
                await self._greet_monolingual(pending_prompt)
            self._pending_post_handoff_prompt = None
            self._arm_silence_timer()
            return
        # Post-handoff case: conversation_history was copied from the previous
        # agent by the select_language tool. The tool already returned a
        # confirmation message, so do NOT re-greet - that'd produce a double
        # greeting ("I'll continue in Uzbek" + "Hello, how can I help?").
        if self.conversation_history:
            logger.info(
                f"on_enter: skipping greeting "
                f"({len(self.conversation_history)} prior turns - post-handoff)"
            )
            self._arm_silence_timer()
            return

        # Fresh call. Pick the greeting shape:
        #   - Returning caller whose language is known and supported:
        #     short monolingual `greetings[lang]` (set in YAML).
        #   - First-time caller, multilingual tenant, AND pre-rendered audio
        #     files exist for every available language: play those files
        #     sequentially so each language's text gets its own TTS voice.
        #   - Otherwise: fall back to bilingual `greeting` via TTS.
        returning_with_known_lang = (
            self._caller_history is not None
            and getattr(self._caller_history, "last_language", "")
            in self.config.languages.available
        )

        if returning_with_known_lang:
            await self._greet_monolingual(self.config.personality.greeting_for(self.language))
            self._arm_silence_timer()
            return

        if self.config.languages.is_multilingual and self._have_all_greeting_audio():
            logger.info("on_enter: playing pre-rendered bilingual greeting audio")
            await self._play_prerendered_greeting()
            self._arm_silence_timer()
            return

        # Fallback: single bilingual TTS pass via whatever TTS is loaded.
        await self._greet_monolingual(self.config.personality.greeting)
        self._arm_silence_timer()

    async def _greet_monolingual(self, greeting: str) -> None:
        """Speak `greeting` via the agent's current TTS."""
        if not greeting or not self.session:
            return
        if self._error_handler:
            self._error_handler.set_agent_speaking(True)
        try:
            handle = self.session.say(
                greeting,
                add_to_chat_ctx=True,
                allow_interruptions=self.config.personality.greeting_allow_interruptions,
            )
            await handle.wait_for_playout()
        finally:
            if self._error_handler:
                self._error_handler.set_agent_speaking(False)

    def _recent_assistant_said(self, text: str, *, window: int = 6) -> bool:
        """Best-effort check whether recent assistant turns already include `text`."""
        needle = (text or "").strip().lower()
        if not needle:
            return False
        history = self.conversation_history[-window:] if self.conversation_history else []
        for turn in history:
            if not isinstance(turn, dict):
                continue
            if str(turn.get("role", "")).lower() != "assistant":
                continue
            content = str(turn.get("content", "")).strip().lower()
            if content == needle:
                return True
        return False

    def _have_all_greeting_audio(self) -> bool:
        """True iff greeting_audio[lang] exists on disk for every available lang."""
        from pathlib import Path

        p = self.config.personality
        for lang in self.config.languages.available:
            rel = p.greeting_audio_for(lang)
            if not rel or not Path(rel).is_file():
                return False
        return True

    async def _play_prerendered_greeting(self) -> None:
        """Stream each language's pre-rendered WAV into session.say() sequentially.

        `add_to_chat_ctx=True` lets LiveKit's own path record the spoken text
        in the chat history (the forwarded-text path does populate chat_ctx
        for pre-rendered audio when the text arg is a plain string - our
        earlier manual stamping attempt using chat_ctx.copy() triggered
        read-only errors and corrupted Gemini's tool-call metadata).
        """
        p = self.config.personality
        for lang in self.config.languages.available:
            audio_path = p.greeting_audio_for(lang)
            text = p.greeting_part_for(lang) or f"[{lang}] greeting"
            frames = await self._load_wav_as_frames(audio_path)
            if frames is None:
                logger.warning(
                    f"greeting audio missing for {lang} at {audio_path!r}; "
                    "aborting pre-rendered path"
                )
                return
            handle = self.session.say(
                text,
                audio=frames,
                add_to_chat_ctx=True,
                allow_interruptions=self.config.personality.greeting_allow_interruptions,
            )
            if self._error_handler:
                self._error_handler.set_agent_speaking(True)
            try:
                await handle.wait_for_playout()
            finally:
                if self._error_handler:
                    self._error_handler.set_agent_speaking(False)

    async def _on_silence_goodbye(self) -> None:
        """End the call after silence goodbye has been spoken."""
        await self.do_end_call()

    def _start_silence_monitor(self) -> None:
        """Start silence monitoring after session startup."""
        try:
            session = self.session
        except Exception:
            session = None
        if (
            getattr(self, "_silence_monitor", None) is not None
            or not getattr(self, "_error_handler", None)
            or not session
        ):
            return

        from resilience.silence_monitor import SilenceMonitor

        language = getattr(self, "language", self.config.languages.default)
        if self._error_handler is not None:
            self._error_handler.set_language(language)

        self._silence_monitor = SilenceMonitor(
            error_handler=self._error_handler,
            session=session,
            check_interval=self.config.silence.monitor_interval,
            goodbye_message=self.config.responses.silence_goodbye_for(language),
            on_goodbye=self._on_silence_goodbye,
        )
        self._silence_monitor.start()

    def _arm_silence_timer(self) -> None:
        """Start a fresh silence window now that the caller can respond."""
        if self._error_handler:
            # Guard against stale speaking=True across handoffs/events.
            self._error_handler.set_agent_speaking(False)
            self._error_handler.reset_silence_timer()

    def _stop_silence_monitor(self) -> None:
        """Stop silence monitoring if active."""
        idle_task = getattr(self, "_silence_idle_task", None)
        if idle_task is not None and not idle_task.done():
            idle_task.cancel()
        self._silence_idle_task = None
        if self._silence_monitor is None:
            return
        self._silence_monitor.stop()
        self._silence_monitor = None

    @staticmethod
    async def _load_wav_as_frames(path: str):
        """Async generator yielding ~40 ms rtc.AudioFrame chunks from a mono PCM WAV."""
        import wave

        from livekit import rtc

        try:
            wf = wave.open(path, "rb")
        except (FileNotFoundError, wave.Error) as e:
            logger.error(f"Cannot open greeting WAV {path!r}: {e}")
            return None

        try:
            sample_rate = wf.getframerate()
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            if sample_width != 2 or channels != 1:
                logger.error(
                    f"Unsupported WAV format at {path!r}: "
                    f"{channels}ch, {sample_width}bytes/sample (expected 1ch, 2bytes)"
                )
                return None
            frames_per_chunk = max(1, sample_rate // 25)  # ~40 ms
            data_per_chunk = frames_per_chunk * sample_width * channels
        finally:
            wf.close()

        async def _gen():
            f = wave.open(path, "rb")
            try:
                while True:
                    buf = f.readframes(frames_per_chunk)
                    if not buf:
                        return
                    # Pad the final chunk so sample count stays consistent.
                    if len(buf) < data_per_chunk:
                        buf = buf + b"\x00" * (data_per_chunk - len(buf))
                    yield rtc.AudioFrame(
                        data=buf,
                        sample_rate=sample_rate,
                        num_channels=channels,
                        samples_per_channel=frames_per_chunk,
                    )
            finally:
                f.close()

        return _gen()

    async def on_user_turn_completed(self, turn_ctx, new_message) -> None:
        user_text = ""
        if hasattr(new_message, "text_content"):
            user_text = (new_message.text_content or "").strip()
        if not user_text:
            return

        # Sliding window
        self._context_window.add_turn(role="user", content=user_text)

        # Session state
        self._session_state.record_turn()

        # Conversation history (for call record)
        self.conversation_history.append(
            {
                "role": "user",
                "content": user_text,
                "timestamp": time.time(),
            }
        )

        # Periodic transcript sync to backend (every 5 turns)
        self._turns_since_last_sync += 1
        if self._turns_since_last_sync >= 5 and self.call_db_id:
            self._turns_since_last_sync = 0
            import asyncio

            asyncio.create_task(self._sync_transcript())

        # Silence timer reset
        if self._error_handler:
            self._error_handler.reset_silence_timer()

        # Flow: inject checklist if active
        if self._flow_engine and self._flow_engine.is_active():
            for msg in self._flow_engine.get_context_for_turn():
                turn_ctx.add_message(role=msg["role"], content=msg["content"])
            return

        # Session state + conversation summary
        session_state = self._session_state.get_state_summary()
        conversation_summary = self._context_window.summary

        if session_state:
            turn_ctx.add_message(
                role="system",
                content=f"[Sessiya] {session_state}",
            )

        if conversation_summary:
            turn_ctx.add_message(
                role="system",
                content=f"[Oldingi suhbat] {conversation_summary}",
            )

        # Caller history - inject ONCE on the first turn as a question
        # prompt. The LLM asks whether the user wants to continue the
        # previous topic. If the user says something different, the LLM
        # answers their actual question and history is never mentioned
        # again (unless the user themselves brings it up).
        if (
            not self._history_injected
            and self._caller_history
            and self._caller_history.has_history
            and self._caller_history.last_topic
        ):
            self._history_injected = True
            h = self._caller_history
            turn_ctx.add_message(
                role="system",
                content=(
                    f"[Qayta qo'ng'iroq] Bu foydalanuvchi avval {h.total_calls} marta "
                    f"qo'ng'iroq qilgan. Oxirgi mavzu: {h.last_topic}."
                    f"{(' Xulosa: ' + h.last_summary + '.') if h.last_summary else ''}"
                    " Foydalanuvchidan avvalgi mavzuda davom etmoqchiligini SO'RANG. "
                    "Agar foydalanuvchi boshqa savol bersa yoki yo'q desa - "
                    "FAQAT uning hozirgi savoliga javob bering va avvalgi "
                    "mavzuni qayta TILGA OLMANG."
                ),
            )
        elif not self._history_injected:
            self._history_injected = True

        # Long call wrap-up
        if self._session_state.should_suggest_wrap_up():
            turn_ctx.add_message(
                role="system",
                content=(
                    "[Tizim] Qo'ng'iroq uzoq davom etmoqda. "
                    "Javobni qisqartiring va suhbatni yakunlashga harakat qiling."
                ),
            )

        if getattr(self, "murojaat_id", None):
            turn_ctx.add_message(
                role="system",
                content=(
                    "[Murojaat qabul qilindi] Bu qo'ng'iroqda murojaat backendga "
                    "muvaffaqiyatli yuborilgan. Uni qayta topshirmang, texnik "
                    "cheklov/xatolik bor demang. Agar foydalanuvchi davom etsa, "
                    "murojaat qabul qilinganini qisqa tasdiqlab, boshqa savoli "
                    "bor-yo'qligini so'rang."
                ),
            )

    # ---- Methods callable by platform tools ----

    async def do_end_call(self) -> str:
        logger.info(f"Call ended by agent: call_id={self.call_id}")
        self._stop_silence_monitor()
        if self.session:
            import asyncio

            # Wait for the goodbye TTS to actually finish playing before
            # disconnecting. The LLM often calls end_call in the same turn
            # as the goodbye message - a fixed 3s delay was cutting audio off.
            asyncio.create_task(self._disconnect_when_done(max_wait=15.0))
        return ""  # LLM already said goodbye, just disconnect

    def _get_tashkent_now(self):
        """Return current Asia/Tashkent datetime."""
        from datetime import datetime

        import pytz

        tz = pytz.timezone("Asia/Tashkent")
        return datetime.now(tz)

    def _format_shift_time(self, dt) -> str:
        """Render Uzbek-friendly day/time string for operator schedule."""
        day_name = _UZBEK_WEEKDAYS[dt.weekday()]
        return f"{day_name} soat {dt.strftime('%H:%M')}"

    def _next_shift_start(self, now, office_days: list[int]):
        """Return next operator shift start datetime, or None if undefined."""
        if not office_days:
            return None

        start_hour = self.config.transfer.office_hours_start
        for day_offset in range(0, 8):
            candidate = now + timedelta(days=day_offset)
            if candidate.weekday() not in office_days:
                continue

            shift_start = candidate.replace(
                hour=start_hour,
                minute=0,
                second=0,
                microsecond=0,
            )
            if shift_start <= now:
                continue
            return shift_start
        return None

    def _operator_schedule_context(self) -> dict[str, str | bool]:
        """Runtime schedule context used for operator transfer decisions."""
        now = self._get_tashkent_now()
        office_days = sorted(set(self.config.transfer.office_days))
        start_hour = self.config.transfer.office_hours_start
        end_hour = self.config.transfer.office_hours_end

        in_day = now.weekday() in office_days
        in_hours = start_hour <= now.hour < end_hour
        available_now = in_day and in_hours

        next_shift_dt = self._next_shift_start(now, office_days)
        next_shift = (
            self._format_shift_time(next_shift_dt) if next_shift_dt is not None else "aniqlanmagan"
        )

        return {
            "current_time": now.strftime("%H:%M"),
            "day_uzbek": _UZBEK_WEEKDAYS[now.weekday()],
            "operators_available_now": "yes" if available_now else "no",
            "next_shift_time": next_shift,
            "available_now": available_now,
        }

    async def do_forward_to_operator(self) -> str:
        """
        Transfer call to operator according to configured transfer_mode:
        - web: web_livekit_only flow (no SIP fallback)
        - sip: direct SIP transfer
        """
        if not self.config.transfer.enabled:
            return "Hozirda operator xizmati mavjud emas. Boshqa savol berishingiz mumkin."
        self._stop_silence_monitor()

        # Check if transfer number is "105" - bypass all time constraints
        # Obyekt bo'lishi mumkinligini hisobga olib, aniq string (str) ga o'giramiz
        transfer_number = str(self.config.transfer.transfer_number or "")
        is_priority_transfer = "105" in transfer_number

        schedule = self._operator_schedule_context()
        if not is_priority_transfer and not schedule["available_now"]:
            # TODO(urdu): this out-of-hours operator message is in Uzbek — add an
            # 'ur'/'en' localized variant (it is one of the framework prompt
            # fragments noted in the README "Urdu integration seams" section).
            return (
                "Hozir operatorlar mavjud emas. "
                f"Bugun {schedule['day_uzbek']}, soat {schedule['current_time']}. "
                f"Navbatdagi operator smenasi: {schedule['next_shift_time']}. "
                "Xohlasangiz callback uchun navbatdagi smenada qayta bog'lanishni taklif qilaman."
            )

        # Build summary from conversation
        self.transfer_reason = "User requested operator"
        self.ai_summary = self._build_conversation_summary()

        transfer_mode = str(getattr(self.config.transfer, "transfer_mode", "web") or "web").lower()
        if transfer_mode not in ("web", "sip"):
            logger.warning(f"Unknown transfer_mode={transfer_mode!r}, defaulting to web")
            transfer_mode = "web"

        # Web (LiveKit room) transfer: strict web-only mode, no SIP fallback.
        if transfer_mode == "web":
            if not self.call_db_id or not self._platform_client:
                logger.error(
                    "Web transfer requested but call_db_id/platform_client missing; "
                    "SIP fallback disabled by transfer_mode=web"
                )
                return (
                    "Operatorga ulashda texnik xatolik yuz berdi. "
                    "Iltimos, birozdan keyin qayta urinib ko'ring."
                )

            room_name = ""
            if self._job_context and hasattr(self._job_context, "room") and self._job_context.room:
                room_name = self._job_context.room.name

            try:
                # Step 1: PATCH call record with transfer metadata FIRST
                from lifecycle.call_tracker import update_call

                await update_call(
                    call_db_id=self.call_db_id,
                    tenant_id=self.config.tenant.id,
                    tenant_slug=self.config.tenant.slug,
                    status="in_progress",
                    transfer_reason=self.transfer_reason,
                    ai_summary=self.ai_summary,
                    metadata={
                        "room_name": room_name,
                        "agent_identity": self.agent_identity or self.call_id,
                        "handoff_mode": "web_livekit_only",
                    },
                )

                # Step 2: POST /calls/:id/operator
                success = await self._platform_client.request_operator(
                    call_db_id=self.call_db_id,
                    tenant_id=self.config.tenant.id,
                    tenant_slug=self.config.tenant.slug,
                    transfer_reason=self.transfer_reason,
                    ai_summary=self.ai_summary,
                    room_name=room_name,
                    agent_identity=self.agent_identity or self.call_id,
                )
                if success:
                    self.transferred = True
                    self._session_state.record_transfer_offered()
                    import asyncio

                    asyncio.create_task(self._poll_operator_status())
                    return self.config.transfer.transfer_message
                logger.warning(
                    "Dashboard handoff request failed; transfer_mode=web so SIP fallback is disabled"
                )
            except Exception as e:
                logger.error(f"Dashboard handoff error: {e}")
            return (
                "Operatorga ulashda texnik xatolik yuz berdi. "
                "Iltimos, birozdan keyin qayta urinib ko'ring."
            )

        # SIP mode: only direct SIP transfer.
        transfer_number = self.config.transfer.transfer_number
        if transfer_number and self._job_context:
            try:
                import asyncio

                asyncio.create_task(self._sip_transfer(transfer_number))
                self.transferred = True
                self._session_state.record_transfer_offered()
                return self.config.transfer.transfer_message
            except Exception as e:
                logger.error(f"SIP transfer failed: {e}")
                return "Operatorga ulanishda xatolik yuz berdi. Iltimos, qayta qo'ng'iroq qiling."

        logger.error("SIP transfer requested but transfer_number or job_context is missing")
        return "Operatorga ulanishda xatolik yuz berdi. Iltimos, qayta qo'ng'iroq qiling."

    async def _poll_operator_status(self) -> None:
        """
        Poll operator handoff status.

        Status lifecycle (per backend spec):
            in_queue ? ringing ? transferred ? in_progress (operator joined)

        Once we see `transferred`, wait until the operator participant actually
        appears in the LiveKit room before disconnecting the AI. This avoids
        cutting off the caller during the brief gap between operator acceptance
        and audio path establishment.
        """
        import asyncio

        if not self._platform_client or not self.call_db_id:
            return

        max_attempts = 60  # 60 * 2s = 120s max wait
        last_status = ""
        for attempt in range(max_attempts):
            await asyncio.sleep(2)
            try:
                status = await self._platform_client.get_operator_status(
                    call_db_id=self.call_db_id,
                    tenant_id=self.config.tenant.id,
                    tenant_slug=self.config.tenant.slug,
                )
                if status.status != last_status:
                    logger.info(f"Operator status: {status.status} (poll {attempt + 1})")
                    last_status = status.status

                if status.status == "transferred":
                    logger.info("Operator accepted, waiting for room join...")
                    joined = await self._wait_for_operator_in_room(timeout=15.0)
                    if joined:
                        logger.info("Operator joined room - removing AI")
                        self.handoff_completed = True
                        self.transferred = True
                        self._session_state.record_transfer_offered()
                        await self._remove_ai_from_room()
                        return
                    else:
                        logger.warning(
                            "Operator did not appear in room within 15s - keeping AI in room"
                        )
                        continue
                elif status.status in ("unknown", "failed"):
                    logger.warning(f"Operator handoff ended: {status.status}")
                    return
            except Exception as e:
                logger.error(f"Operator poll error: {e}")

        logger.warning("Operator handoff timed out after 120s")

    async def _wait_for_operator_in_room(self, timeout: float = 15.0) -> bool:
        """
        Wait until a stable operator participant joins the LiveKit room.
        Returns True if seen stably, False on timeout.
        """
        import asyncio

        if not self._job_context or not self._job_context.room:
            return False

        room = self._job_context.room
        deadline = time.time() + timeout
        stable_observations = 0
        required_stable_observations = 3  # ~1.5s with 0.5s polling
        while time.time() < deadline:
            try:
                # remote_participants is a dict of identity -> RemoteParticipant
                operator_present = False
                for p in room.remote_participants.values():
                    identity = getattr(p, "identity", "") or ""
                    # Require explicit operator identity to avoid transient non-SIP participants.
                    if identity.startswith("operator:"):
                        operator_present = True
                        break

                if operator_present:
                    stable_observations += 1
                    if stable_observations >= required_stable_observations:
                        return True
                else:
                    stable_observations = 0
            except Exception as e:
                logger.debug(f"room poll error: {e}")
            await asyncio.sleep(0.5)
        return False

    async def _remove_ai_from_room(self) -> None:
        """Remove AI participant from room after operator joins."""
        self._stop_silence_monitor()
        try:
            if self._job_context and self._job_context.room:
                await self._job_context.room.disconnect()
        except Exception as e:
            logger.debug(f"Disconnect error (expected): {e}")

    def _build_conversation_summary(self) -> str:
        """Build a brief summary from conversation history for handoff context."""
        if not self.conversation_history:
            return ""
        user_msgs = [e["content"] for e in self.conversation_history if e["role"] == "user"]
        if not user_msgs:
            return ""
        # Take last 3 user messages as summary context
        recent = user_msgs[-3:]
        return "Foydalanuvchi so'ragan mavzular: " + "; ".join(recent)

    async def _sync_transcript(self) -> None:
        """Periodically sync transcript to backend so operator dashboard has live data."""
        try:
            from datetime import datetime, timezone

            from lifecycle.call_tracker import update_call

            transcript = []
            for entry in self.conversation_history:
                role = entry.get("role", "unknown")
                if role == "assistant":
                    role = "agent"
                # Convert epoch float to ISO 8601 with Z suffix (backend requirement)
                ts_epoch = entry.get("timestamp")
                if isinstance(ts_epoch, (int, float)):
                    dt = datetime.fromtimestamp(ts_epoch, tz=timezone.utc)
                    ts_iso = dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"
                else:
                    ts_iso = ""
                transcript.append(
                    {
                        "role": role,
                        "text": entry.get("content", ""),
                        "timestamp": ts_iso,
                    }
                )
            duration = int(time.time() - self.call_start_time) if self.call_start_time else 0
            await update_call(
                call_db_id=self.call_db_id,
                tenant_id=self.config.tenant.id,
                tenant_slug=self.config.tenant.slug,
                status="in_progress",
                duration_seconds=duration,
                transcript=transcript,
                metadata={"language": self.language},
            )
            logger.debug(f"Transcript synced: {len(transcript)} entries")
        except Exception as e:
            logger.warning(f"Transcript sync failed: {e}")

    async def _sip_transfer(self, phone: str) -> None:
        """Fallback: direct SIP transfer via LiveKit API."""
        try:
            import os

            from livekit import api as lk_api
            from livekit.protocol.models import ParticipantInfo

            room_name = self._job_context.room.name if self._job_context else ""

            async with lk_api.LiveKitAPI(
                url=os.getenv("LIVEKIT_URL", ""),
                api_key=os.getenv("LIVEKIT_API_KEY", ""),
                api_secret=os.getenv("LIVEKIT_API_SECRET", ""),
            ) as lk:
                participants = await lk.room.list_participants(
                    lk_api.ListParticipantsRequest(room=room_name)
                )

                # Accept either an E.164 phone (will be wrapped as tel:)
                # or a full URI like "sip:105@host" / "tel:+998..." passed as-is.
                if phone.startswith("sip:") or phone.startswith("tel:"):
                    transfer_to = phone
                else:
                    transfer_to = f"tel:{phone}"

                for p in participants.participants:
                    if p.kind == ParticipantInfo.SIP:
                        await lk.sip.transfer_sip_participant(
                            lk_api.TransferSIPParticipantRequest(
                                participant_identity=p.identity,
                                room_name=room_name,
                                transfer_to=transfer_to,
                            )
                        )
                        logger.info(f"SIP transfer initiated to {transfer_to}")
                        self.transferred = True
                        self._session_state.record_transfer_offered()
                        await self._record_sip_transfer_initiated(
                            transfer_to=transfer_to,
                            room_name=room_name,
                        )
                        return

                logger.warning(
                    f"No SIP participant found for transfer "
                    f"(participants: {[(p.identity, p.kind) for p in participants.participants]})"
                )
        except Exception as e:
            logger.error(f"SIP transfer error: {e}")

    async def _record_sip_transfer_initiated(self, *, transfer_to: str, room_name: str) -> None:
        """Persist SIP handoff state as soon as LiveKit accepts the transfer."""
        if not self.call_db_id:
            return
        try:
            from lifecycle.call_tracker import update_call

            duration = int(time.time() - self.call_start_time) if self.call_start_time else 0
            await update_call(
                call_db_id=self.call_db_id,
                tenant_id=self.config.tenant.id,
                tenant_slug=self.config.tenant.slug,
                status="transferred",
                duration_seconds=duration,
                transfer_reason=self.transfer_reason,
                ai_summary=self.ai_summary,
                metadata={
                    "room_name": room_name,
                    "agent_identity": self.agent_identity or self.call_id,
                    "transferred": True,
                    "handoff_completed": False,
                    "handoff_mode": "sip",
                    "sip_transfer_to": transfer_to,
                    "transfer": dict(getattr(self, "_pending_transfer_metadata", {}) or {}),
                },
            )
        except Exception as e:
            logger.warning(f"Failed to persist SIP transfer state: {e}")

    async def _delayed_disconnect(self, delay: float) -> None:
        import asyncio

        await asyncio.sleep(delay)
        await self._remove_ai_from_room()

    async def _disconnect_when_done(self, max_wait: float = 15.0) -> None:
        """Wait until the agent finishes speaking, then end the call entirely.

        Uses `AgentSession.wait_for_inactive` so the goodbye TTS plays out
        completely. After that, deletes the LiveKit room which hangs up the
        SIP leg too - leaving the room intact would keep the caller on a
        silent line.
        """
        import asyncio

        try:
            if self.session and hasattr(self.session, "wait_for_inactive"):
                await asyncio.wait_for(self.session.wait_for_inactive(), timeout=max_wait)
            else:
                await asyncio.sleep(min(max_wait, 5.0))
        except asyncio.TimeoutError:
            logger.warning(f"wait_for_inactive timed out after {max_wait}s; ending anyway")
        except Exception as e:
            logger.debug(f"wait_for_inactive error: {e}")
        # small grace period so the last RTP packet flushes
        await asyncio.sleep(0.3)
        await self._end_call_completely()

    async def _end_call_completely(self) -> None:
        """Hang up the entire call by deleting the LiveKit room.

        This kicks both the AI and the SIP participant - used when the
        agent itself has decided the conversation is over (`do_end_call`).
        For operator handoff, use `_remove_ai_from_room` instead so the
        room stays alive for the operator.
        """
        self._stop_silence_monitor()
        try:
            import os

            from livekit import api as lk_api

            room_name = ""
            if self._job_context and self._job_context.room:
                room_name = self._job_context.room.name

            if not room_name:
                # Fall back to local disconnect if we don't know the room
                await self._remove_ai_from_room()
                return

            async with lk_api.LiveKitAPI(
                url=os.getenv("LIVEKIT_URL", ""),
                api_key=os.getenv("LIVEKIT_API_KEY", ""),
                api_secret=os.getenv("LIVEKIT_API_SECRET", ""),
            ) as lk:
                await lk.room.delete_room(lk_api.DeleteRoomRequest(room=room_name))
                logger.info(f"Deleted LiveKit room '{room_name}' to end call")
        except Exception as e:
            logger.error(f"Failed to delete room: {e}")
            # Best-effort: at least drop our own connection
            await self._remove_ai_from_room()
