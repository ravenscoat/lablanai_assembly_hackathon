"""select_language platform tool.

Enabled only when `config.languages.available` has more than one entry.
When the LLM calls it with a valid language code, we:

1. Play the pre-rendered confirmation audio for the TARGET language
   (so the user hears the right voice immediately — LiveKit's activity
   swap is async and the session's TTS is still the old language until
   the swap completes, which is usually after the current speech finishes;
   playing our own audio bypasses that).
2. Build a fresh TenantAgent configured for the chosen language
   (language-specific STT/TTS, LLM instructions, etc.).
3. Return `(new_agent, tool_message)` where `tool_message` is a deterministic
   first help prompt in the selected language (and includes confirmation text
   only when pre-rendered audio isn't available).

The in-YAML `personality.language_switch_text` / `language_switch_audio`
dicts provide the confirmation text + pre-rendered WAV paths. Without
pre-rendered audio the tool falls back to returning confirmation + help
as the tool result.
"""

from __future__ import annotations

import logging
import wave
from pathlib import Path
from typing import Any

from livekit import rtc
from livekit.agents import RunContext, function_tool

from config.schema import TenantConfig

logger = logging.getLogger(__name__)


# Fallback confirmations used only when the tenant YAML has no
# language_switch_text for the target language. Kept as a last-resort
# default so the tool remains useful on tenants that haven't yet
# configured the new fields.
_CONFIRMATION_FALLBACK: dict[str, str] = {
    "uz": "Yaxshi, suhbatni o'zbek tilida davom ettiraman.",
    "ru": "Хорошо, продолжу общение на русском языке.",
}

_REJECTION: dict[str, str] = {
    "uz": "Kechirasiz, {lang} tili qo'llab-quvvatlanmaydi.",
    "ru": "Извините, язык {lang} не поддерживается.",
}


_HELP_PROMPT: dict[str, str] = {
    "uz": "Sizga qanday yordam bera olaman?",
    "ru": "Чем я могу вам помочь?",
}


async def _wav_frames(path: str):
    """Async generator yielding ~40ms rtc.AudioFrame chunks from a mono 16-bit WAV."""
    try:
        probe = wave.open(path, "rb")
    except (FileNotFoundError, wave.Error) as e:
        logger.warning(f"select_language: cannot open confirmation WAV {path!r}: {e}")
        return
    try:
        sample_rate = probe.getframerate()
        channels = probe.getnchannels()
        sample_width = probe.getsampwidth()
    finally:
        probe.close()

    if sample_width != 2 or channels != 1:
        logger.warning(
            f"select_language: unsupported WAV format at {path!r} "
            f"({channels}ch, {sample_width}bytes/sample) — skipping pre-rendered play"
        )
        return

    frames_per_chunk = max(1, sample_rate // 25)  # ~40 ms
    data_per_chunk = frames_per_chunk * sample_width * channels

    f = wave.open(path, "rb")
    try:
        while True:
            buf = f.readframes(frames_per_chunk)
            if not buf:
                return
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


def create_select_language_tool(config: TenantConfig, **kwargs) -> Any:
    """Factory: returns the select_language tool, or None for single-language tenants."""
    if not config.languages.is_multilingual:
        return None

    allowed = set(config.languages.available)
    fallback_lang = config.languages.default

    @function_tool(name="select_language")
    async def select_language(context: RunContext, lang: str) -> Any:
        """Switch the conversation to the caller's preferred language.

        **Call this tool ONLY when:**
        - The caller is answering the initial "which language?" question
          with a language name (e.g. "узбекский" / "русский" / "o'zbek" / "uz").
        - The caller EXPLICITLY asks to switch language mid-call
          (e.g. "давай по-русски", "перейди на узбекский", "ruschaga o'ting").

        **Do NOT call this tool when:**
        - The caller just says hello, asks a question, or speaks normally.
        - The caller uses a single foreign word — that alone is not a request
          to switch. Keep the current language unless they explicitly ask.
        - The language is already the one the caller is using.

        Args:
            lang: ISO 639-1 code, one of the tenant's configured languages
                  (e.g. 'uz' or 'ru').
        """
        normalized = (lang or "").strip().lower()
        if normalized not in allowed:
            msg = _REJECTION.get(fallback_lang, _REJECTION["uz"]).format(lang=lang)
            logger.info(f"select_language rejected: lang={lang!r}")
            return msg

        agent = context.session.current_agent
        current_lang = getattr(agent, "language", fallback_lang)

        # Resolve the confirmation text: YAML-configured first, fallback to builtin.
        confirmation = config.personality.language_switch_text_for(
            normalized
        ) or _CONFIRMATION_FALLBACK.get(normalized, "")

        if normalized == current_lang:
            logger.info(
                f"select_language noop: already in {current_lang!r} — "
                "LLM called the tool redundantly"
            )
            return confirmation

        # Play the pre-rendered confirmation in the TARGET voice. This plays
        # via the OLD activity's audio output (audio frames bypass TTS), but
        # it's still spoken in the correct voice because the WAV was rendered
        # with yulduz/yulduz_ru offline.
        #
        # Known limitation: LiveKit's agent swap (triggered by returning
        # (new_agent, ...) below) is async — the new activity only becomes
        # active after the OLD activity's speech queue drains. The LLM's
        # post-tool reflection runs DURING that window and its TTS goes
        # through the OLD activity. We've tried in-place agent._tts swaps
        # (breaks Gemini tool metadata) and "stay silent" prompts (Gemini
        # ignores them). The remaining mitigations are:
        #   1. Keep language_switch_text short (one sentence) — minimizes the
        #      amount of yulduz_ru-reading-uzbek audio the caller endures.
        #   2. Gemini's post-tool response is usually brief (~2s) and then
        #      the new activity takes over for subsequent turns.
        audio_path = config.personality.language_switch_audio_for(normalized)
        pre_rendered_played = False
        session = getattr(context, "session", None)
        if audio_path and Path(audio_path).is_file() and session is not None:
            try:
                handle = session.say(
                    confirmation,
                    audio=_wav_frames(audio_path),
                    add_to_chat_ctx=True,
                )
                await handle.wait_for_playout()
                pre_rendered_played = True
                logger.info(
                    f"select_language: played pre-rendered confirmation for "
                    f"{normalized!r} from {audio_path}"
                )
            except Exception as e:
                logger.warning(
                    f"select_language: pre-rendered audio play failed ({e!r}) — "
                    "falling back to tool-result confirmation"
                )

        # Step 2: rebuild the agent for the chosen language.
        from agents.factory import get_factory

        factory = get_factory()
        new_agent = factory.create_agent(
            config,
            chat_ctx=getattr(agent, "chat_ctx", None),
            job_context=getattr(agent, "_job_context", None),
            platform_client=getattr(agent, "_platform_client", None),
            current_language=normalized,
        )
        # Guarantee the post-switch localized "how can I help?" prompt from
        # the new agent's own TTS path when we already played a pre-rendered
        # confirmation (to avoid old-agent post-tool reflection voice/race).
        # When pre-rendered confirmation is NOT played, we return confirmation
        # + help directly as tool_message and must not schedule the pending
        # prompt, otherwise the caller hears the same help line twice.
        if pre_rendered_played:
            setattr(new_agent, "_pending_post_handoff_prompt", _HELP_PROMPT.get(normalized, ""))

        # Carry over call state that must survive the handoff.
        for attr in (
            "call_id",
            "call_db_id",
            "caller_phone",
            "agent_phone",
            "call_start_time",
            "conversation_history",
            "agent_identity",
            "_caller_history",
            "_sf_instance",
            "_sf_session_id",
            "_telephony_tracker",
        ):
            if hasattr(agent, attr):
                setattr(new_agent, attr, getattr(agent, attr))

        logger.info(f"select_language handoff: {current_lang} -> {normalized}")

        # Stop the old agent's silence loop before handing off.
        if hasattr(agent, "_stop_silence_monitor"):
            try:
                agent._stop_silence_monitor()
            except Exception as e:
                logger.debug(f"select_language: failed to stop old silence monitor: {e}")

        # Step 3: return deterministic post-switch speech.
        # Always ask "how can I help?" once in the target language.
        # If pre-rendered confirmation was played, return only the help prompt.
        # Otherwise include confirmation + help in a single fallback response.
        help_prompt = _HELP_PROMPT.get(normalized) or _HELP_PROMPT[fallback_lang]
        if pre_rendered_played:
            # Avoid duplicate post-switch helper prompt:
            # new_agent.on_enter will speak `_pending_post_handoff_prompt`.
            tool_message = ""
        else:
            tool_message = " ".join(part for part in (confirmation, help_prompt) if part).strip()
        return new_agent, tool_message

    return select_language
