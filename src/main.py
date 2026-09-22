"""
urdu-voice-agent: Multi-tenant voice AI agent building block.

Single binary, N tenants. Tenant behavior defined entirely by YAML config.
Uses LiveKit Agents SDK with native Agent class, function_tool, and multi-agent handoff.
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from livekit.agents import AgentServer, JobContext, JobProcess, RoomInputOptions, cli
from livekit.plugins import noise_cancellation, silero

from agents.factory import AgentFactory, set_factory
from api.platform_client import get_platform_client
from config.registry import TenantRegistry
from config.routing import resolve_tenant_for_call
from knowledge.kb_manager import KBManager
from knowledge.rag_engine import RAGEngine
from lifecycle.call_tracker import get_caller_history
from lifecycle.session_events import setup_session_events
from lifecycle.shutdown import register_shutdown_callbacks
from observability.health_server import start_health_server
from observability.langfuse_setup import record_latency_span, setup_langfuse
from observability.network_topology import install_network_observer, register_service_route
from observability.prometheus_metrics import init_agent_info
from observability.telephony_latency import TelephonyLatencyTracker
from pipeline.greeting_stt_override import should_use_greeting_stt_override
from pipeline.session_builder import SessionBuilder
from pipeline.voice_factory import VoiceFactory
from tools.registry import get_tool_registry
from utils.monitor_probe import is_monitor_probe
from utils.phone import extract_called_phone, extract_caller_phone

load_dotenv()

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("voice-agent")

install_network_observer()
register_service_route(
    "platform_api",
    os.getenv("PLATFORM_API_URL", "http://localhost:3000"),
    provider="platform_api",
)
register_service_route(
    "qdrant",
    os.getenv("QDRANT_URL", "http://localhost:6333"),
    provider="qdrant",
)

# --- Singletons ---
tenant_registry = TenantRegistry()
tool_registry = get_tool_registry()
platform_client = get_platform_client()
rag_engine = RAGEngine()
kb_manager = KBManager(rag_engine)
agent_factory = AgentFactory(tool_registry=tool_registry, kb_manager=kb_manager)
set_factory(agent_factory)

# --- AgentServer ---
# Pre-warm a pool of idle worker processes so inbound calls don't pay the
# ~9s Python + Silero VAD + STT/LLM/TTS cold-start tax. Each idle worker
# already has the prewarm() VAD model loaded and forkserver primed, so a
# new call gets answered in ~1s instead of ~9s. Tuned for prod (4 vCPU);
# initialize_process_timeout bumped from default 10s -> 30s to give the
# Silero VAD download/load room to finish on first boot.
# Override via NAVAI_NUM_IDLE_PROCESSES env var if needed.
_NUM_IDLE = int(os.getenv("NAVAI_NUM_IDLE_PROCESSES", "5"))
_INIT_TIMEOUT = float(os.getenv("NAVAI_INIT_PROCESS_TIMEOUT", "30"))
server = AgentServer(
    port=8088,
    num_idle_processes=_NUM_IDLE,
    initialize_process_timeout=_INIT_TIMEOUT,
)


def prewarm(proc: JobProcess):
    """Preload per-process resources so inbound calls skip cold-start work.

    Loads:
      1. Silero VAD model (~0.5-2s on first boot; cached on subsequent forks).
      2. Per-tenant Qdrant KB collections (~1.3s each on first boot).
         Without this, kb_manager.warmup() runs on the first call each worker
         handles, adding ~1.3s to pickup latency. Pre-warming here moves that
         cost off the critical path so calls land on a fully ready worker.
    """
    vad = silero.VAD.load(
        min_silence_duration=0.8,
        min_speech_duration=0.12,
        activation_threshold=0.45,
        prefix_padding_duration=0.4,
        max_buffered_speech=60.0,
    )
    proc.userdata["vad"] = vad
    logger.info("VAD loaded (balanced: 800ms silence, 0.45 threshold)")

    # Pre-warm KB collections for all loaded tenants in this worker process.
    # Skipped via NAVAI_PREWARM_KB=0 in case Qdrant is unreachable at boot.
    if os.getenv("NAVAI_PREWARM_KB", "1") != "0":
        try:
            import asyncio

            async def _warm_all_kbs() -> None:
                for slug in tenant_registry.list_tenants():
                    cfg = tenant_registry.resolve_by_slug(slug)
                    if not cfg or not cfg.knowledge_base.enabled:
                        continue
                    collection = cfg.knowledge_base.collection
                    if not collection:
                        continue
                    try:
                        await kb_manager.warmup(cfg.tenant.id, collection)
                    except Exception as exc:
                        logger.warning(
                            "KB prewarm failed for tenant=%s collection=%s: %s",
                            slug,
                            collection,
                            exc,
                        )

            asyncio.run(_warm_all_kbs())
            logger.info("KB prewarm complete (per-process Qdrant collections cached)")
        except Exception as exc:
            logger.warning("KB prewarm skipped (continuing): %s", exc)


server.setup_fnc = prewarm


@server.rtc_session(agent_name=os.getenv("AGENT_NAME", "voice-agent"))
async def entrypoint(ctx: JobContext):
    """
    Unified entrypoint for all tenants.

    Flow:
    1. Extract called phone number from SIP metadata
    2. Resolve tenant config (YAML files + Platform API)
    3. Fetch caller history (before agent construction, exclude_call_id=None)
    4. Determine effective language from caller history or tenant default
    5. Create STT/TTS/LLM for effective language
    6. Build Agent via AgentFactory (current_language for returning callers)
    7. Create call record with language metadata
    8. Create AgentSession, start, connect
    9. Register lifecycle callbacks
    10. Wire up telephony latency tracking (room + session events)
    """
    from livekit import rtc

    # 0. Start telephony latency tracking for real calls only.
    #    Monitor probes carry skip_call_tracking and are excluded from both
    #    telephony tracking and call-record persistence.
    #    NOTE(blueprint): the upstream NavAI "landing web-demo" call path was
    #    removed here — this blueprint is the core agent only (no web-demo
    #    surface). `call_source` stays for the call record's `source` field.
    skip_call_tracking = is_monitor_probe(ctx)
    call_source = None
    telephony_tracker = None if skip_call_tracking else TelephonyLatencyTracker(tenant="pending")
    if telephony_tracker:
        telephony_tracker.mark_entrypoint()

    # 1. Resolve tenant with strict routing policy.
    phone = extract_called_phone(ctx)
    config, experiment_meta, routing_block_reason = await resolve_tenant_for_call(
        called_phone=phone,
        tenant_registry=tenant_registry,
    )
    if not config:
        logger.error(
            "[tenant_routing_exit] reason=%s called_phone=%s action=terminate_before_session",
            routing_block_reason or "unresolved_tenant",
            phone or "none",
        )
        return

    logger.info(f"Tenant resolved: {config.tenant.slug} (phone: {phone})")
    if experiment_meta is not None:
        logger.info(
            "[experiment_assignment] tenant=%s experiment=%s variant=%s room=%s",
            config.tenant.slug,
            experiment_meta["experiment_id"],
            experiment_meta["variant"],
            ctx.room.name if ctx.room else "",
        )

    # 2. Fetch caller history FIRST so we know the caller's preferred language.
    caller_phone = extract_caller_phone(ctx) or "unknown"
    if "MagicMock" in str(caller_phone):
        caller_phone = "console"

    caller_history = None
    if caller_phone not in ("console", "unknown"):
        try:
            caller_history = await get_caller_history(
                phone=caller_phone,
                tenant_id=config.tenant.id,
                tenant_slug=config.tenant.slug,
                exclude_call_id=None,  # call record not yet created; nothing to exclude
            )
            if caller_history and caller_history.has_history:
                logger.info(
                    f"Returning caller: {caller_history.total_calls} previous calls, "
                    f"last topic: {caller_history.last_topic}, "
                    f"last language: {caller_history.last_language or '(none)'}"
                )
        except Exception as e:
            logger.warning(f"Caller history unavailable: {e}")

    # 3. Determine effective language: caller's remembered choice if available,
    #    else the tenant's default. When we have a remembered choice, the
    #    MULTILINGUAL POLICY prompt is skipped and the greeting is monolingual.
    preferred_language: str | None = None
    if caller_history and caller_history.last_language:
        if caller_history.last_language in config.languages.available:
            preferred_language = caller_history.last_language
            logger.info(f"Using remembered language: {preferred_language}")

    effective_lang = preferred_language or config.languages.default

    # 4. Create voice pipeline components for the effective language.
    use_greeting_stt_override = should_use_greeting_stt_override(config, preferred_language)
    logger.info(
        "[stt_selection] tenant=%s preferred_language=%s effective_lang=%s "
        "is_multilingual=%s greeting_stt_provider=%r use_greeting_override=%s",
        config.tenant.slug,
        preferred_language,
        effective_lang,
        config.languages.is_multilingual,
        config.voice.greeting_stt_provider,
        use_greeting_stt_override,
    )

    # Acquire the VAD before STT so the streaming navai_ws STT can reuse the
    # session's prewarmed, tenant-tuned Silero VAD for endpointing.
    vad = ctx.proc.userdata.get("vad") or VoiceFactory.load_vad(config)

    if use_greeting_stt_override:
        try:
            stt_inst = VoiceFactory.create_greeting_stt_for_language_choice(
                config,
                fallback_language=effective_lang,
            )
            logger.info(
                "[greeting_stt_override] active: tenant=%s provider=%s languages=%s",
                config.tenant.slug,
                config.voice.greeting_stt_provider,
                config.voice.greeting_stt_languages or ["ru", "uz"],
            )
        except Exception as e:
            logger.warning(
                "[greeting_stt_override] fallback: failed to init greeting STT (%s)",
                e,
            )
            stt_inst = VoiceFactory.create_stt_for_language(config, effective_lang, vad=vad)
            logger.info("[stt_selection] using fallback STT class=%s", type(stt_inst).__name__)
    else:
        stt_inst = VoiceFactory.create_stt_for_language(config, effective_lang, vad=vad)
        logger.info("[stt_selection] using default STT class=%s", type(stt_inst).__name__)
    tts_inst = VoiceFactory.create_tts_for_language(config, effective_lang)
    llm_inst = VoiceFactory.create_llm(config)
    if telephony_tracker and hasattr(stt_inst, "_on_stt_duration"):

        def record_stt_duration(duration_ms=None):
            if not isinstance(duration_ms, (int, float)):
                return
            telephony_tracker.record_stt_duration(duration_ms)
            record_latency_span(
                "stt_request",
                float(duration_ms),
                {
                    "tenant": config.tenant.slug,
                    "tenant_id": config.tenant.id,
                    "provider": config.voice.stt_provider_for(effective_lang),
                    "language": effective_lang,
                },
            )

        stt_inst._on_stt_duration = record_stt_duration

    # 5. Warm up tenant KB (non-blocking).
    if config.knowledge_base.enabled:
        try:
            await kb_manager.warmup(config.tenant.id, config.knowledge_base.collection)
        except Exception as e:
            logger.warning(f"KB warmup failed (continuing without KB): {e}")

    # 6. Build agent — passing current_language when we have a remembered choice.
    agent = agent_factory.create_agent(
        config,
        job_context=ctx,
        platform_client=platform_client,
        current_language=preferred_language,  # None for new callers; explicit for returning
        stt=stt_inst,
        tts=tts_inst,
        llm=llm_inst,
    )

    agent.caller_phone = caller_phone
    agent.agent_phone = phone or "unknown"
    agent._caller_history = caller_history  # keep for downstream context injection

    # 7. Create call record (console and monitor probes excluded).
    call_db_id = None
    if skip_call_tracking:
        logger.info(
            "Skipping call tracking for monitor probe: room=%s",
            ctx.room.name if ctx.room else "",
        )
    elif caller_phone != "console":
        try:
            from lifecycle.call_tracker import create_call

            room_name = ctx.room.name if ctx.room else agent.call_id
            call_db_id = await create_call(
                tenant_id=config.tenant.id,
                tenant_slug=config.tenant.slug,
                caller_phone=caller_phone,
                agent_phone=phone or "unknown",
                call_sid=room_name,
                metadata={
                    "room_name": room_name,
                    "agent_identity": agent.call_id,
                    "source": call_source or "voice_agent",
                    "agent_type": config.tenant.slug,
                    "language": agent.language,
                    **(experiment_meta or {}),
                },
            )
            agent.call_db_id = call_db_id
            agent.agent_identity = agent.call_id
        except Exception as e:
            logger.warning(f"Call tracking unavailable: {e}")

    langfuse_metadata = {
        "langfuse.session.id": ctx.room.name if ctx.room else agent.call_id,
        "service.name": "voice-agent",
        "tenant": config.tenant.slug,
        "tenant_id": config.tenant.id,
        "caller_phone": caller_phone,
        "agent_phone": phone or "unknown",
        "agent_call_id": agent.call_id,
        "call_db_id": call_db_id,
        "language": agent.language,
        "deployment.environment": os.getenv("AGENT_ENV", "unknown"),
        **(experiment_meta or {}),
    }
    langfuse_tracing = setup_langfuse(metadata=langfuse_metadata)
    if langfuse_tracing and hasattr(ctx, "add_shutdown_callback"):

        @ctx.add_shutdown_callback
        async def flush_langfuse_trace():
            langfuse_tracing.flush()

    # 8. Build session
    session = SessionBuilder.build(
        config,
        stt=stt_inst,
        tts=tts_inst,
        llm=llm_inst,
        vad=vad,
    )

    # 10. Wire telephony tracker to agent (session_events reads it from agent._telephony_tracker)
    if telephony_tracker:
        telephony_tracker.tenant = config.tenant.slug
    agent._telephony_tracker = telephony_tracker

    # 10. Setup events and shutdown
    setup_session_events(session, agent, config)
    register_shutdown_callbacks(ctx, agent, config)

    # 11. Start session and connect
    if telephony_tracker:
        telephony_tracker.mark_session_started()
    await session.start(
        agent=agent,
        room=ctx.room,
        # BVC (Background Voice Cancellation) rejects the agent's own echo
        # captured by the caller's mic AND background voices that aren't the
        # primary speaker. This is the practical fix for the "STT keeps
        # transcribing ایکس ایکس when nobody is speaking" feedback loop in
        # the LiveKit playground (no AEC in the browser by default).
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )
    if hasattr(agent, "_start_silence_monitor"):
        agent._start_silence_monitor()
    await ctx.connect()

    # 12. Wire up room-level events for SIP participant timing.
    if telephony_tracker:

        def on_participant_connected(participant: rtc.RemoteParticipant):
            if participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP:
                telephony_tracker.mark_participant_connected(participant.identity)

                attrs = participant.attributes or {}
                call_status = attrs.get("sip.callStatus")
                if call_status:
                    telephony_tracker.record_sip_status(call_status)

        def on_participant_attributes_changed(
            changed_attributes: dict, participant: rtc.Participant
        ):
            if participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP:
                status = changed_attributes.get("sip.callStatus")
                if status:
                    telephony_tracker.record_sip_status(status)

        def on_track_subscribed(
            track: rtc.Track,
            publication: rtc.RemoteTrackPublication,
            participant: rtc.RemoteParticipant,
        ):
            if track.kind == rtc.TrackKind.KIND_AUDIO:
                telephony_tracker.mark_track_subscribed()

        ctx.room.on("participant_connected", on_participant_connected)
        ctx.room.on("participant_attributes_changed", on_participant_attributes_changed)
        ctx.room.on("track_subscribed", on_track_subscribed)

        # Mark already-connected SIP participants (inbound calls arrive before entrypoint)
        for participant in ctx.room.remote_participants.values():
            if participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP:
                telephony_tracker.mark_participant_connected(participant.identity)
                attrs = participant.attributes or {}
                status = attrs.get("sip.callStatus")
                if status:
                    telephony_tracker.record_sip_status(status)

    logger.info(
        f"Agent ready: tenant={config.tenant.slug}, "
        f"caller={caller_phone}, call_id={agent.call_id}"
    )


def main():
    """Start the agent server."""
    start_health_server()
    init_agent_info()
    logger.info("Voice agent worker starting...")
    logger.info(f"Tenants loaded: {tenant_registry.list_tenants()}")
    cli.run_app(server)


if __name__ == "__main__":
    main()
