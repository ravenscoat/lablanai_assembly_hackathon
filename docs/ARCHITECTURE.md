# NavAI Multi-Tenant Voice-Agent Platform — Master Architecture

> Synthesized from the nine per-subsystem research docs (`01`–`09`) under
> `/Users/mudassar/Desktop/nov-ai/blueprint-research/`. This is the single technical map of the
> production system as read on 2026-06-17. It exists to support replicating the architecture as a
> reusable building block for an Urdu voice-agent venture; the concrete plan for that lives in
> `BLUEPRINT-PLAN.md`.

---

## 1. System Overview

NavAI is a **single Python voice-agent binary serving N tenants**, fronted by a SIP/WebRTC bridge
(Asterisk + LiveKit Cloud) and backed by a TypeScript control plane (Express API + React dashboard +
MongoDB/Redis) and a per-tenant vector store (Qdrant). The LLM is the conversation controller; tenant
behaviour is defined entirely by YAML + a Qdrant knowledge-base collection. Routing to a tenant is by
the **called (destination) phone number**, never by agent name or room name.

Three independent control planes meet only at clean seams:

| Plane | Owns | Tech |
|-------|------|------|
| **PSTN / SIP** | Carrier interconnect, number registration, NAT, codecs | Asterisk 20.6 PJSIP |
| **Media / room** | Per-call WebRTC room, SIP↔WebRTC bridge, agent dispatch | LiveKit Cloud (inbound trunk + dispatch rule) |
| **Application** | Tenant resolution, STT/LLM/TTS, transfer, call records, KB, reconciliation, dashboards | navai-agent (Python) + Platform API (Node) + Mongo/Redis/Qdrant |

The agent never speaks SIP; Asterisk never speaks to the LLM; LiveKit is the bridge.

### 1.1 Component & data-flow diagram (ASCII)

```
                                    ┌──────────────────────────────────────────────────────┐
   PSTN caller                     │                EXTERNAL MODEL PROVIDERS                │
   dials a DID                     │  STT: navai_ws (WS) / yandex / custom-GPU / deepgram   │
        │                          │  TTS: navai_ws (WS) / yandex / custom-GPU / cartesia   │
        ▼                          │  LLM: Gemini (Vertex ADC, no key) / openai / vLLM      │
 ┌──────────────┐                  │  Embeddings: gemini-embedding-001 (Vertex, 768-dim)    │
 │ Carrier /    │                  └───────────────▲───────────────────────▲───────────────┘
 │ Axona PBX    │                                  │ STT/TTS/LLM            │ embeddings
 └──────┬───────┘                                  │                        │
        │ SIP (registered trunk)                   │                        │
        ▼                                   ┌───────┴────────┐        ┌──────┴───────┐
 ┌──────────────────────┐  TLS 5061  ┌────► │  NavAI AGENT   │        │   Qdrant     │
 │  ASTERISK PJSIP       │───────────┤      │  (LiveKit      │◄──────►│ <tenant>_faq │
 │  [inc] exten          │           │      │   worker :8088)│  RAG   │  (one per    │
 │  Macro→PJSIP/<DID>@LK │           │      │  health :8082  │        │   tenant)    │
 │  [outbound-calls]     │◄──────────┤      └───┬────────┬───┘        └──────────────┘
 │  trunk-out → <OUTBOUND_CARRIER_HOST>  │ REFER/out │          │ outbound (PlatformClient, x-api-key)
 │  trunk-human (REFER)  │           │          ▼
 └──────────────────────┘           │   ┌──────────────────────────────────────────────┐
        ▲                           │   │           PLATFORM API (Express :3000)        │
        │ PBX webhook + CRM API     │   │  /internal/* (agent ingest)  /auth /calls /kb │
        │ (recordings, history)     │   │  /operator /murojatlar /webhooks/pbx /admin   │
        │                           │   │  metrics :9091                                │
 ┌──────┴────────────┐             │   └───┬───────────────┬──────────────┬───────────┘
 │  LiveKit Cloud     │◄────────────┘       │ Mongoose      │ ioredis      │ REST
 │  inbound SIP trunk │  agent dispatched   ▼               ▼              ▼
 │  dispatch rule     │  by AGENT_NAME  ┌─────────┐   ┌──────────┐   (Qdrant KB sync,
 │  room=<caller>_xx  │                 │ MongoDB │   │  Redis   │    Gemini summaries)
 └──────┬─────────────┘                 │ calls,  │   │ queue,   │
        │ operator joins room (WebRTC)  │ tenants,│   │ blacklist│
        │ for web-mode transfer         │ users…  │   │ pub/sub  │
        ▼                               └─────────┘   └─────┬────┘
 ┌────────────────────┐                       ▲             │ SSE/WS realtime
 │ DASHBOARD (React/   │                       │ JWT cookie  ▼
 │ nginx :80)          │───────────────────────┴──── operator console / admin / KB UI
 └────────────────────┘

 OBSERVABILITY: agent /metrics:8082 + Platform /metrics:9091 → Prometheus :9090 → Grafana :3200;
                Loki/promtail logs; Langfuse LLM traces; Alertmanager → Telegram.
```

### 1.2 Two-port rule (recurring source of confusion)

- **Agent `:8088`** — LiveKit `AgentServer` worker. Outbound-only to LiveKit Cloud; **not published**.
- **Agent `:8082`** — stdlib health/observability HTTP server: `/health`, `/metrics`,
  `/network/topology`, `/network/telephony`. The only published agent port.
- **API `:3000`** — Platform HTTP; **API `:9091`** — Prometheus metrics.

---

## 2. Subsystems

### 2.1 Agent core & media pipeline (doc 01)

Entry point `src/main.py` constructs a LiveKit `AgentServer(port=8088, num_idle_processes=5)` and
registers ONE per-call entrypoint via `@server.rtc_session(agent_name=AGENT_NAME)` (default
`navai-agent`). A prewarm hook loads Silero VAD (Uzbek-tuned) and warms per-tenant Qdrant collections.
Per call: classify → extract called phone → resolve tenant → caller history → effective language →
**build STT/TTS/LLM** → warm KB → build agent → create call record → build `AgentSession` → start →
connect.

The realtime pipeline is LiveKit's `AgentSession` (assembled in `pipeline/session_builder.py`):
**mic → VAD endpointing → STT → LLM (with tools) → TTS → room playback**. STT/TTS/LLM are forwarded
into `Agent.__init__` so a mid-call language handoff rebinds the right components.

**The single provider swap file is `src/pipeline/voice_factory.py`** — three string-keyed dispatch
methods (`create_stt_for_language`, `create_tts_for_language`, `create_llm`) plus a `_LANG_MAP`
locale table. Providers subclass `livekit.agents.stt.STT` / `tts.TTS` and live in
`src/pipeline/providers/`. Current default stack: `navai_ws` STT (WS streaming) → `gemini` LLM
(Vertex ADC) → `navai_ws` TTS, Silero VAD. (CLAUDE.md's "Yandex" default is stale.) This file is the
primary Urdu integration seam (see `BLUEPRINT-PLAN.md`).

### 2.2 Multi-tenancy & configuration (doc 02)

A tenant = one `configs/tenants/<slug>.yaml`, validated by the Pydantic `TenantConfig`
(`src/config/schema.py`). The loader **deep-merges** `_defaults.yaml ← tenant.yaml`, then applies
`environments.yaml` per-`CONFIG_ENV` (only `tenant.id` + `phone` differ dev vs prod), then validates.
Routing is by called phone (`extract_called_phone` reads dispatch/room metadata → `TenantRegistry`).
`MULTI_TENANT_STRICT_ROUTING=true` (prod default) fails closed on unknown numbers.
A/B experiments: variant YAMLs sharing `experiment.id` are weighted-random-picked per call.

Prompts are authored in YAML `personality.system_prompt` (+ guardian + sub-agent blocks) and
assembled by `AgentFactory._build_instructions`, which injects **hard-coded Uzbek/Russian** policy
fragments (transfer, flow, KB, CSAT, response-format). Those fragments + STT/TTS Urdu coverage + the
per-tenant API-key map in `config/api_keys.py` are the only non-generic blockers for an Urdu tenant.

### 2.3 RAG / knowledge base (doc 03)

One Qdrant **COSINE** collection per tenant (`<kb_dir>_faq`), 768-dim vectors from
`gemini-embedding-001` (Vertex, no API key). One FAQ entry = one chunk; embedded text =
`question + keywords + answer`. Point IDs are deterministic UUIDv5 under a fixed namespace that
**must match** the platform's `kb-point-id.ts` (NAV-231 prune-clobber safety).

Retrieval is an **LLM-called tool** (`search_knowledge_base`), not auto context injection. The tool is
heavily anti-fabrication hardened (NAV-214): out-of-scope keyword gate pre-embedding, confidence
thresholds, keyword-boost rescue, top-K chunk surfacing, telemetry to `calls.metrics.tool_calls[]`.
Ingest via `scripts/populate_qdrant.py` (`--source` or `--auto` driven by `repopulate_kb` flags;
idempotent point-count skip). Verify with `eval/check_kb.py --ssh`.

### 2.4 Tools / function-calling (doc 03)

`ToolRegistry` maps names → factories. Always-on platform tools: `search_knowledge_base`,
`request_escalation`/`confirm_escalation` (two-stage human escalation), `start_flow`,
`select_language` (auto when multilingual), `get_current_time`, `collect_csat` (universal),
`end_call`, and `transfer_to_<name>` sub-agent handoff (only `AppealAgent` wired, with dynamic
`set_<field>`/`confirm_and_submit`). Flows are `.flow.md` files driven step-by-step by `FlowEngine`.

### 2.5 Human handoff / operator transfer (docs 03, 09)

Decision core (`transfer_guardian.py`): OOH gate (Asia/Tashkent) → AppealAgent callback intake;
strict reasons require a reflection + `confirm_escalation`; lenient reasons fire immediately;
loop-break after 2 unconfirmed requests. Two fire modes:
- **web** (default): PATCH call → `POST /calls/:id/operator`; operator joins the **same LiveKit room**
  via a minted token; agent polls status, waits for a stable `operator:` participant, then
  `_remove_ai_from_room()`. No SIP fallback.
- **sip**: LiveKit `transfer_sip_participant` REFER moves the caller's SIP leg via Asterisk to a
  human extension/queue. Operator-leg outcome is resolved later by PBX webhook + reconciliation.

`end_call` deletes the LiveKit room (drops AI + SIP leg).

### 2.6 Observability, resilience, deploy (doc 04)

Observability: stdlib server on `:8082` (`/health`, `/metrics`, `/network/*`); Prometheus `navai_*`
families (tenant-labeled) bridged cross-process via JSON-file persistence (not the prometheus_client
multiproc collector — bridge hardcodes yandex/gemini/yandex labels). Network observer monkey-patches
aiohttp/httpx; Langfuse/OTel env-gated.

Resilience is narrow: **no circuit breakers / backoff**. Only failovers are the manual
`DISABLE_CUSTOM_STT` kill-switch (custom STT → Yandex at factory time), per-turn graceful degradation
(empty transcript / partial audio, WS `*_MAX_SECONDS` ceilings, ChunkedStream retries gated to
zero-audio-only), and the fully-wired silence monitor. No mid-call vendor failover.

The agent serves no inbound control API — `src/api/` is the **outbound** `PlatformClient` (10s
timeout, fail-fast validation, every method exception-safe). Deploy: Dockerfile (`python:3.11-slim`,
`PYTHONPATH=/app/src`) → compose (agent + agent-monitor sidecar + qdrant) → CI `deploy.yml`
(tests+eval gate → build/push GHCR → SSH deploy with rollback trap + 24× health gate + smoke + git
tag; staging auto, prod manual). `repopulate-kb.yml` updates Qdrant in-place without redeploy.

### 2.7 Platform API (doc 05)

**Express 4 + TypeScript + Mongoose + ioredis** (NOT NestJS), package `@navai/api`, `:3000` +
metrics `:9091`. Routers under `/api/v1`: auth, calls, internal, murojatlar, pbx, webhooks,
analytics, admin, config, users, kb, tenants, system, operator(s), soundflare, langfuse. Swagger is
runtime-generated from routes + zod.

Multi-tenancy is **row-level** (`tenant_id` everywhere) with three resolution paths: JWT `tid` →
`tenantIsolation` helpers; per-tenant agent keys (`*_AGENT_API_KEY` → slug, override body
`tenant_id`); phone-number resolution against `Tenant.phone_numbers`. There is **no
`MULTI_TENANT_STRICT_ROUTING`** here — strict = `ALLOW_DEFAULT_TENANT_FALLBACK` off in prod + scoped
keys. Auth: RS256 JWT (HS256 dev fallback, Redis jti blacklist) for humans; `x-api-key`
INTERNAL_API_KEY for `/internal/*`; PBX_CRM_TOKEN for webhooks; RBAC + CSRF + rate limits.

Agent↔platform critical path: `POST /internal/calls` (idempotent by call_sid / 60s phone window),
`PATCH /internal/calls/:id` (ignores client duration), `GET /internal/caller-history`. `updateCall`
schedules recording fetch, summarization, transcription, evaluation, and emits realtime events. PBX
reconciliation is dual: real-time `/webhooks/pbx` + per-minute cron matching operators by
`ext_number`.

### 2.8 Dashboard (doc 06)

**React 18 + Vite 5** (`:3001` dev), react-router 6, TanStack Query 5, Zustand 4, axios,
livekit-client. **Cookie-based httpOnly sessions** (not bearer), CSRF double-submit, auto-refresh on
401. Pages: Login, role-switched Dashboard/Calls/Murojatlar, KB, Metrics (Langfuse), Experiments,
Settings, System, SuperAdmin Tenants. Roles: operator/agent/manager/admin/super_admin. The standout
feature is the **operator handoff overlay** (WS `/operator/ws` + SSE fallback → accept → LiveKit room
join). Tenant resolved from **subdomain** → `GET /tenants/:slug/public` for branding; isolation is
server-side. Real agent config editing lives in the super-admin tenant provisioning flow (the
per-tenant Settings tabs are mostly mock UI). Built 2-stage → **nginx:alpine** SPA + `/api` proxy.

### 2.9 Platform infra & monitoring (doc 07)

`docker-compose-platform.yml`: api (3000 + 9091), dashboard (8080/3001→80), mongodb (27017), redis
(6379) on `navai-platform-network`; node-exporter/cAdvisor/promtail co-located. Required vars:
`REDIS_PASSWORD`, `JWT_SECRET`, `INTERNAL_API_KEY`. `docker-compose-monitoring.yml`: prometheus
(9090, 30d) + grafana (3200) by default; loki/promtail/alertmanager/caddy behind `--profile full`;
Langfuse behind `--profile langfuse`. CI: `test.yml`, `deploy.yml` (build → GHCR dual tags → SSH
deploy with rollback trap → smoke), `deploy-monitoring.yml` (hash-gated config sync). Backups:
mongodump → age-encrypt → GCS, systemd daily timer, node-exporter textfile metrics + staleness
alerts. SSL: Caddy auto-LetsEncrypt for monitoring; nginx+certbot for platform.
**Flagged drift:** committed Telegram token in `alertmanager.yml`, Caddy `grafana:3000` vs `3200`,
port drift in `health-check.sh`, stale CI docs.

### 2.10 Telephony integration (doc 09)

INBOUND: PSTN → Axona → Asterisk `[inc]` exten → `Macro(timed-dial-livekit)` → TLS to LiveKit inbound
trunk → dispatch rule (room `<caller>_<suffix>`) → agent worker (dispatched by name) → STT/LLM/TTS.
Tenant by **called** number from dispatch/room metadata. OUTBOUND: operator-PSTN via
`[outbound-calls]` → `trunk-out` (surfaced to platform only via PBX webhook); `[livekit-inbound]` →
`outbound-calls` is the unused seam for programmatic dialing; `[otp-*]` Asterisk-native origination.
TRANSFER: shared decision core, web vs sip modes (see 2.5). Provider seam: carrier abstracted behind
Asterisk trunks; a Pakistani swap = trunk wizards + PK numbering plan (`_+92…`) + recording/egress +
change OOH TZ `Asia/Tashkent → Asia/Karachi` (the one code coupling).

---

## 3. End-to-End Call Lifecycle

### 3.1 Inbound
1. PSTN caller dials the tenant DID → Axona/carrier → Asterisk registered trunk → context `[inc]`.
2. `[inc]` exten runs `Macro(timed-dial-livekit, PJSIP/<DID>@livekit_<project>, …)` → TLS 5061 to
   `*.sip.livekit.cloud`. The dialed user part is the `called` number.
3. LiveKit inbound SIP trunk (number + Asterisk-IP allowlist, no SIP auth) matches → dispatch rule
   creates room `<caller>_<suffix>` and dispatches the agent **by AGENT_NAME**.
4. Agent entrypoint: `extract_called_phone` (metadata, not room name) → `resolve_tenant_for_call`
   (strict fail-closed) → `extract_caller_phone` → caller history → effective language → build
   STT/TTS/LLM → warm KB → `create_call` (call_sid = room_name) → `session.start` → `ctx.connect`.
5. Conversation: STT → LLM (tools: KB search, flows, escalation, CSAT…) → TTS. SIP-participant timing
   wired into `TelephonyLatencyTracker`. Transcript synced to platform every 5 turns.
6. `end_call` (or silence goodbye) deletes the room; `lifecycle/shutdown.py` PATCHes final record
   (status, duration, transcript, csat, transfer metadata); platform schedules recording fetch +
   summarization + evaluation.

### 3.2 Outbound
- **Operator outbound (today):** human dials from CRM softphone → Asterisk `[outbound-calls]` →
  `trunk-out` → <OUTBOUND_CARRIER_HOST> → callee. Platform learns of it only via the PBX `OUTGOING` webhook
  (`direction:'outbound'`, `metadata.source='operator_outbound'`).
- **Programmatic (seam, not wired):** LiveKit `CreateSIPParticipant` against an outbound trunk →
  `[livekit-inbound]` → `[outbound-calls]` → `trunk-out`. No `create_sip_participant`/`originate` in
  either repo today.
- **OTP origination (seam):** Asterisk-native `[otp-*]` plays a generated `.sln` file.

### 3.3 Transfer
1. LLM calls `request_escalation(reason)` → `transfer_guardian.evaluate_request` (OOH gate →
   AppealAgent; strict → reflection + `confirm_escalation`; lenient → fire; loop-break after 2).
2. **web:** PATCH call → `POST /calls/:id/operator` → operator accepts in dashboard → minted LiveKit
   token → operator joins room → agent polls status, sees `operator:` participant, removes AI;
   caller ↔ operator remain.
3. **sip:** `transfer_sip_participant` REFER → Asterisk `_XXX` → `trunk-human`/queue; operator-leg
   outcome resolved later by PBX webhook + per-minute reconciliation cron.

---

## 4. Multi-Tenant Model (across all planes)

| Plane | Tenant key | Mechanism |
|-------|-----------|-----------|
| **Telephony** | Called DID | 5 artifacts must agree on the dialed number: Asterisk trunk + `[inc]` exten + LiveKit inbound trunk + dispatch rule (`--agent-name`) + tenant YAML `phone_numbers`. |
| **Agent** | Called phone → slug | `extract_called_phone` → `TenantRegistry` (YAML index, then A/B cohort, then Platform-API fallback, then default). One binary, N YAMLs. `MULTI_TENANT_STRICT_ROUTING` fails closed. |
| **Agent config** | YAML | `_defaults.yaml ← <slug>.yaml ← environments.yaml`, Pydantic-validated. Per-tenant API key in `config/api_keys.py` (code change today). |
| **RAG** | `<slug>_faq` Qdrant collection | Deterministic UUIDv5 ids shared with platform. |
| **Platform API** | `tenant_id` row-level | JWT `tid` (humans), scoped `*_AGENT_API_KEY` (agent, overrides body), phone resolution (webhooks). `ALLOW_DEFAULT_TENANT_FALLBACK` off in prod = strict. |
| **Dashboard** | Subdomain → slug | `GET /tenants/:slug/public` for branding; isolation server-side by session. Super-admin provisions tenants cross-tenant. |

Onboarding a tenant touches: tenant YAML + KB JSON + Qdrant collection + `environments.yaml`
prod/dev entries + `api_keys.py` (code) + LiveKit dispatch/trunk + Asterisk trunk/exten + Platform
`Tenant` record + operators. See the runbook in `BLUEPRINT-PLAN.md`.

---

## 5. External Dependencies & Env-Var Map

| Category | Dependency | Key env vars |
|----------|-----------|--------------|
| **RTC bridge** | LiveKit Cloud | `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`, `LIVEKIT_OPERATOR_TOKEN_TTL_SECONDS` |
| **Agent identity/routing** | — | `AGENT_NAME`, `AGENT_PHONE_NUMBER`, `MULTI_TENANT_STRICT_ROUTING`, `SINGLE_TENANT_MODE`, `CONFIG_ENV`, `NAVAI_NUM_IDLE_PROCESSES` |
| **STT** | navai_ws / yandex / custom-GPU | `NAVAI_WS_STT_URL`, `NAVAI_API_KEY`, `NAVAI_WS_STT_MAX_SECONDS`; `YANDEX_API_KEY`/`YANDEX_IAM_TOKEN`/`YANDEX_FOLDER_ID`; `CUSTOM_STT_URL`, `DISABLE_CUSTOM_STT`; `NAVAI_STT_URL`, `OPERATOR_STT_URL` |
| **TTS** | navai_ws / yandex / custom-GPU | `NAVAI_WS_TTS_URL`, `NAVAI_WS_VOICE_ID`, `NAVAI_WS_TTS_MAX_SECONDS`; `YANDEX_VOICE_ID`; `CUSTOM_TTS_URL`, `CUSTOM_VOICE_ID`; `NAVAI_TTS_URL`, `NAVAI_VOICE_ID` |
| **LLM** | Gemini (Vertex ADC) / openai / vLLM | (Gemini: ADC, no key, `GOOGLE_CLOUD_PROJECT`/`_LOCATION`); `OPENAI_API_KEY`, `OPENAI_BASE_URL`; `CUSTOM_LLM_URL`, `CUSTOM_LLM_MODEL`, `CUSTOM_LLM_API_KEY`; `LEXANTEI_*` |
| **Embeddings/RAG** | Gemini + Qdrant | `QDRANT_URL`, `QDRANT_COLLECTION`, `GEMINI_EMBEDDING_MODEL`, `RAG_HIGH/MEDIUM_CONFIDENCE`, `NAVAI_POPULATE_KB`/`POP_*` |
| **Platform DB** | MongoDB, Redis | `MONGODB_URI`, `MONGO_USER`, `MONGO_PASSWORD`; `REDIS_PASSWORD`, `REDIS_URL` |
| **Platform auth** | JWT, internal key | `JWT_ALGORITHM`/`JWT_PRIVATE_KEY`/`JWT_PUBLIC_KEY` (or `JWT_SECRET`), `INTERNAL_API_KEY`, per-tenant `*_AGENT_API_KEY`, `PLATFORM_API_URL`, `PLATFORM_API_JWT`, `ALLOW_DEFAULT_TENANT_FALLBACK`, `CORS_ORIGIN`, `COOKIE_DOMAIN` |
| **LLM ops (platform)** | Gemini/Vertex | `GEMINI_API_KEY`, `GEMINI_SUMMARIZATION_MODEL`, `GEMINI_VERTEX_PROJECT/LOCATION`, `CALL_EVALUATION_MODEL` |
| **Telephony/PBX** | Asterisk + carrier CRM | `PBX_DOMAIN`, `PBX_API_KEY`, `PBX_NUMBER`, `PBX_CRM_TOKEN`, `CALL_RECONCILIATION_*`, transfer `OPERATOR_HANDOFF_MODE`/`OPERATOR_SIP_FALLBACK` |
| **Observability** | Prometheus/Grafana/Loki/Langfuse | `HEALTH_PORT` (8082), `METRICS_PORT` (9091 API; 9090 declared-unused in agent), `NETWORK_OBS_ENABLED`, `LANGFUSE_*`, `SOUNDFLARE_*` |
| **Monitor sidecar** | Telegram | `MONITOR_INTERVAL_SECONDS`, `MONITOR_HEALTH_URL`, `MONITOR_NOTIFY_MODE`, `MONITOR_TELEGRAM_*` |
| **Backups** | GCS + age | `BACKUP_GCS_BUCKET`, `BACKUP_AGE_RECIPIENT`, `BACKUP_SERVICE_ACCOUNT_JSON`, `BACKUP_TEXTFILE_DIR` |

Gemini reasoning + embeddings authenticate via **Vertex AI ADC (attached service account, no API
key)** on the agent side; the platform's Gemini ops use `GEMINI_API_KEY`.

---

## 6. Repo Generations — Which Is Authoritative

| Generation | Repos | Role | Authoritative? |
|-----------|-------|------|----------------|
| **Gen 1 (legacy)** | `navai-voice-agent` (pre-restructure) | The original agent before the split; superseded. | No — historical only. |
| **Gen 2 (current, production)** | `navai-agent` (Python LiveKit agent) **+** `platform` / `navai-platform` (Express API + React dashboard + `packages/navai-shared`) | The live system. `navai-agent` is the busiest repo (234 commits/60d); platform 170/60d. | **YES** — both are the source of truth. Agent = conversational core; platform = control plane. |
| **Gen 3 (experiment)** | `navai-monorepo` (pnpm workspace: `apps/` web+landing, `services/api` NestJS, `ml/` FastAPI STT/TTS, `packages/types`) | Aspirational FAANG-style consolidation that de facto became a **landing-page repo**. RESTRUCTURE_PLAN explicitly **removed the agent**; `services/api` is an older fork of the platform. | No for production — except in practice the landing site. **Use only as a structural/taxonomy reference**, do not fork. |

**Conclusion:** build the Urdu blueprint by clone-and-adapting **`navai-agent`** for the agent core
and treating **`navai-platform`** as the authoritative control-plane reference; borrow only the
monorepo's *patterns* (apps/services/packages/ml/infrastructure taxonomy, pnpm + `packages/types`,
Makefile facade, Docker/Prometheus layering). The actionable plan is in `BLUEPRINT-PLAN.md`.
