# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

## Project Overview

**urdu-voice-agent** is a multi-tenant voice AI agent building block on the
LiveKit Agents SDK. A single binary serves N tenants, each defined entirely by
YAML config. The LLM drives all decisions — KB search, flow activation, tool
calls, topic handling.

This is a **blueprint/scaffolding**: it is runnable with a managed STT/TTS
provider and Gemini, and exposes clearly-marked **Urdu integration seams**
(STUBs) for plugging in an Urdu speech engine. It does not ship a real Urdu
engine. See the "Urdu integration seams" section of `README.md`.

## Commands

```bash
# Install
uv venv && uv pip install -e ".[dev]"

# Run (PYTHONPATH=src is required — main.py uses flat imports like `from config.registry import ...`)
PYTHONPATH=src python src/main.py start

# Test (pyproject sets pythonpath=src for pytest)
pytest tests/unit -v
pytest tests/integration -v
pytest tests --cov=src --cov-report=html

# Lint & Format
ruff check src tests scripts --fix
black src tests scripts

# Populate a tenant KB into Qdrant
PYTHONPATH=src python scripts/populate_qdrant.py \
    --source knowledge/example-tenant/faq_source.json \
    --collection example_tenant_faq

# Local dev without the real Platform API (in-memory stand-in)
python scripts/mock_backend.py          # port 3000

# Docker (agent + Qdrant). Health server is exposed on 8082.
docker compose up --build
```

Two ports: `AgentServer` listens on **8088** (LiveKit worker); the
observability health server listens on **`observability.health_port`** (default
**8082**, also the Dockerfile `EXPOSE`). `/health`, `/network/topology`, and
`/network/telephony` live on the health port.

## Architecture

### Entrypoint & Lifecycle

`src/main.py` creates an `AgentServer`. On each call:
1. SIP metadata → extract the **called** phone number → resolve `TenantConfig`
2. `VoiceFactory` creates STT/TTS/LLM/VAD from config
3. `AgentFactory.create_agent()` builds a `TenantAgent` with resolved tools + sub-agents
4. `SessionBuilder.build()` creates the `AgentSession`
5. Session starts; lifecycle callbacks handle events + shutdown

### Config-Driven Multi-Tenancy

- `configs/tenants/_defaults.yaml` — shared defaults all tenants inherit.
- `configs/tenants/_template.yaml` — documented skeleton to copy per tenant.
- `configs/tenants/example-tenant.yaml` — minimal working example.
- `configs/tenants/<slug>.yaml` — per-tenant overrides.
- `src/config/schema.py` — `TenantConfig` Pydantic model (source of truth for
  config shape; YAML keys must mirror this schema).
- `src/config/loader.py` — deep-merge + `environments.yaml` overrides + Platform
  API fallback.
- `src/config/registry.py` — `TenantRegistry`, maps phone numbers → configs.

Routing is by the **called** phone number (`utils/phone.extract_called_phone` →
registry → deep-merge). `MULTI_TENANT_STRICT_ROUTING=true` fails closed.

Per-tenant Platform API keys are resolved data-driven in `src/config/api_keys.py`
from `{SLUG_UPPER}_AGENT_API_KEY` (slug uppercased, non-alphanumeric → `_`).
Onboarding a tenant is a config/env change, not a code edit.

### Voice Pipeline (`src/pipeline/`)

- `voice_factory.py` — string-keyed factory for STT/TTS/LLM/VAD per provider.
  **The single provider swap file** (`create_stt_for_language`,
  `create_tts_for_language`, `create_llm`, `_LANG_MAP`).
- `session_builder.py` — assembles `AgentSession`.
- `providers/` — custom STT/TTS implementations subclassing
  `livekit.agents.stt.STT` / `tts.TTS`. Includes the Urdu STUB providers
  `urdu_stt.py` / `urdu_tts.py`.

Default stack: managed LiveKit inference STT (deepgram) → Gemini LLM (Vertex
ADC) → managed inference TTS (cartesia), Silero VAD.

### Multilingual Tenants

Tenants declare languages via `languages.available` + `languages.default`.
>1 entry enables the auto `select_language` tool, per-language STT/TTS selection,
and a multilingual greeting. `_LANG_MAP` maps ISO codes → locales (`ur` → `ur-PK`).

### Tools (`src/tools/`), Flows (`src/flows/`), Sub-agents (`src/agents/sub_agents/`)

- `registry.py` maps tool names → factories. Always-on platform tools:
  `search_knowledge_base`, `request_escalation`/`confirm_escalation` (two-stage
  escalation), `get_current_time`, `end_call`, `start_flow`, `collect_csat`,
  `select_language` (auto when multilingual).
- Flows are `.flow.md` files driven step-by-step by `FlowEngine`.
- The only wired sub-agent type is the `appeal` (callback-intake) agent.

### RAG (`src/knowledge/`)

One Qdrant collection per tenant (`<slug>_faq`), `gemini-embedding-001` (Vertex,
no key). Deterministic UUIDv5 point ids (`point_id.py`) — must match the
platform's scheme. Retrieval is an LLM-called tool (`search_knowledge_base`),
anti-fabrication hardened (out-of-scope keyword gate, confidence thresholds).

### Observability / Resilience / Platform client

- `src/observability/` — stdlib health server (`:8082`), Prometheus metrics,
  network-topology HTTP observer, Langfuse hooks.
- `src/resilience/` — config-driven silence monitor.
- `src/api/` — the **outbound** `PlatformClient` (create/update calls, caller
  history). The agent serves no inbound control API.

### A/B Experiments

Variant YAMLs sharing `experiment.id` are grouped into cohorts and
weighted-random picked per call. All variants must share tenant identity
(`id`, `slug`, `name`, `phone_numbers`). Each tagged call carries
`metadata.experiment_id` + `metadata.variant`.

## Urdu integration seams

The core extension point. See `README.md` → "Urdu integration seams" for the
exact files/functions to edit (all marked `TODO(urdu)` in code):
`src/pipeline/providers/urdu_stt.py`, `urdu_tts.py`, the factory branches +
`_LANG_MAP` in `voice_factory.py`, the `ur` prompt fragments in
`src/agents/factory.py`, `get_time.py`, and `URDU_*` env vars.

## Key Conventions

- **Python 3.11+**, full type annotations; **Ruff** (line-length 100, rules
  E/F/W/I) + **Black** (line-length 100); **Pydantic v2**;
  **pytest-asyncio** (`asyncio_mode = "auto"`).
- `PYTHONPATH=src` — imports are relative to `src/`.
- Singleton registries (`TenantRegistry`, `ToolRegistry`, `AgentFactory`).

## Environment Variables

See `.env.example` (grouped: LiveKit, agent identity/routing, Platform link,
LLM, STT, TTS, RAG/Qdrant, observability, monitor). Gemini auth on the agent
side comes from Vertex AI ADC (attached service account) — no `GEMINI_API_KEY`.

## Related Docs

- `docs/ARCHITECTURE.md` — full system architecture (agent + platform +
  telephony context).
- `AGENTS.md` — short contributor guide (commit prefixes, PR conventions).
- `.claude/pr-rules.yaml` — per-repo PR config read by the `/create-pr` skill.
