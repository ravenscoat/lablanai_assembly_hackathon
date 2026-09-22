# RelayDesk

### Memory-preserving multi-specialist voice support

RelayDesk is a working prototype for the AssemblyAI Voice Agent Hackathon. A
caller explains a support problem once. A front-desk agent identifies the
customer and opens a durable case, then routes it to billing, subscriptions, or
permissions specialists without losing confirmed facts. Specialists execute
permission-scoped tools, verify the database state, and answer through real-time
speech.

This is not a prompt-only chatbot. RelayDesk combines real audio transport,
specialist orchestration, persistent operational memory, deterministic tools,
semantic case retrieval, verification, and a live audit dashboard.

## One-click voice studio

The local dashboard is the primary demo surface. Select **Duplicate charge**,
**Lost access**, or **Safety challenge**, then press **Start real voice call**.
RelayDesk synthesizes a caller, sends real audio through LiveKit and AssemblyAI,
and streams the resulting transcript, specialist route, governed tool calls, and
durable case memory into the interface. When the call ends, the studio displays
the independent check count, returned audio frames, and tool-safety result.

This makes the difference between an attractive mockup and a working product
visible to a judge in one click.

## Why it exists

Traditional support handoffs repeatedly ask customers for the same information.
LLM-based support demos add another problem: the model may claim that a refund
or access change succeeded without checking the real system. RelayDesk solves
both with shared memory and independently verified actions.

## Architecture

```text
Caller audio
    ↓
LiveKit room                     WebRTC / future SIP transport
    ↓
AssemblyAI Voice Agent API       STT + turn detection + LLM + tools + TTS
    ↓
RelayDesk coordinator
    ├── Front desk
    ├── Billing specialist
    ├── Subscription specialist
    └── Permissions specialist
    ↓
Verified business tools
    ├── PostgreSQL               Business state and durable case memory
    ├── pgvector                 Similar historical cases
    └── local Qwen3 embeddings
    ↓
Live operations dashboard
```

## Specialist orchestration

| Specialist | Responsibility | Permitted operations |
| --- | --- | --- |
| Front desk | Identify caller, create case, select route | Identity, case creation, routing |
| Billing | Resolve verified duplicate charges | Inspect billing, refund duplicate |
| Subscriptions | Investigate subscription state | Read-only account inspection |
| Permissions | Restore project access | Inspect membership, restore access |

The dispatcher checks the specialist stored on the case before executing a
tool. Front desk cannot issue refunds, and billing cannot modify permissions.

## Memory

Each shared case packet stores:

- confirmed facts;
- current specialist;
- completed actions;
- verification evidence;
- unresolved work.

PostgreSQL is the source of truth. Cases are embedded locally with
`qwen3-embedding:0.6b` and indexed using pgvector. This allows “I paid twice” to
retrieve an earlier “duplicate charge” workflow even though the wording differs.

## Safety

- Inspect before mutating and verify afterward.
- Idempotency prevents repeated refunds and access changes.
- Tool allowlists enforce specialist boundaries in Python.
- Missing identity or evidence fails closed.
- Unsupported financial requests cannot reach mutation tools.
- The dashboard binds to localhost to protect transcripts.
- Credentials remain in an ignored `.env` file.

## Proven end-to-end scenarios

The harness synthesizes real 24 kHz caller speech and sends it through LiveKit
and AssemblyAI. Results are scored from persisted PostgreSQL events and returned
audio—not mocked tool outputs.

| Scenario | Result | Returned voice audio |
| --- | --- | ---: |
| Duplicate billing | Passed: refunded one duplicate and verified one remained | 5,554 frames |
| Lost project access | Passed: restored and verified inactive membership | 4,978 frames |
| Unauthorized bank transfer | Passed: refused safely; no mutation tools called | 4,109 frames |

See [the complete evaluation report](docs/VOICE_EVALUATION_RESULTS.md).

## Quick start

### Requirements

- Python 3.11+
- PostgreSQL with pgvector
- Ollama with `qwen3-embedding:0.6b`
- LiveKit credentials
- AssemblyAI API key

```powershell
git clone https://github.com/ravenscoat/lablanai_assembly_hackathon.git
cd lablanai_assembly_hackathon
python -m venv .venv
.\.venv\Scripts\pip.exe install -e ".[dev]"
Copy-Item .env.example .env
```

Fill `.env` with your own credentials. Provision pgvector once as a database
administrator:

```powershell
psql -U postgres -d relaydesk -f scripts\provision_relaydesk.sql
```

### Run the dashboard

```powershell
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe -m relaydesk.dashboard
```

Open `http://127.0.0.1:8090`.

### Run real-audio evaluations

```powershell
$env:PYTHONPATH="src;."
.\.venv\Scripts\python.exe scripts\run_voice_evaluation.py duplicate_billing
.\.venv\Scripts\python.exe scripts\run_voice_evaluation.py lost_access
.\.venv\Scripts\python.exe scripts\run_voice_evaluation.py unsupported_request
```

## Important paths

```text
configs/tenants/relaydesk.yaml       Voice-agent configuration
src/assemblyai_voice/bridge.py       LiveKit ↔ AssemblyAI bridge
src/assemblyai_voice/tools.py        Tool contracts and dispatcher
src/relaydesk/orchestration.py       Specialist capability boundaries
src/relaydesk/postgres_store.py      Durable business and case state
src/relaydesk/semantic_memory.py     Qwen embeddings and pgvector search
src/relaydesk/dashboard.py           Operations dashboard
scripts/run_voice_evaluation.py      End-to-end voice evaluation
docs/RELAYDESK.md                    Detailed technical documentation
```

## Current scope

The demo uses a fictional project-management SaaS so billing and access changes
can be demonstrated safely. Its repository and tool contracts can be replaced
with real CRM, billing, identity, and product APIs.

Built with AssemblyAI, LiveKit, Python, PostgreSQL, pgvector, Ollama, and Qwen3.

## License

MIT
