# RelayDesk Hackathon Build

RelayDesk is a memory-preserving multi-agent voice support prototype. A caller
explains a connected problem once; specialist agents handle billing,
subscriptions, and permissions without losing confirmed facts during handoffs.

## Current architecture

```text
Caller -> LiveKit room -> RelayDesk bridge -> AssemblyAI Voice Agent API
                                      <- 24 kHz PCM reply audio <-
```

- LiveKit is the WebRTC and future SIP transport.
- AssemblyAI provides realtime STT, LLM, TTS, turn detection, interruption
  handling, and tool-call events over one WebSocket.
- The existing tenant registry selects prompt, greeting, voice, keyterms, and
  turn-detection settings from YAML.
- `configs/tenants/relaydesk.yaml` is the fictional SaaS demo tenant.
- Secrets stay in the ignored local `.env`.

## Working tools and shared memory

The agent now has real, deterministic tools rather than prompt-only behavior:

- identify a fictional customer (`alex@relaydesk.demo`)
- inspect duplicate charges and refund exactly one duplicate
- inspect a project membership and restore access
- create and route a durable support case
- read the shared handoff packet

PostgreSQL is the durable runtime source of truth. It persists customers,
charges, memberships, cases, JSONB handoff memory, and idempotency keys. SQLite
implements the same repository contract for fast, isolated unit tests. Tool
results are independently checked against saved state. Repeating a refund or
access-restoration request returns the first result instead of changing state
twice.

Every verified tool result can be attached to a case. The handoff packet keeps
confirmed facts, completed actions, evidence, unresolved tasks, and the current
specialist. This is how a billing or permissions specialist can continue without
asking the caller to explain the same problem again.

## Specialist orchestration

RelayDesk uses a controlled manager-and-specialists pattern. Front desk owns a new
case and may identify the customer, create the case, or route it. Billing,
subscriptions, and permissions each have a separate prompt and tool allowlist. The
dispatcher checks the persisted case owner before executing a specialist tool, so
the language model cannot bypass routing merely by naming a tool. A successful
handoff returns both the specialist instructions and the full shared case packet.

## Run the current voice bridge

```powershell
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe -m assemblyai_voice.bridge
```

The worker joins `ASSEMBLYAI_ROOM_NAME`, waits for one remote microphone track,
opens an AssemblyAI session, streams caller audio, and publishes reply audio.

## Verified

- AssemblyAI temporary-token request succeeds.
- LiveKit authentication succeeds and the worker joins `relaydesk-demo`.
- RelayDesk YAML loads through the existing multi-tenant registry.
- AssemblyAI session configuration tests pass.

## Next implementation order

The PostgreSQL implementation has been exercised through a complete isolated
integration scenario: identify customer, create case, route to billing, detect
and refund a duplicate, route to permissions, inspect membership, restore
access, and confirm all four evidence records in shared case memory.

Historical cases are also indexed in PostgreSQL with pgvector. The local
`qwen3-embedding:0.6b` Ollama model creates 1,024-dimensional embeddings, and
cosine distance retrieves similar prior workflows. Extension installation is a
one-time administrator operation in `scripts/provision_relaydesk.sql`; the
runtime database role does not receive extension-management privileges. A live
integration check correctly ranked a duplicate-payment case first for a
paraphrased refund query (similarity 0.797).

## Remaining implementation order

The localhost operations dashboard now displays live transcripts, specialist
routes, tools, verified evidence, and shared case memory from PostgreSQL.

Three real-audio end-to-end scenarios pass. See
`docs/VOICE_EVALUATION_RESULTS.md` for the measured results. The remaining
release activity is recording and editing the hackathon submission video.
