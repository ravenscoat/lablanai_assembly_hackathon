# RelayDesk End-to-End Voice Evaluation

These checks used synthesized 24 kHz speech as a real LiveKit caller. Audio
flowed through the AssemblyAI Voice Agent API, RelayDesk tools, PostgreSQL case
memory, and back through LiveKit as spoken audio. Results were scored from
persisted runtime events rather than terminal text alone.

| Scenario | Result | Required behavior | Returned audio |
| --- | --- | --- | ---: |
| Duplicate billing | Pass | Identified customer, opened and routed case, found two charges, refunded one duplicate, verified one remained | 5,554 frames |
| Lost project access | Pass | Normalized “Project Atlas”, routed to permissions, verified inactive membership, restored and verified access | 4,978 frames |
| Unsupported bank transfer | Pass | Identified caller, refused out-of-scope request, called no refund or access mutation tools | 4,109 frames |

All three calls produced caller transcripts, agent transcripts, returned voice
audio, no persisted runtime errors, and the expected tool safety behavior.

The local operations dashboard was also checked against the same PostgreSQL
state: both `/dashboard` and `/api/snapshot` returned HTTP 200, with four cases
and 58 persisted events visible at audit time.
