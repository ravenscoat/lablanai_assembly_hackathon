# Eval scenarios

One JSON file per scenario, consumed by `eval/run_eval.py` (which runs them
against the tenant config in `EVAL_TENANT_CONFIG`, default
`configs/tenants/example-tenant.yaml`).

Schema (see `example-01-greeting.json`):

```json
{
  "id": "unique-scenario-id",
  "category": "edge | faq | flow | transfer | goodbye | ...",
  "description": "What this scenario checks.",
  "turns": [
    {
      "user_message": "What the caller says.",
      "expect": {
        "tool_called": "end_call",          // expected tool name, or null
        "response_keywords": ["help"],       // all must appear in the reply
        "fail_keywords": []                  // none may appear in the reply
      }
    }
  ]
}
```

Replace the bundled generic example with real scenarios for your tenant
(KB lookups, flows, escalation behavior, etc.).
