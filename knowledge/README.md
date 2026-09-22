# Knowledge bases (per-tenant)

Each tenant has its own knowledge base (KB), stored as a Qdrant vector
collection and seeded from a JSON source file in this directory.

## Layout

```
knowledge/
  <kb-dir>/
    faq_source.json        # the FAQ entries (one entry = one chunk)
    flows/                 # optional guided .flow.md flows for this tenant
      <flow-id>.flow.md
```

Conventions:

- `<kb-dir>` is the tenant's KB directory name (use the tenant slug, with
  hyphens; the populate script maps `youth_agency` ↔ `youth-agency`).
- The Qdrant collection name is `<kb-dir-with-underscores>_faq`
  (e.g. `example-tenant` → `example_tenant_faq`). Set this as
  `knowledge_base.collection` in the tenant YAML.
- Point IDs are deterministic UUIDv5 (see `src/knowledge/point_id.py`) and MUST
  match the platform's KB point-id scheme so KB sync/prune stays consistent.

## `faq_source.json` schema

A JSON array of objects with this shape (see `example-tenant/faq_source.json`):

```json
{
  "id": "unique_stable_id",
  "category": "free-text category label",
  "question": "The question (and paraphrases, separated by spaces).",
  "answer": "The answer the agent should give.",
  "keywords": ["optional", "boost", "terms"],
  "source": "provenance label, e.g. faq_v1"
}
```

The embedded text is `question + keywords + answer`. Authoring questions with a
few natural paraphrases improves retrieval recall.

## Populating Qdrant

```bash
# One tenant from its source file
PYTHONPATH=src python scripts/populate_qdrant.py \
    --source knowledge/example-tenant/faq_source.json \
    --collection example_tenant_faq

# Or auto mode, driven by the repopulate_kb flags in _defaults.yaml
PYTHONPATH=src python scripts/populate_qdrant.py --auto
```

Verify retrieval health with `eval/check_kb.py` (see its `--help`).
