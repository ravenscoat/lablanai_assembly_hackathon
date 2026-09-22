"""
KB Retrieval Health Check — verifies the knowledge base returns relevant answers.

Runs inside the agent container on the target server via SSH.
Tests a known query against Qdrant and validates:
  1. Qdrant is reachable
  2. Embedding generation works (Gemini)
  3. Search returns results with score above threshold
  4. Answer contains expected keywords

Configure the queries + collection for your tenant via env vars (see below) or
by editing TEST_QUERIES. The sample queries below are generic English placeholders
for the bundled `example-tenant` KB — replace them with real tenant queries.

Usage (local):
    KB_COLLECTION=example_tenant_faq python eval/check_kb.py --ssh \
        --host user@<your-server>

Exit codes:
    0 = KB healthy
    1 = KB check failed
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

# Test queries with expected results. Replace with real tenant queries/keywords.
# These generic samples match the bundled example-tenant FAQ entries.
TEST_QUERIES = [
    {
        "query": "What are your business hours?",
        "min_score": 0.70,
        "expected_keywords": ["hours"],
        "description": "Business hours lookup",
    },
    {
        "query": "How do I reset my password?",
        "min_score": 0.70,
        "expected_keywords": ["password"],
        "description": "Password reset",
    },
]

# Env-driven so the check is not tied to any specific tenant/project.
COLLECTION = os.getenv("KB_COLLECTION", "example_tenant_faq")
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
GCP_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "")
GCP_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "europe-west4")

# Script that runs INSIDE the docker container
CONTAINER_SCRIPT = '''
import requests, json, os, sys

from google import genai
from google.genai.types import EmbedContentConfig

queries = {queries_json}
collection = "{collection}"
qdrant_url = "{qdrant_url}"

_project = "{gcp_project}" or os.getenv("GOOGLE_CLOUD_PROJECT")
_location = "{gcp_location}" or os.getenv("GOOGLE_CLOUD_LOCATION", "europe-west4")
_client_kwargs = {{"vertexai": True, "location": _location}}
if _project:
    _client_kwargs["project"] = _project
client = genai.Client(**_client_kwargs)
results = []

for q in queries:
    try:
        # 1. Generate embedding
        emb_result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=q["query"],
            config=EmbedContentConfig(output_dimensionality=768),
        )
        embedding = emb_result.embeddings[0].values

        # 2. Query Qdrant
        resp = requests.post(
            f"{{qdrant_url}}/collections/{{collection}}/points/query",
            json={{"query": embedding, "limit": 3, "with_payload": True}},
            timeout=10
        )
        data = resp.json()
        points = data.get("result", {{}}).get("points", [])

        if not points:
            results.append({{"query": q["query"], "passed": False, "reason": "No results returned"}})
            continue

        top = points[0]
        score = top["score"]
        answer = top["payload"].get("answer", "")

        # 3. Check score threshold
        if score < q["min_score"]:
            results.append({{
                "query": q["query"],
                "passed": False,
                "reason": f"Score {{score:.3f}} below threshold {{q['min_score']}}"
            }})
            continue

        # 4. Check keywords in answer
        answer_lower = answer.lower()
        missing = [kw for kw in q["expected_keywords"] if kw not in answer_lower]
        if missing:
            results.append({{
                "query": q["query"],
                "passed": False,
                "reason": f"Missing keywords: {{missing}}",
                "score": score,
                "answer_preview": answer[:150]
            }})
            continue

        results.append({{
            "query": q["query"],
            "passed": True,
            "score": score,
            "answer_preview": answer[:150]
        }})

    except Exception as e:
        results.append({{"query": q["query"], "passed": False, "reason": str(e)}})

# Output JSON
print(json.dumps(results))
'''


def run_via_ssh(host: str, container: str = "voice-agent", port: int = 22) -> list[dict]:
    """Run KB check inside docker container via SSH using base64 to avoid quoting issues."""
    import base64

    script = CONTAINER_SCRIPT.format(
        queries_json=json.dumps(TEST_QUERIES, ensure_ascii=False),
        collection=COLLECTION,
        qdrant_url=QDRANT_URL,
        gcp_project=GCP_PROJECT,
        gcp_location=GCP_LOCATION,
    )

    # Base64 encode the script to avoid shell quoting issues
    encoded = base64.b64encode(script.encode()).decode()

    ssh_command = (
        f"docker exec {container} python3 -c "
        f"\"import base64;exec(base64.b64decode('{encoded}').decode())\""
    )

    cmd = [
        "ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=15",
        "-p", str(port),
        host,
        ssh_command,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

    if result.returncode != 0:
        print(f"SSH/Docker command failed:\n{result.stderr}")
        return [{"query": "ALL", "passed": False, "reason": f"SSH failed: {result.stderr[:200]}"}]

    try:
        return json.loads(result.stdout.strip())
    except json.JSONDecodeError:
        print(f"Failed to parse output:\n{result.stdout}")
        return [{"query": "ALL", "passed": False, "reason": f"Parse error: {result.stdout[:200]}"}]


def main() -> int:
    parser = argparse.ArgumentParser(description="KB retrieval health check")
    parser.add_argument("--ssh", action="store_true", help="Run via SSH to prod server")
    parser.add_argument("--host", required=True, help="SSH host (e.g. user@<your-server>)")
    parser.add_argument("--port", type=int, default=22, help="SSH port")
    parser.add_argument("--container", default="voice-agent", help="Docker container name")
    args = parser.parse_args()

    if not args.ssh:
        print("ERROR: Only --ssh mode is supported (runs inside prod container)")
        return 1

    print(f"🔍 KB Health Check — querying via SSH to {args.host}")
    print(f"   Container: {args.container}")
    print(f"   Collection: {COLLECTION}")
    print(f"   Test queries: {len(TEST_QUERIES)}\n")

    results = run_via_ssh(args.host, args.container, args.port)

    # Report
    all_passed = True
    for r in results:
        if r["passed"]:
            print(f"  ✅ PASS  score={r.get('score', 'N/A'):.3f}  {r['query'][:50]}")
        else:
            print(f"  ❌ FAIL  {r['query'][:50]}")
            print(f"           Reason: {r['reason']}")
            all_passed = False

    print()
    if all_passed:
        print("🎉 KB retrieval check PASSED — all queries returned valid results")
        return 0
    else:
        print("❌ KB retrieval check FAILED — knowledge base may be degraded")
        return 1


if __name__ == "__main__":
    sys.exit(main())
