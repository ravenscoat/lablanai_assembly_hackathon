"""
E2E Eval — Multi-turn behavioral test for the voice agent's LLM brain.

Tests that the agent calls the correct tools and responds appropriately
to known user inputs across multiple categories. Calls Gemini directly
with the same system prompt and tool definitions the production agent uses.

Features:
  - Multi-turn conversation support
  - Category-based reporting (FAQ, Transfer, Flow, Edge, Goodbye)
  - Keyword and fail-condition checking
  - JSON report artifact output
  - Model fallback on errors

Usage:
    python eval/run_eval.py

Auth: Uses Vertex AI via Application Default Credentials. On the VM this comes
from the attached service account (no env var needed). Locally, run
`gcloud auth application-default login` first.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from glob import glob
from pathlib import Path

# Load .env file if present (for local development)
_env_file = Path(__file__).parent.parent / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())

from google import genai
from google.genai import types

# --- Constants ---
SCENARIOS_DIR = Path(__file__).parent / "scenarios"
REPORTS_DIR = Path(__file__).parent / "reports"
CONFIG_PATH = Path(
    os.getenv(
        "EVAL_TENANT_CONFIG",
        str(Path(__file__).parent.parent / "configs" / "tenants" / "example-tenant.yaml"),
    )
)
MODEL = os.getenv("EVAL_MODEL", "gemini-2.5-flash")
FALLBACK_MODELS = ["gemini-2.5-flash-lite"]
GCP_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "")
GCP_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "europe-west4")
MAX_RETRIES = 3
RETRY_DELAY = 10
SCENARIO_DELAY = 3
# Minimum pass rate to pass CI.
MIN_PASS_RATE = 0.8


def load_system_prompt() -> str:
    """Load system prompt from tenant YAML config."""
    import yaml

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    return config["personality"]["system_prompt"].strip()


def get_tool_declarations() -> list[types.Tool]:
    """Define the agent's tools in Gemini function calling format."""
    tools = types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="search_knowledge_base",
                description="Search the knowledge base for answers to user questions about the tenant's services",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "query": types.Schema(type="STRING", description="Search query"),
                    },
                    required=["query"],
                ),
            ),
            types.FunctionDeclaration(
                name="escalate_to_human",
                description="Transfer the call to a human operator",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "reason": types.Schema(type="STRING", description="Reason for transfer"),
                    },
                    required=["reason"],
                ),
            ),
            types.FunctionDeclaration(
                name="start_flow",
                description=(
                    "Start a guided flow process. Set valid flow_id values to "
                    "match the tenant's .flow.md files under knowledge/<kb-dir>/flows/."
                ),
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "flow_id": types.Schema(type="STRING", description="Flow identifier"),
                    },
                    required=["flow_id"],
                ),
            ),
            types.FunctionDeclaration(
                name="end_call",
                description="End the current phone call",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "reason": types.Schema(type="STRING", description="Reason for ending"),
                    },
                ),
            ),
            types.FunctionDeclaration(
                name="get_current_time",
                description="Get the current date and time in the tenant's timezone",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={},
                ),
            ),
            types.FunctionDeclaration(
                name="transfer_to_appeal",
                description="Transfer to appeal sub-agent for formal complaint filing",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={},
                ),
            ),
        ]
    )
    return [tools]


def load_scenarios() -> list[dict]:
    """Load all scenario files from scenarios/ directory."""
    scenarios = []
    for filepath in sorted(glob(str(SCENARIOS_DIR / "*.json"))):
        with open(filepath) as f:
            scenarios.append(json.load(f))
    return scenarios


def call_gemini(
    client: genai.Client,
    system_prompt: str,
    tools: list[types.Tool],
    conversation: list[types.Content],
) -> tuple[str | None, dict, str]:
    """Call Gemini with conversation history. Returns (tool_called, tool_args, response_text)."""
    response = None
    last_error = None
    models_to_try = [MODEL] + FALLBACK_MODELS

    for model in models_to_try:
        for attempt in range(MAX_RETRIES):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=conversation,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        tools=tools,
                        temperature=0.1,
                    ),
                )
                break
            except Exception as e:
                last_error = e
                if "429" in str(e) and attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))
                    continue
                else:
                    break
        if response is not None:
            break

    if response is None:
        raise RuntimeError(f"All models failed: {last_error}")

    tool_called = None
    tool_args: dict = {}
    response_text = ""

    if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
        for part in response.candidates[0].content.parts:
            if part.function_call:
                tool_called = part.function_call.name
                tool_args = dict(part.function_call.args or {})
            if part.text:
                response_text += part.text

    return tool_called, tool_args, response_text


def check_turn(
    tool_called: str | None,
    response_text: str,
    expect: dict,
    tool_args: dict | None = None,
) -> tuple[bool, str]:
    """Check a single turn against expectations. Returns (passed, reason)."""
    expected_tool = expect.get("tool_called")
    expected_tool_alt = expect.get("tool_called_alt")

    # Tool check
    if expected_tool is None and expected_tool_alt is None:
        # Expect NO tool call
        if tool_called is not None:
            return False, f"Expected no tool call, got '{tool_called}'"
    elif expected_tool is None and expected_tool_alt is not None:
        # Either no tool or alt tool is acceptable
        if tool_called is not None and tool_called != expected_tool_alt:
            return False, f"Expected no tool or '{expected_tool_alt}', got '{tool_called}'"
    else:
        acceptable = [expected_tool]
        if expected_tool_alt is not None:
            acceptable.append(expected_tool_alt)
        if tool_called not in acceptable:
            return False, f"Expected tool '{expected_tool}' (or alt), got '{tool_called}'"

    # Tool argument check — assert a called tool's args contain expected substrings.
    # e.g. {"flow_id": "kitob"} verifies start_flow targeted the book flow, not just
    # any flow. Only enforced when the expectation declares tool_args_contains.
    expected_args = expect.get("tool_args_contains")
    if expected_args:
        args = tool_args or {}
        for key, substr in expected_args.items():
            actual = str(args.get(key, ""))
            if substr.lower() not in actual.lower():
                return False, f"Tool arg '{key}' missing '{substr}' (got '{actual}')"

    # Response keyword check
    keywords = expect.get("response_keywords", [])
    if keywords and response_text:
        text_lower = response_text.lower()
        if not any(kw.lower() in text_lower for kw in keywords):
            return False, f"Response missing keywords: {keywords}"

    # Fail keyword check
    fail_keywords = expect.get("fail_keywords", [])
    if fail_keywords and response_text:
        text_lower = response_text.lower()
        for fk in fail_keywords:
            if fk.lower() in text_lower:
                return False, f"Response contains fail keyword: '{fk}'"

    return True, ""


def run_scenario(client: genai.Client, system_prompt: str, tools: list[types.Tool], scenario: dict) -> dict:
    """Run a single scenario (potentially multi-turn)."""
    scenario_id = scenario["id"]
    category = scenario.get("category", "unknown")
    turns = scenario.get("turns", [])

    turn_results = []
    conversation = []
    all_passed = True
    fail_reason = ""

    for turn_idx, turn in enumerate(turns):
        user_msg = turn["user_message"]
        expect = turn["expect"]

        # Add user message to conversation
        conversation.append(
            types.Content(role="user", parts=[types.Part(text=user_msg)])
        )

        try:
            tool_called, tool_args, response_text = call_gemini(
                client, system_prompt, tools, conversation
            )
        except RuntimeError as e:
            return {
                "id": scenario_id,
                "category": category,
                "passed": False,
                "reason": str(e),
                "turns": turn_results,
            }

        # Add assistant response to conversation for context in next turn
        if response_text:
            conversation.append(
                types.Content(role="model", parts=[types.Part(text=response_text)])
            )

        passed, reason = check_turn(tool_called, response_text, expect, tool_args)

        turn_results.append({
            "turn": turn_idx,
            "user_message": user_msg,
            "tool_called": tool_called,
            "tool_args": tool_args or None,
            "response_text": response_text[:200] if response_text else None,
            "passed": passed,
            "reason": reason,
        })

        if not passed:
            all_passed = False
            fail_reason = f"Turn {turn_idx}: {reason}"
            break  # Stop on first failure in multi-turn

        # Small delay between turns
        if turn_idx < len(turns) - 1:
            time.sleep(1)

    return {
        "id": scenario_id,
        "category": category,
        "passed": all_passed,
        "reason": fail_reason,
        "turns": turn_results,
    }


def generate_report(results: list[dict], model: str) -> dict:
    """Generate a structured JSON report."""
    total = len(results)
    passed = sum(1 for r in results if r["passed"])

    # Category breakdown
    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"total": 0, "passed": 0, "failed": []}
        categories[cat]["total"] += 1
        if r["passed"]:
            categories[cat]["passed"] += 1
        else:
            categories[cat]["failed"].append(r["id"])

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "summary": {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": round(passed / total, 2) if total > 0 else 0,
        },
        "categories": categories,
        "results": results,
    }


def main() -> int:
    """Run all eval scenarios and report results."""
    client = genai.Client(
        vertexai=True,
        project=GCP_PROJECT,
        location=GCP_LOCATION,
    )

    print("Loading system prompt from config...")
    system_prompt = load_system_prompt()
    tools = get_tool_declarations()
    scenarios = load_scenarios()

    if not scenarios:
        print("ERROR: No scenario files found in eval/scenarios/")
        return 1

    print(f"Running {len(scenarios)} eval scenarios against {MODEL}...\n")

    results = []

    for i, scenario in enumerate(scenarios):
        if i > 0:
            time.sleep(SCENARIO_DELAY)
        result = run_scenario(client, system_prompt, tools, scenario)
        results.append(result)

        status = "✅ PASS" if result["passed"] else "❌ FAIL"
        print(f"  {status}  {result['id']} [{result['category']}]")
        if not result["passed"]:
            print(f"         Reason: {result['reason']}")
        for tr in result.get("turns", []):
            if tr.get("tool_called"):
                print(f"         Turn {tr['turn']} tool: {tr['tool_called']}")

    # Summary
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    pass_rate = passed / total if total > 0 else 0

    print(f"\n{'='*50}")
    print(f"Results: {passed}/{total} passed ({pass_rate:.0%})")
    print(f"Minimum required: {MIN_PASS_RATE:.0%}")

    # Category breakdown
    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"passed": 0, "total": 0}
        categories[cat]["total"] += 1
        if r["passed"]:
            categories[cat]["passed"] += 1

    print(f"\nBy category:")
    for cat, stats in sorted(categories.items()):
        cat_rate = stats["passed"] / stats["total"]
        icon = "✅" if cat_rate >= MIN_PASS_RATE else "⚠️"
        print(f"  {icon} {cat}: {stats['passed']}/{stats['total']}")

    # Generate and save report
    report = generate_report(results, MODEL)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "eval-report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved: {report_path}")

    if pass_rate >= MIN_PASS_RATE:
        print("🎉 Eval PASSED — above minimum threshold")
        return 0
    else:
        print(f"❌ Eval FAILED — below {MIN_PASS_RATE:.0%} threshold")
        return 1


if __name__ == "__main__":
    sys.exit(main())
