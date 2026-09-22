# Repository Guidelines

## Project Structure & Module Organization
Core application code lives in `src/`. The main entrypoint is `src/main.py`, with domain modules split by responsibility: `agents/`, `pipeline/`, `config/`, `knowledge/`, `flows/`, `tools/`, `lifecycle/`, and `observability/`. Tenant definitions live in `configs/tenants/`; keep shared defaults in `configs/tenants/_defaults.yaml` and tenant-specific overrides in separate YAML files. Knowledge assets live under `knowledge/<tenant>/`, with flow documents in `knowledge/<tenant>/flows/`. Tests are organized into `tests/unit/` and `tests/integration/`. Utility scripts, including KB ingestion, live in `scripts/`.

- A/B variants of the same tenant live in sibling YAMLs sharing tenant identity (e.g., `<slug>-a.yaml`, `<slug>-b.yaml`); see the **A/B Experiments** subsection in `CLAUDE.md`.

## Build, Test, and Development Commands
Install dependencies with `pip install -e .[dev]`.
Run the agent locally with `PYTHONPATH=src python src/main.py start`.
Run tests with `pytest`.
Collect coverage with `pytest --cov=src`.
Format code with `black src tests scripts`.
Lint imports and style with `ruff check src tests scripts`.
Populate a Qdrant knowledge base with `PYTHONPATH=src python scripts/populate_qdrant.py --source knowledge/example-tenant/faq_source.json --collection example_tenant_faq`.

## Coding Style & Naming Conventions
Target Python 3.11 and keep formatting compatible with Black and Ruff. Use 4-space indentation and a 100-character line limit. Follow standard Python naming: `PascalCase` for classes, `snake_case` for modules, functions, variables, and test names. Keep new modules focused on a single concern and place them in the existing domain package instead of creating flat top-level files. YAML keys should mirror the Pydantic schema in `src/config/schema.py`.

## Testing Guidelines
Use `pytest` with `pytest-asyncio`; async tests are supported by the repo config. Place fast logic tests in `tests/unit/` and integration or provider-facing tests in `tests/integration/`. Name files `test_<feature>.py` and test functions `test_<behavior>`. There is no enforced coverage threshold today, but new work should include tests for config loading, tool behavior, and failure paths when practical.

## Commit & Pull Request Guidelines
Recent history follows Conventional Commit prefixes such as `feat:`, `fix:`, `refactor:`, and `chore:`. Keep commit messages imperative and scoped to one change. PRs should explain the user-visible effect, note any config or environment-variable changes, and link the relevant issue or task. For voice or tenant-behavior changes, include the affected tenant YAML, knowledge assets, and a short validation note or sample log output.

## Configuration & Security Tips
Start from `.env.example` and keep secrets in a local `.env` only. Do not commit API keys, phone numbers used for testing, or environment-specific endpoints unless they are intentional shared defaults.

## Developer Tooling
The `.mcp.json` at repo root configures two MCP servers for teammates using Claude Code or other MCP-aware agents: `livekit-docs` (LiveKit documentation search over HTTP) and `context7` (on-demand library docs). It's dev-only — no runtime code reads it. Feel free to extend it per your local setup; changes are safe to commit when they benefit the team.

## Creating Pull Requests (AI-assisted)

When you — or an AI assistant like Claude Code or Codex — are about to open a PR, follow these rules. The Claude Code skill `/create-pr` automates this; on Codex or other tools, apply them manually.

### Branch & base

- Branch from **`main`**.
- Branch name: `<github-user>/<ticket>-<short-slug>`.
- Never commit directly to `main`. Never force-push shared branches.

### Commits

- Conventional Commit prefixes: `feat:`, `fix:`, `refactor:`, `chore:`, `docs:`, `test:`.
- Subject line imperative, ≤72 chars ("add X", not "added X").
- Never use `--no-verify` to skip hooks — fix the underlying issue.

### PR

- Title: `<type>: <subject> (<ticket>)`
- Body sections in this order, emoji headers preserved:
  📝 Summary → 🎯 Key Changes → 🔗 Related Context → 🧪 Testing & Validation → 📸 Visuals (optional) → 🏁 Peer Review Checklist.
- Base branch: **`main`**.
- Reviewers: see `.claude/pr-rules.yaml` for the authoritative list (author auto-excluded). It ships empty — add your repo's reviewers.

### Issue tracker

- Include your issue-tracker identifier in the branch name and PR title so the
  tracker's GitHub integration links the PR.

### Authoritative source

Full config lives in `.claude/pr-rules.yaml`. If you change conventions, update that file (for the Claude skill) and this section (for Codex and humans) together.
