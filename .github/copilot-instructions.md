# GitHub Copilot Instructions

This repository's AI-assisted workflow rules live in [`AGENTS.md`](../AGENTS.md). Read it at the start of any non-trivial session.

Most-relevant sections:
- **Coding Style & Naming Conventions** — Python 3.11, Black/Ruff with line-length 100, snake_case modules, PascalCase classes.
- **Testing Guidelines** — `pytest` with `pytest-asyncio`; unit tests in `tests/unit/`, integration in `tests/integration/`.
- **Creating Pull Requests (AI-assisted)** — branch naming, commit prefixes, PR body template, reviewer list, Linear ticket handling.

Per-repo PR config (base branch, default reviewers, PR body template) is in [`.claude/pr-rules.yaml`](../.claude/pr-rules.yaml). It's YAML, self-documenting; use it as the authoritative source when `AGENTS.md` and it disagree.

Deep project architecture and module-by-module guidance for Claude Code lives in [`CLAUDE.md`](../CLAUDE.md); Copilot can skim it for the same context.
