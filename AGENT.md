# AGENT.md — Working on Daedalus

> **Scope:** This file is read by AI agents that work on the Daedalus codebase itself.
> It is intentionally NOT read when Daedalus operates on external target repositories —
> in that case the target repo's own `AGENT.md` applies.
>
> **Self-improvement mode:** Set `TARGET_REPO_PATH` to a **separate clone** of Daedalus.
> Never point it at this working copy.

---

## What is Daedalus?

A local, autonomous multi-agent coding system. It receives a task, plans it, implements
it with two alternating LLM coders (configurable provider), cross-reviews, tests, and commits —
all with human approval at critical gates.

Daedalus is a **tool**, not a library. It operates on OTHER repositories via `TARGET_REPO_PATH`.

---

## Architecture at a Glance

```
User Input
    ↓
router_node          — classifies intent: code / status / research / resume
    ↓
context_loader_node  — reads target repo's AGENT.md, README, detects tech stack
    ↓
planner_plan_node    — breaks task into TodoItems with acceptance criteria
    ↓
┌── coder_node ──────────────────────────────────────────────┐
│   Coder 1  — even items                               │
│   Coder 2  — odd items                                │
└────────────────────────────────────────────────────────────┘
    ↓
peer_review_node     — OTHER coder reviews (cross-model)
    ↓
learn_from_review_node — extracts insights into memory/
    ↓
planner_review_node  — Planner final gate: APPROVE or REWORK
    ↓
tester_node          — runs tests, linter, verifies acceptance criteria
    ↓
planner_decide_node  — PASS → continue, FAIL → rework
    ↓
human_gate_node      — pauses for human approval before commit
    ↓
committer_node       — git commit + push, checkpoint saved
    ↓
next item or DONE
```

**Key files:**
- `app/core/nodes.py` — all node implementations (~1800 lines)
- `app/core/orchestrator.py` — LangGraph graph definition and routing
- `app/core/state.py` — `GraphState`, `TodoItem`, `WorkflowPhase`, `ItemStatus`
- `app/agents/models.py` — `get_llm(role)` factory, `load_system_prompt(role)`
- `app/agents/prompts/` — system prompts per agent role
- `app/tools/` — sandboxed tools (filesystem, git, shell, search, build)
- `app/core/memory.py` — shared long-term memory between coders

---

## Before You Touch Anything

```bash
pytest                    # All tests must pass before and after your change
ruff check app/ tests/    # No new lint warnings
```

Read these before modifying behaviour:
1. The relevant node in `nodes.py`
2. `app/core/orchestrator.py` — routing logic and graph edges
3. `app/core/state.py` — what fields exist in `GraphState`

---

## Code Conventions

| Topic | Rule |
|-------|------|
| Python version | 3.11+ |
| Line length | 120 chars (ruff configured) |
| Imports | `from __future__ import annotations` at top of every module |
| Type hints | Always. Use `X | None` not `Optional[X]` |
| Logging | `get_logger(__name__)` from `app.core.logging` — never `print()` |
| Naming | `snake_case` functions/vars, `PascalCase` classes, `UPPER_SNAKE` constants |
| Error handling | Tools return error strings, never raise to the LLM. Nodes use `try/except`. |

---

## Node Conventions

Every node must follow this signature and pattern:

```python
def my_node(state: GraphState) -> dict:
    emit_node_start("role", "Node Name", ...)
    # ... work ...
    emit_node_end("role", "Node Name", "result summary")
    return {"field": value}  # partial state update, never mutate state in-place
```

- Emit events via `app.core.events` — the web UI depends on them
- Call LLMs via `_invoke_agent(role, messages, tools, inject_memory=True/False)`
- Never call LLM providers directly

---

## Tool Conventions

All tools live in `app/tools/` and are `@tool`-decorated LangChain functions.

- **Filesystem:** All file access goes through `filesystem.py` — sandboxed to `TARGET_REPO_PATH`
- **Shell:** All commands go through `shell.py` — blocklist + timeout enforced
- **Git:** Only allowed operations (no force-push, no reset --hard, no merge)
- **Search:** `search_in_repo` in `search.py` — no shell grep, pure Python

Tools must return strings. Never raise exceptions to the LLM.

---

## Agent Prompts

System prompts live in `app/agents/prompts/`:

| File | Role | Model |
|------|------|-------|
| `router.txt` | Intent classification | planner model |
| `supervisor_planner.txt` | Planner / project manager | GPT-4o-mini |
| `coder_a.txt` | Coder 1 | Configurable |
| `coder_b.txt` | Coder 2 | Configurable |
| `peer_reviewer_a.txt` | Reviewer 1 (reviews Coder 2's work) | Configurable |
| `peer_reviewer_b.txt` | Reviewer 2 (reviews Coder 1's work) | Configurable |
| `tester.txt` | Test agent | GPT-4o-mini |
| `documenter.txt` | Documentation agent | configurable |

When changing agent behaviour, update the prompt file — not just the node code.

---

## Shared Memory

Three files in `memory/` are injected into every coder and reviewer via `inject_memory=True`:

- `memory/coding-style.md` — naming, patterns, error handling
- `memory/architecture-decisions.md` — ADRs (why decisions were made)
- `memory/shared-insights.md` — codebase quirks, gotchas, useful helpers

If you discover something reusable, add it to the appropriate memory file.
The `learn_from_review_node` does this automatically after each peer review.

---

## Testing

- Tests in `tests/test_*.py`, run with `pytest`
- Mock `_invoke_agent` for node tests — never call real LLMs in tests
- Use `monkeypatch` for settings and `tmp_path` for filesystem tests
- Every new node or tool change needs at least one test

---

## Context Loader — Important Rule

The `context_loader_node` reads documentation from `TARGET_REPO_PATH`.
It deliberately **skips** `AGENT.md` / `AGENTS.md` when `TARGET_REPO_PATH`
resolves to the Daedalus root itself — this file is for contributors, not task instructions.

**Self-improvement mode:** set `TARGET_REPO_PATH` to a separate Daedalus clone.
The clone's own `AGENT.md` will then be read as normal task instructions.

---

## Git Rules

- Feature branches only: `feature/<date>-<slug>` — never commit to `main`
- Conventional Commits: `feat(scope): ...`, `fix(scope): ...`, `docs: ...`, `test: ...`, `refactor: ...`, `chore: ...`
- No merge, rebase, reset --hard, force-push, or clean -fd (blocked)
- Human creates the PR and merges

---

## Definition of Done

A task is done only when ALL of these are true:

- [ ] All existing tests pass (`pytest`)
- [ ] New tests added for the change
- [ ] Linter clean (`ruff check app/ tests/`)
- [ ] Planner reviewed and approved the diff
- [ ] Tester verified acceptance criteria with evidence
- [ ] Committed with Conventional Commit message on a feature branch
- [ ] `tasks/todo.md` updated
- [ ] `CHANGELOG.md` updated if user-facing
- [ ] `tasks/lessons.md` updated if a mistake was corrected

---

## Current State (v2.0)

All core workflow components are implemented and tested:

- ✅ Router node with `router.txt` system prompt
- ✅ Context loader with self-referential AGENT.md protection
- ✅ Dual-coder system (Coder 1 / Coder 2 alternating, provider-agnostic)
- ✅ Cross-model peer review with shared memory
- ✅ Learn-from-review node (auto-extracts insights)
- ✅ Human gate before every commit
- ✅ Checkpoint / Resume system
- ✅ Safe tools (filesystem sandbox, shell blocklist, git allow-list)
- ✅ Web UI (FastAPI + WebSocket)
- ✅ Telegram bot interface

**Open items (Phase 2):**
- GitHub integration (clone via API, create issues/PRs)
- Bug detection (static analysis, security scanning)
- `documenter.txt` — hardcoded Windows/PowerShell assumption needs replacing with runtime `execution_platform`