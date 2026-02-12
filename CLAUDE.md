# CLAUDE.md — Project Context for Claude Code

## What is this project?

**Daedalus** is a local multi-agent AI coding system. It receives a coding task, breaks it into subtasks, implements them with two alternating LLM coders, peer-reviews across models, tests, and commits — all autonomously.

It is NOT a library. It is a standalone tool that operates on OTHER repositories.

## Tech Stack

- **Python 3.11+** — no older versions
- **LangGraph** — orchestrates the multi-agent workflow as a state machine
- **LangChain** — LLM abstraction (langchain-openai, langchain-anthropic)
- **FastAPI + WebSockets** — web UI with real-time event streaming
- **python-telegram-bot** — Telegram interface
- **Pydantic v2 + pydantic-settings** — config and data models
- **pytest + ruff** — testing and linting

## Architecture

```
app/
├── agents/          # LLM configuration + system prompts
│   ├── models.py    # get_llm(role) factory, auto-detects Anthropic vs OpenAI
│   └── prompts/     # System prompts: coder_a, coder_b, peer_reviewer_a/b, supervisor_planner, tester
├── core/            # Brain
│   ├── config.py    # Settings (from .env via pydantic-settings)
│   ├── events.py    # Event bus — nodes emit events, web server broadcasts via WebSocket
│   ├── memory.py    # Shared long-term memory (coding style, architecture, insights)
│   ├── nodes.py     # All LangGraph node implementations
│   ├── orchestrator.py  # Graph definition + routing + run_workflow()
│   ├── state.py     # GraphState, TodoItem, WorkflowPhase, ItemStatus
│   └── logging.py   # Structured logging
├── tools/           # LangChain tools (sandboxed)
│   ├── filesystem.py  # read_file, write_file, list_directory, patch_file
│   ├── git.py         # git_command, git_create_branch, git_commit_and_push, git_status
│   ├── shell.py       # run_shell (sandboxed, blocklist, timeout)
│   └── build.py       # run_tests, run_linter (auto-detects project type)
├── web/             # FastAPI server + static UI
│   ├── server.py
│   └── static/index.html
├── telegram/        # Telegram bot interface
│   └── bot.py
└── main.py          # Entry point (starts web + telegram)
```

## Workflow (the graph)

```
Planner → Coder (A|B) → Peer Review (B|A) → Learn → Planner Review → Tester → Decide → Commit
               ↑               |                         |                |
               └───────────────┘ REWORK ─────────────────┘ REWORK ───────┘ FAIL
```

- **Dual-coder**: Even items → Coder A (Claude), odd items → Coder B (GPT-5.3)
- **Cross-model peer review**: The OTHER coder reviews each implementation
- **Learn node**: Extracts reusable insights from every review into shared memory files
- **All rework loops** return to the original coder (reviewer never implements)

## Shared Memory System

Three memory files in `memory/` are read by every coder and reviewer:
- `coding-style.md` — naming, patterns, error handling conventions
- `architecture-decisions.md` — ADRs (why decisions were made)
- `shared-insights.md` — codebase quirks, gotchas, useful helpers

Memory is injected into system prompts via `inject_memory=True` in `_invoke_agent()`.
After each peer review, `learn_from_review_node` extracts new insights automatically.
Planner compresses files at session start when they exceed 8000 chars.

## Key Conventions

### Code Style
- **Line length**: 120 chars (ruff)
- **Imports**: `from __future__ import annotations` at top of every module
- **Type hints**: Always. Use `X | None` not `Optional[X]`
- **Strings**: Double quotes for user-facing, single quotes acceptable in code
- **Naming**: snake_case for functions/variables, PascalCase for classes, UPPER_SNAKE for constants
- **Logging**: Use `get_logger(__name__)` from `app.core.logging`, never bare `print()`
- **Error handling**: Return error strings from tools (never raise inside tools), use `try/except` in nodes

### LangChain Tools
- Tools are `@tool`-decorated functions in `app/tools/`
- ALL filesystem access goes through `filesystem.py` (sandboxed to repo root)
- ALL shell commands go through `shell.py` (blocklist + timeout)
- Tools return strings, never raise exceptions to the LLM

### Graph Nodes
- Every node function signature: `def xxx_node(state: GraphState) -> dict`
- Return dict of state updates (partial), never modify state in-place
- Every node emits events via `app.core.events` (emit_node_start, emit_status, emit_verdict, etc.)
- Nodes call `_invoke_agent(role, messages, tools, inject_memory=True/False)` for LLM interaction

### Testing
- Tests in `tests/test_*.py`, run with `pytest`
- Use `unittest.mock.patch` for settings and external dependencies
- Test classes group related tests: `TestXxx`
- No integration tests that call real LLMs — mock `_invoke_agent` for node tests

### Git
- Never commit to main/master — always feature branches
- Conventional Commits: `feat(scope): description`, `fix(scope): ...`, `docs:`, `test:`, `refactor:`, `chore:`
- Never merge, rebase, reset --hard, force push, or clean -fd (blocked in git tool)

## Running

```bash
pip install -e .        # Install all deps
cp .env.example .env    # Configure API keys
pytest                  # Run tests (101 tests)
ruff check app/ tests/  # Lint
python -m app.main      # Start server on :8420
```

## Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `OPENAI_API_KEY` | OpenAI API key | (required) |
| `ANTHROPIC_API_KEY` | Anthropic API key | (required) |
| `PLANNER_MODEL` | Planner LLM | `gpt-4o-mini` |
| `CODER_A_MODEL` | Coder A LLM | `claude-sonnet-4-20250514` |
| `CODER_B_MODEL` | Coder B LLM | `gpt-5.2` |
| `TESTER_MODEL` | Tester LLM | `gpt-4o-mini` |
| `TARGET_REPO_PATH` | Path to the repo to work on | (required) |
| `TELEGRAM_BOT_TOKEN` | Telegram bot token | (optional) |
| `WEB_PORT` | Web UI port | `8420` |

## When Making Changes

1. **Read the relevant node** in `nodes.py` before modifying behavior
2. **Read `orchestrator.py`** to understand graph routing
3. **Run `pytest`** after every change — all 101 tests must pass
4. **Emit events** from any new node/step via `app.core.events`
5. **Update prompts** in `app/agents/prompts/` if changing agent behavior
6. **Update memory seeds** in `memory.py` if adding new memory categories
7. **Never break the tool sandbox** — all file/shell access must stay sandboxed
8. **Keep diffs minimal** — change only what's necessary