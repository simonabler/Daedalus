<!-- markdownlint-disable MD030 -->
<h1 align="center">Daedalus</h1>

<p align="center">
<img src="https://github.com/simonabler/daedalus/blob/main/images/daedalus.png?raw=true">
</p>

<div align="center">

[![Release Notes](https://img.shields.io/github/release/simonabler/Daedalus)](https://github.com/simonabler/Daedalus/releases)
[![GitHub fork](https://img.shields.io/github/forks/simonabler/Daedalus?style=social)](https://github.com/simonabler/Daedalus/fork)

</div>


A local, multi-agent AI coding system that autonomously plans, implements, tests, documents, and commits code changes to Git repositories.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     USER INTERFACES                         │
│                Telegram Bot  ·  Web Chat                    │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                LANGGRAPH ORCHESTRATOR                       │
│                                                             │
│  ┌──────────┐  ┌─────────┐  ┌───────────┐  ┌────────────┐   │
│  │ Planner  │→ │Coder A  │→ │Peer Review│→ │  Planner   │   │
│  │(GPT-4o-m)│  │(Claude) │  │by Coder B │  │  Review    │   │
│  │          │  │         │  │(gpt-5.2)  │  │ (final ok) │   │
│  └──────────┘  └─────────┘  └───────────┘  └─────┬──────┘   │
│       │                                          │          │
│       │       ┌─────────┐  ┌───────────┐         │          │
│       │       │Coder B  │→ │Peer Review│─────────┘          │
│       │       │(gpt-5.2)│  │by Coder A │                    │
│       │       │         │  │(Claude)   │                    │
│       │       └─────────┘  └───────────┘                    │
│       │                                                     │
│  ┌────▼─────┐  ┌──────────────────────┐                     │
│  │ Tester   │→ │ Decide → Commit/Push │                     │
│  │(Tools+LLM│  └──────────────────────┘                     │
│  └──────────┘                                               │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                     SAFE TOOLS                              │
│       Filesystem (sandboxed) · Shell (blocklist)            │
│       Git (allow/block list) · Build/Test runners           │
└─────────────────────────────────────────────────────────────┘
```

### Dual-Coder Peer Review Workflow

The system uses **two coders** that alternate and cross-review each other:

- **Even-numbered items** (0, 2, 4…): Coder A (Claude) implements → Coder B (gpt-5.2) reviews
- **Odd-numbered items** (1, 3, 5…): Coder B (gpt-5.2) implements → Coder A (Claude) reviews

This cross-review catches different classes of bugs — each model has different strengths and blind spots.

```
Planner Plan
    → Coder (A or B, alternating)
        → Peer Review (by the OTHER coder)
            → Planner Final Review
                → Tester
                    → Decide → Commit & Push
                        → next item (alternate coder) or DONE

On REWORK (peer review or planner review): → back to the original coder
On TEST FAIL: → back to the original coder
On unexpected error: → STOP and re-plan
```

### Agent Roles

| Role | Model | Responsibility |
|------|-------|---------------|
| **Planner** | GPT-4o-mini | Understands goals, creates plans, final review gate, manages tasks |
| **Coder A** | Claude (Anthropic) | Implements even-numbered items, peer-reviews Coder B's work |
| **Coder B** | gpt-5.2 (OpenAI) | Implements odd-numbered items, peer-reviews Coder A's work |
| **Tester** | GPT-4o-mini + tools | Runs tests/linters/builds, verifies acceptance criteria |

## Quick Start

### 1. Clone and Install

```bash
git clone <repo-url> daedalus
cd daedalus
pip install -e .
```

Or with uv:
```bash
uv pip install -e .
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env with your API keys and settings
```

Required settings:
- `OPENAI_API_KEY` — for the Planner agent (GPT-4o-mini)
- `ANTHROPIC_API_KEY` — for the Coder agent (Claude)
- `TARGET_REPO_PATH` — path to the Git repository the agents will work on

Optional:
- `TELEGRAM_BOT_TOKEN` — enables Telegram bot interface
- `TELEGRAM_ALLOWED_USER_IDS` — restrict who can use the bot

### 3. Run

```bash
python -m app.main
```

This starts:
- **Web UI** at `http://127.0.0.1:8420` (configurable via `WEB_HOST`/`WEB_PORT`)
- **Telegram bot** (if `TELEGRAM_BOT_TOKEN` is set)
- **Background task processor**

### 4. Submit a Task

**Via Web UI**: Open `http://127.0.0.1:8420` and type a task in the chat.

**Via Telegram**:
```
/task Add user authentication with JWT tokens
/status
/logs
/stop
```

**Via API**:
```bash
curl -X POST http://127.0.0.1:8420/api/task \
  -H "Content-Type: application/json" \
  -d '{"task": "Add user authentication with JWT tokens"}'
```

## Project Structure

```
daedalus/
├── app/
│   ├── core/                # Domain logic
│   │   ├── config.py        # Settings (pydantic-settings, .env)
│   │   ├── logging.py       # Centralized logging
│   │   ├── state.py         # GraphState, TodoItem, enums
│   │   ├── nodes.py         # LangGraph node implementations
│   │   └── orchestrator.py  # Graph builder and runner
│   ├── agents/              # Agent definitions
│   │   ├── models.py        # LLM factory (role → provider)
│   │   └── prompts/         # System prompts per role
│   ├── tools/               # Safe LangChain tools
│   │   ├── filesystem.py    # Sandboxed file I/O
│   │   ├── shell.py         # Blocklist-protected shell
│   │   ├── git.py           # Allow/block git operations
│   │   └── build.py         # Project-aware test/lint/build
│   ├── web/                 # FastAPI web server
│   │   ├── server.py        # REST + WebSocket endpoints
│   │   └── static/          # Web UI (HTML/CSS/JS)
│   ├── telegram/            # Telegram bot
│   │   └── bot.py
│   └── main.py              # Entry point
├── tasks/
│   ├── todo.md              # Active plan + progress tracking
│   └── lessons.md           # Learned rules from mistakes
├── docs/
│   └── definition-of-done.md
├── tests/                   # Pytest test suite
├── logs/                    # Rotating log files
├── .env.example
├── pyproject.toml
├── CHANGELOG.md
└── README.md
```

## Safety

### Filesystem
- All file operations sandboxed to `TARGET_REPO_PATH`
- Path traversal (`../`) blocked and validated
- Absolute paths rejected

### Shell
- Commands execute only inside repo root
- Dangerous commands blocklisted (`rm -rf /`, `sudo`, `shutdown`, `mkfs`, pipe-to-sh, etc.)
- All executions logged with command, cwd, exit code, output
- Output truncated to prevent token overflow
- Configurable timeout (`SHELL_TIMEOUT_SECONDS`)

### Git
- Allowed: `status`, `diff`, `add`, `commit`, `checkout`, `push`, `pull`, `fetch`, `log`, `branch`, `show`, `stash`, `tag`
- Blocked: `merge`, `rebase`, `reset --hard`, `clean -fd`, `push --force`
- Commits only after Planner approval + tests pass
- Feature branches only — merge is forbidden (humans merge)

### Iteration Limits
- Max iterations per TODO item: configurable (`MAX_ITERATIONS_PER_ITEM`, default 5)
- Exceeding the limit → workflow stops and asks user for input

## Git Workflow

1. Agent creates a feature branch: `feature/<date>-<slug>`
2. Works through TODO items one at a time
3. Each completed item → Conventional Commit → push
4. When all items done → "Ready for PR/Merge" status
5. **Human** creates PR and merges

## Task Management

### tasks/todo.md
```markdown
## Plan: Add Authentication
- [x] Item 1: Set up JWT library
  - AC: JWT encode/decode works
  - Verify: `pytest tests/test_auth.py`
- [ ] Item 2: Add login endpoint
  - AC: POST /login returns token
  - Verify: `pytest tests/test_api.py`
```

### tasks/lessons.md
```markdown
### Rule 1: Always check for existing tests
- Date: 2026-02-06
- Mistake: Overwrote existing test file
- Rule: Read existing tests before writing new ones
- Enforcement: Coder must list test files before editing
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/task` | Submit a new task |
| GET | `/api/status` | Current workflow status |
| GET | `/api/logs` | Recent log entries |
| WS | `/ws` | Real-time status + log stream |
| GET | `/` | Web UI |

## Running Tests

```bash
pytest
pytest -v                    # verbose
pytest tests/test_shell.py   # specific test file
```

## Configuration Reference

See `.env.example` for all available settings. Key options:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | (required) | OpenAI API key for Planner + Coder B |
| `ANTHROPIC_API_KEY` | (required) | Anthropic API key for Coder A |
| `TARGET_REPO_PATH` | (required) | Path to target Git repo |
| `CODER_A_MODEL` | `claude-sonnet-4-20250514` | Model for Coder A |
| `CODER_B_MODEL` | `gpt-5.2` | Model for Coder B |
| `TELEGRAM_BOT_TOKEN` | (optional) | Telegram bot token |
| `WEB_PORT` | 8420 | Web UI port |
| `MAX_ITERATIONS_PER_ITEM` | 5 | Max rework attempts per item |
| `SHELL_TIMEOUT_SECONDS` | 120 | Shell command timeout |
| `LOG_LEVEL` | INFO | Logging level |

## License

MIT