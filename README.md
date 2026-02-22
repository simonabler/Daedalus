<!-- markdownlint-disable MD030 -->
<h1 align="center">Daedalus</h1>

<p align="center">
<img src="https://github.com/simonabler/daedalus/blob/main/images/daedalus.png?raw=true">
</p>

<div align="center">

[![Release Notes](https://img.shields.io/github/release/simonabler/Daedalus)](https://github.com/simonabler/Daedalus/releases)
[![GitHub fork](https://img.shields.io/github/forks/simonabler/Daedalus?style=social)](https://github.com/simonabler/Daedalus/fork)

</div>

A production-ready, multi-agent AI coding system that autonomously plans, implements, tests, and commits code changes with intelligent context awareness, human approval gates, and full checkpoint/resume capabilities.

## ✨ What's New in v2.0

🎯 **Intelligent Intent Routing** - Distinguishes between coding tasks, status queries, and research requests  
🔍 **Repository Context Awareness** - Analyzes tech stack, test frameworks, and code conventions before coding  
✅ **Human Approval Gates** - Requires approval before commits and risky operations  
🔄 **Checkpoint & Resume** - Never lose progress; resume from any interruption  
🔎 **Safe Code Search** - Search repository patterns without shell execution  
📚 **Context-Aware Agents** - Planner and coders know your project structure and conventions  

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     USER INTERFACES                         │
│                Telegram Bot  ·  Web Chat                    │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                INTELLIGENT ORCHESTRATOR                     │
│                                                             │
│  ┌──────────┐  ┌───────────────┐  ┌──────────┐             │
│  │ Router   │→ │ Context Loader│→ │ Planner  │             │
│  │(Intent)  │  │(Repo Analysis)│  │(GPT-4o-m)│             │
│  └──────────┘  └───────────────┘  └──────────┘             │
│       ├─ status → Status Node                              │
│       ├─ research → Research Node (read-only)              │
│       ├─ resume → Resume from Checkpoint                   │
│       └─ code → (workflow below)                           │
│                                                             │
│  ┌──────────┐  ┌─────────┐  ┌───────────┐  ┌────────────┐ │
│  │ Planner  │→ │Coder 1  │→ │Peer Review│→ │  Planner   │ │
│  │(GPT-4o-m)│  │(configu-│  │by Coder 2 │  │  Review    │ │
│  │          │  │rable)   │  │           │  │ (final ok) │ │
│  └──────────┘  └─────────┘  └───────────┘  └─────┬──────┘ │
│       │                                          │        │
│       │       ┌─────────┐  ┌───────────┐         │        │
│       │       │Coder 2  │→ │Peer Review│─────────┘        │
│       │       │(configu-│  │by Coder 1 │                  │
│       │       │ rable)  │  │           │                  │
│       │       └─────────┘  └───────────┘                  │
│       │                                                   │
│  ┌────▼─────┐  ┌─────────────┐  ┌──────────────────────┐ │
│  │ Tester   │→ │ Human Gate  │→ │ Commit & Checkpoint  │ │
│  │(Tools+LLM│  │(Approval)   │  │                      │ │
│  └──────────┘  └─────────────┘  └──────────────────────┘ │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                     SAFE TOOLS                              │
│  Filesystem (sandboxed) · Shell (blocklist) · Git (curated) │
│  Search (no shell) · Build/Test · Context Analysis          │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Key Features

### 🎯 Intelligent Request Routing

Daedalus now understands different types of requests:

- **Code Tasks** - Full workflow with planning, implementation, testing, and approval
- **Status Queries** - Quick answers without modifying files ("What's the status?")
- **Research Requests** - Read-only analysis ("Find all uses of X in the codebase")
- **Resume Commands** - Continue from last checkpoint after interruption

The router automatically classifies your intent and routes to the appropriate handler.

### 🔍 Repository Context Awareness

Before planning any code changes, Daedalus analyzes your repository:

- **Tech Stack Detection** - Language, framework, package manager
- **Test Framework Discovery** - pytest, jest, unittest, etc. with correct commands
- **Code Conventions** - Linting tools, formatting rules, line length limits
- **CI/CD Configuration** - GitHub Actions, GitLab CI, Jenkins awareness
- **Project Structure** - Entry points, architecture patterns, dependencies

This context is injected into all agent prompts, ensuring they use the correct tools and follow your conventions.

### ✅ Human Approval Gates

Never auto-commit without review. Daedalus pauses before:

- **Every commit** - Always requires approval
- **Large changes** - Diffs over 400 lines
- **File deletions** - Any file removal operation
- **CI/CD changes** - Modifications to workflow configs

You receive a diff preview and can approve or reject before any changes are committed.

### 🔄 Checkpoint & Resume

Full crash recovery and task resumption:

- Checkpoints saved after planning, coding, testing, and commits
- Resume from last checkpoint with a simple "continue" command
- Handles interruptions, crashes, and manual stops gracefully
- State stored in `.daedalus/checkpoints/` directory

### 🔎 Safe Code Search

Search your codebase without shell commands:

```python
# Agents can use: search_in_repo("pattern", "*.py", max_hits=50)
# Returns: path:line: matched_text
```

- Pure Python implementation (no shell execution)
- Skips .git, node_modules, build artifacts automatically
- Configurable file patterns and result limits

### Dual-Coder Peer Review Workflow

The system uses **two coders** that alternate and cross-review each other:

- **Even-numbered items** (0, 2, 4…): Coder 1 implements → Coder 2 reviews
- **Odd-numbered items** (1, 3, 5…): Coder 2 implements → Coder 1 reviews

Both coders are fully configurable via `.env` — they can run on OpenAI, Anthropic, or local Ollama models independently.

```
Router (classify intent)
    → Context Loader (analyze repo)
        → Planner (create plan with context)
            → Coder (A or B, alternating)
                → Peer Review (by the OTHER coder)
                    → Planner Final Review
                        → Tester
                            → Human Gate (approval required)
                                → Commit & Checkpoint
                                    → next item (alternate coder) or DONE

On REWORK (peer review or planner review): → back to the original coder
On TEST FAIL: → back to the original coder
On unexpected error: → STOP and re-plan
```

### Agent Roles

| Role | Model | Responsibility |
|------|-------|---------------|
| **Router** | GPT-4o-mini | Classifies intent (code, status, research, resume) |
| **Context Loader** | Filesystem + LLM | Analyzes repository structure and conventions |
| **Planner** | GPT-4o-mini | Understands goals, creates context-aware plans, final review gate |
| **Coder 1** | Configurable (see `.env`) | Implements even-numbered items, peer-reviews Coder 2's work |
| **Coder 2** | Configurable (see `.env`) | Implements odd-numbered items, peer-reviews Coder 1's work |
| **Tester** | GPT-4o-mini + tools | Runs tests/linters/builds with detected commands |
| **Human Gate** | Interactive | Approval checkpoint before commits |

## Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/simonabler/Daedalus.git daedalus
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
- `TARGET_REPO_PATH` — path to the Git repository the agents will work on
- API keys only for the providers you use: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`
- `CODER_1_MODEL` / `CODER_2_MODEL` — model strings determine the provider automatically

Model string format:
- OpenAI: `gpt-4o`, `gpt-4o-mini`, …
- Anthropic: `claude-opus-4-5`, `claude-sonnet-4-20250514`, …
- Ollama (local): `ollama:llama3.1:70b`, `ollama:deepseek-coder-v2`, …

### 3. Run

```bash
python -m app.main
```

This starts:
- **Web UI** at `http://127.0.0.1:8420` (configurable via `WEB_HOST`/`WEB_PORT`)
- **Telegram bot** (if `TELEGRAM_BOT_TOKEN` is set)
- **Background task processor**

### 4. Submit Tasks

#### Via Web UI
Open `http://127.0.0.1:8420` and type a task in the chat.

#### Via Telegram
```
/task Add user authentication with JWT tokens
/status
/logs
/stop
```

#### Via API

**Code Task (Full Workflow):**
```bash
curl -X POST http://127.0.0.1:8420/api/task \
  -H "Content-Type: application/json" \
  -d '{"task": "Add health check endpoint to /api/health"}'
```

**Status Query (Quick Answer):**
```bash
curl -X POST http://127.0.0.1:8420/api/task \
  -H "Content-Type: application/json" \
  -d '{"task": "What is the current status?"}'
```

**Research Request (Read-Only):**
```bash
curl -X POST http://127.0.0.1:8420/api/task \
  -H "Content-Type: application/json" \
  -d '{"task": "Find all imports of GraphState in the codebase"}'
```

**Resume After Interruption:**
```bash
curl -X POST http://127.0.0.1:8420/api/task \
  -H "Content-Type: application/json" \
  -d '{"task": "continue"}'
```

### 5. Approve Changes

When the workflow pauses for approval:

```bash
# Check pending approval
curl http://127.0.0.1:8420/api/status

# Approve and continue
curl -X POST http://127.0.0.1:8420/api/approve \
  -H "Content-Type: application/json" \
  -d '{"approved": true}'

# Reject and stop
curl -X POST http://127.0.0.1:8420/api/approve \
  -H "Content-Type: application/json" \
  -d '{"approved": false}'
```

## 📁 Project Structure

```
daedalus/
├── AGENT.md                 # Instructions for AI agents working on this repo
├── app/
│   ├── core/                # Domain logic
│   │   ├── config.py        # Settings (pydantic-settings, .env)
│   │   ├── logging.py       # Centralized logging
│   │   ├── state.py         # GraphState, TodoItem, enums
│   │   ├── nodes.py         # LangGraph node implementations
│   │   ├── orchestrator.py  # Graph builder and runner
│   │   ├── repo_context.py  # Repository context data models
│   │   ├── checkpoints.py   # Checkpoint save/load management
│   │   ├── memory.py        # Shared agent memory system
│   │   └── task_routing.py  # Intent classification helpers
│   ├── agents/              # Agent definitions
│   │   ├── models.py        # LLM factory (role → provider)
│   │   ├── analyzer.py      # CodebaseAnalyzer for context
│   │   └── prompts/         # System prompts per role
│   ├── tools/               # Safe LangChain tools
│   │   ├── filesystem.py    # Sandboxed file I/O
│   │   ├── shell.py         # Blocklist-protected shell
│   │   ├── git.py           # Allow/block git operations
│   │   ├── search.py        # Safe repository search
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
├── memory/                  # Shared agent memory
│   ├── architecture-decisions.md
│   ├── coding-style.md
│   └── shared-insights.md
├── docs/
│   └── definition-of-done.md
├── tests/                   # Pytest test suite
│   ├── test_router.py       # Intent routing tests
│   ├── test_context_loader.py
│   ├── test_search.py
│   ├── test_human_gate.py
│   ├── test_checkpoints.py
│   └── ...
├── logs/                    # Rotating log files
├── .env.example
├── pyproject.toml
├── CHANGELOG.md
└── README.md
```

## 🔒 Safety

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
- **Allowed**: `status`, `diff`, `add`, `commit`, `checkout`, `push`, `pull`, `fetch`, `log`, `branch`, `show`, `stash`, `tag`
- **Blocked**: `merge`, `rebase`, `reset --hard`, `clean -fd`, `push --force`
- **Human approval required** before all commits
- Feature branches only — merge is forbidden (humans merge)

### Iteration Limits
- Max iterations per TODO item: configurable (`MAX_ITERATIONS_PER_ITEM`, default 5)
- Exceeding the limit → workflow stops and asks user for input

### Human Approval Gates
- **Always triggers**: Before every commit
- **Also triggers**: Large diffs (>400 lines), file deletions, CI/CD config changes
- Provides diff preview, file list, and change summary
- User can approve or reject via API

## 🔄 Workflow Examples

### Example 1: Code Task with Full Context

```bash
# Submit task
POST /api/task
{"task": "Add rate limiting to API endpoints"}
```

**What Happens:**
1. **Router** classifies as "code" intent
2. **Context Loader** analyzes repository:
   - Detects FastAPI framework
   - Finds pytest as test framework
   - Extracts ruff + black for linting/formatting
   - Notes GitHub Actions CI/CD
3. **Planner** creates plan knowing:
   - Use FastAPI middleware patterns
   - Test with pytest
   - Follow black formatting (88 char lines)
   - Don't break GitHub Actions
4. **Coder** implements using detected conventions
5. **Tester** runs `pytest` (correct command from context!)
6. **Human Gate** pauses with diff preview
7. **You approve** → Commits with conventional commit message
8. **Checkpoint saved** for future resume

### Example 2: Status Query

```bash
POST /api/task
{"task": "What's the current status?"}
```

**What Happens:**
1. **Router** classifies as "status" intent
2. **Status Node** returns immediate answer:
   - Current phase
   - Todo items (total, done, in progress)
   - Last commit
   - No files modified ✓

### Example 3: Research Request

```bash
POST /api/task
{"task": "Find all uses of 'search_in_repo' in the codebase"}
```

**What Happens:**
1. **Router** classifies as "research" intent
2. **Research Node** uses `search_in_repo` tool:
   - Searches Python files
   - Returns matches with line numbers
   - No files modified ✓
   - Read-only operation ✓

### Example 4: Resume After Crash

```bash
# Daedalus crashes while implementing item 3 of 5
# ... restart Daedalus ...

POST /api/task
{"task": "continue"}
```

**What Happens:**
1. **Router** classifies as "resume" intent
2. **Resume Node** loads `.daedalus/checkpoints/latest.json`:
   - Restores plan (5 items)
   - Restores current position (item 3)
   - Restores repo context
3. **Workflow continues** from item 3
4. **No progress lost** ✓

## 🔧 Git Workflow

1. Agent creates a feature branch: `feature/<date>-<slug>`
2. Works through TODO items one at a time
3. Each completed item:
   - Tests pass
   - Human approval obtained
   - Conventional Commit created
   - Checkpoint saved
   - Branch pushed
4. When all items done → "Ready for PR/Merge" status
5. **Human** creates PR and merges

## 📋 Task Management

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
- Date: 2026-02-21
- Mistake: Overwrote existing test file
- Rule: Read existing tests before writing new ones
- Enforcement: Coder must list test files before editing
```

### .daedalus/checkpoints/
```
.daedalus/
└── checkpoints/
    ├── latest.json                    # Most recent state
    ├── plan_complete_abc123.json      # After planning
    ├── code_complete_def456.json      # After coding
    └── test_pass_ghi789.json          # After tests pass
```

## 📡 API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/task` | Submit a new task (code, status, research, resume) |
| POST | `/api/approve` | Approve or reject pending commit |
| GET | `/api/status` | Current workflow status + pending approvals |
| GET | `/api/logs` | Recent log entries |
| WS | `/ws` | Real-time status + log stream |
| GET | `/` | Web UI |

### /api/task Request Body

```json
{
  "task": "Add health check endpoint"
}
```

**Intent Classification:**
- Contains "add", "fix", "implement" → **code** (full workflow)
- Contains "status", "what's", "show" → **status** (quick answer)
- Contains "find", "search", "analyze" → **research** (read-only)
- Contains "continue", "resume" → **resume** (load checkpoint)

### /api/approve Request Body

```json
{
  "approved": true
}
```

### /api/status Response

```json
{
  "phase": "waiting_for_approval",
  "needs_human_approval": true,
  "pending_approval": {
    "type": "commit",
    "summary": "3 files changed, 45 insertions(+), 12 deletions(-)",
    "files": ["app/core/nodes.py", "tests/test_nodes.py", "README.md"],
    "diff_preview": "diff --git a/app/core/nodes.py ...",
    "triggers": [
      {"type": "commit", "reason": "Commit requires approval"},
      {"type": "large_diff", "reason": "Large diff: 57 lines changed"}
    ]
  },
  "todo_items": [...],
  "current_item_index": 2
}
```

## 🧪 Running Tests

```bash
# All tests
pytest

# Verbose output
pytest -v

# Specific test file
pytest tests/test_router.py

# With coverage
pytest --cov=app tests/

# New feature tests
pytest tests/test_router.py tests/test_context_loader.py tests/test_search.py tests/test_human_gate.py tests/test_checkpoints.py
```

## ⚙️ Configuration Reference

See `.env.example` for all available settings. Key options:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | OpenAI API key (required if any model uses OpenAI) |
| `ANTHROPIC_API_KEY` | — | Anthropic API key (required if any model uses Anthropic) |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL (required if any model uses Ollama) |
| `TARGET_REPO_PATH` | (required) | Path to target Git repo |
| `CODER_1_MODEL` | `gpt-4o-mini` | Model for Coder 1 — prefix `ollama:` for local models |
| `CODER_2_MODEL` | `gpt-4o-mini` | Model for Coder 2 — prefix `ollama:` for local models |
| `PLANNER_MODEL` | `gpt-4o-mini` | Model for Planner |
| `TESTER_MODEL` | `gpt-4o-mini` | Model for Tester |
| `DOCUMENTER_MODEL` | `gpt-4o-mini` | Model for Documenter |
| `TELEGRAM_BOT_TOKEN` | — | Telegram bot token (optional) |
| `TELEGRAM_ALLOWED_USER_IDS` | — | Comma-separated user IDs (optional) |
| `WEB_HOST` | `127.0.0.1` | Web UI host |
| `WEB_PORT` | `8420` | Web UI port |
| `MAX_ITERATIONS_PER_ITEM` | `5` | Max rework attempts per item |
| `SHELL_TIMEOUT_SECONDS` | `120` | Shell command timeout |
| `LOG_LEVEL` | `INFO` | Logging level |

## 🆕 What Changed in v2.0

### New Features
- ✨ **Intent Routing** - Automatically classifies and routes different request types
- ✨ **Repository Context** - Analyzes tech stack, tests, conventions before coding
- ✨ **Human Approval** - Required before all commits and risky operations
- ✨ **Checkpointing** - Save/resume state at any point
- ✨ **Safe Search** - Search codebase without shell commands
- ✨ **Context-Aware Agents** - All agents know project structure and conventions

### New Components
- `app/core/repo_context.py` - Repository context data models
- `app/core/checkpoints.py` - Checkpoint management
- `app/agents/analyzer.py` - Codebase analyzer
- `app/tools/search.py` - Safe search tool
- Router, Context Loader, Human Gate nodes

### Enhanced Components
- `app/core/state.py` - 17 new fields for context and approval
- `app/core/nodes.py` - Context injection in all prompts
- `app/core/orchestrator.py` - Entry point changed to router
- `app/web/server.py` - New `/api/approve` endpoint

### New Dependencies
- `tomli>=2.0.0` - TOML parsing for config files
- `pyyaml>=6.0` - YAML parsing for CI/CD configs

### Breaking Changes
**None!** All changes are backward compatible. Existing workflows continue to work.

## 📚 Documentation

- **AGENT.md** - Instructions for AI agents working on this codebase
- **CHANGELOG.md** - Detailed version history
- **CLAUDE.md** - Claude-specific guidance
- **docs/definition-of-done.md** - Acceptance criteria for tasks

## 🤝 Contributing

See [AGENT.md](AGENT.md) for development guidelines and workflow.

## 📄 License

MIT

---

<div align="center">

**Daedalus v2.0** - Production-ready AI coding with intelligence, safety, and reliability.

[Report Bug](https://github.com/simonabler/Daedalus/issues) · [Request Feature](https://github.com/simonabler/Daedalus/issues)

</div>

