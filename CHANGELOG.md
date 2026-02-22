# Changelog

All notable changes to the Daedalus project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Conventional Commits](https://www.conventionalcommits.org/).

## [2.0.0] - 2026-02-21

### 🚀 Major Release: Production-Ready Codex-Like Interaction

This release transforms Daedalus from a prototype into a production-ready AI coding system with intelligent intent routing, repository context awareness, human approval gates, and full checkpoint/resume capabilities.

### Added

#### Core Features
- **Intent Router** - Intelligent classification of user inputs into: code, status, research, resume
- **Repository Context Loader** - Automatic analysis of:
  - Tech stack (language, framework, package manager)
  - Test frameworks and commands (pytest, jest, unittest, etc.)
  - Code conventions (linting tools, formatting rules, line length)
  - CI/CD configuration (GitHub Actions, GitLab CI, Jenkins)
  - Project structure and dependencies
- **Human Approval Gates** - Required approval before:
  - All commits (always)
  - Large diffs (>400 lines)
  - File deletions
  - CI/CD configuration changes
- **Checkpoint & Resume System** - Full crash recovery with:
  - Automatic state saving after plan, code, test, commit
  - Resume from last checkpoint with "continue" command
  - State persistence in `.daedalus/checkpoints/` directory
- **Safe Search Tool** - Repository search without shell execution:
  - Pure Python implementation
  - Pattern matching with glob support
  - Automatic filtering of .git, node_modules, etc.

#### New Components
- `app/core/repo_context.py` - Data models for repository context (TechStack, TestFramework, CodeConventions, CICDConfig, RepoContext)
- `app/core/checkpoints.py` - CheckpointManager for state persistence
- `app/agents/analyzer.py` - CodebaseAnalyzer for automatic repository analysis
- `app/tools/search.py` - Safe repository search tool (search_in_repo)
- Router node - Entry point for intent classification
- Context Loader node - Repository analysis before planning
- Status node - Quick answers without file modifications
- Research node - Read-only analysis and search
- Resume node - Checkpoint loading and continuation
- Human Gate node - Approval checkpoint before commits

#### Enhanced Features
- **Context-Aware Agents** - Planner and coders now receive:
  - Repository structure analysis
  - Detected test frameworks and commands
  - Code style conventions
  - CI/CD awareness
- **Intelligent Routing** - Different workflows for different intents:
  - Code tasks → Full workflow with approval
  - Status queries → Immediate response
  - Research requests → Read-only search
  - Resume commands → Checkpoint restoration

#### API Endpoints
- `POST /api/approve` - Approve or reject pending commits
- Enhanced `GET /api/status` - Includes pending_approval details
- Enhanced `POST /api/task` - Supports all intent types (code, status, research, resume)

#### State Management
- 17 new GraphState fields:
  - `input_intent` - Classified intent type
  - `agent_instructions` - Content from AGENT.md/CLAUDE.md
  - `repo_facts` - Structured repository analysis
  - `context_listing` - Directory structure
  - `context_loaded` - Context loading flag
  - `needs_human_approval` - Approval gate flag
  - `pending_approval` - Approval payload with diff preview
  - `approval_history` - Log of all approvals
  - `state_checkpoint_id` - Current checkpoint ID
  - `last_checkpoint_path` - Path to last checkpoint
  - `resumed_from_checkpoint` - Resume flag

#### Testing
- `tests/test_router.py` - Intent classification tests
- `tests/test_context_loader.py` - Repository analysis tests
- `tests/test_search.py` - Search tool tests
- `tests/test_human_gate.py` - Approval gate tests
- `tests/test_checkpoints.py` - Checkpoint save/load tests
- `tests/test_planner_intent.py` - Planner behavior tests

#### Dependencies
- `tomli>=2.0.0` - TOML parsing for configuration files
- `pyyaml>=6.0` - YAML parsing for CI/CD configs

#### Documentation
- `AGENT.md` - Instructions for AI agents working on the codebase
- Updated README.md with v2.0 features and examples
- Enhanced inline documentation and docstrings

### Changed

#### Architecture
- **Entry Point**: Changed from `planner` node to `router` node
- **Workflow**: Added context loading step before planning for all code tasks
- **Approval Flow**: Inserted human gate before all commits
- **State Persistence**: Checkpoints now saved at every major step

#### Workflow Enhancements
- Planner now receives full repository context in system prompt
- Coder agents use detected test commands and style conventions
- Tester runs commands from repository analysis instead of assumptions
- All agents share repository context for consistent behavior

#### Breaking Changes
**None!** All changes are backward compatible. Existing workflows continue to work.

### Fixed
- Improved error handling in checkpoint save/load
- Better validation of search patterns
- Enhanced safety checks in human gate triggers
- More robust context parsing for edge cases

### Security
- Human approval now required for all commits (previously optional)
- Safe search tool eliminates need for shell execution in search
- Additional gates for large diffs and file deletions
- CI/CD config changes trigger special approval with warnings

### Performance
- Faster response for status queries (no unnecessary planning)
- Efficient context caching (loaded once per task)
- Optimized search with early termination at max_hits
- Reduced token usage through smart context truncation

---

## [0.1.0] - 2026-02-06

### Added
- Initial project scaffold with Clean Architecture structure
- LangGraph orchestrator with multi-agent workflow (Planner → Coder → Review → Test → Commit)
- Safe tools: filesystem (sandboxed), shell (blocklist), git (allow/block list)
- Build tools for Python, Node/TS, and .NET projects
- Agent role prompts (Planner/GPT-4o-mini, Coder/Claude, Tester)
- FastAPI web server with WebSocket live streaming
- Web UI with chat interface and status/log panel
- Telegram bot integration (/task, /status, /logs, /stop)
- Configuration via .env with pydantic-settings
- Rotating file + console logging
- Task management via tasks/todo.md and tasks/lessons.md
- Definition of Done document
