# Changelog

All notable changes to the AI Dev Worker project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Conventional Commits](https://www.conventionalcommits.org/).

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
