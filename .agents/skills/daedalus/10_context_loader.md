# Skill: Context Loader (Repo + AGENT.md + Conventions)

## Goal
Collect repository context *before* planning or coding.

## Steps
1) Read: AGENT.md, README, CONTRIBUTING (if present)
2) Detect stack: python/node/go/rust/dotnet
3) Detect commands:
   - test commands (pytest, npm test, pnpm test, etc.)
   - lint commands (ruff, eslint, etc.)
   - format commands (black, prettier, etc.)
4) Identify entrypoints and key modules (top-level app/core/*)

## Output contract (JSON)
{
  "agent_instructions": "...",
  "repo_facts": {
    "detected_stack": "...",
    "test_commands": [],
    "lint_commands": [],
    "format_commands": [],
    "entrypoints": [],
    "conventions": []
  },
  "context_files": []
}

## Rules
- Prefer small snippets; do not paste massive files into context
- Do not invent missing files
