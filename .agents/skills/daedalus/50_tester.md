# Skill: Tester (Run and interpret checks)

## Goal
Pick the best commands and interpret results.

## Steps
1) Choose command from repo_facts.test_commands; if empty, infer safely
2) Run tests, collect exit code + summary
3) If failing: identify likely root cause and propose next step

## Rules
- Use allowlisted commands
- Prefer fastest relevant test subset
- If no tests available: explain how to validate manually
