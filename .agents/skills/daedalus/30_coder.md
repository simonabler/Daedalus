# Skill: Coder (Read-first, patch-first, safe tools)

## Goal
Implement one plan step at a time using small diffs.

## Loop
1) Identify relevant files and read them
2) Search for references (safe search tool preferred)
3) Produce a small patch
4) Run tests/lint relevant to the change
5) Update state: patches, commands_run, test_results

## Rules
- Never modify code you did not read
- Prefer unified diffs / minimal edits
- If change is broad → split into smaller steps
- No secrets in outputs
