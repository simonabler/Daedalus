# .agents

This folder contains **agent-facing configuration** for Daedalus (skills, playbooks, prompt fragments).

## Folder structure

- `skills/` — small, composable capability descriptions ("how to do X")
  - These files are intended to be *injected into* the supervising agent / subagents
  - Keep them short and operational (rules + contracts + outputs)

## Recommended usage in Daedalus

1) **Router first**
   - Load `.agents/skills/00_router.md`
   - Decide intent: `code|status|research|resume|question`

2) **Context loader**
   - Load `.agents/skills/10_context_loader.md`
   - Read `AGENT.md` (repo root) + key stack files
   - Extract `repo_facts` (stack + test/lint/format commands)

3) **Plan**
   - Load `.agents/skills/20_planner.md`
   - Produce a small plan (max 8 steps)

4) **Work loop**
   - Load `.agents/skills/30_coder.md` + `.agents/skills/50_tester.md` + `.agents/skills/40_reviewer.md`
   - Iterate: read → patch → test → evaluate until done or stuck

5) **Human gate**
   - Load `.agents/skills/60_human_gate.md`
   - Require approval before risky actions (commit/push/delete/large diffs/secrets)

6) **Resume**
   - Load `.agents/skills/70_resume.md`
   - Continue the last run safely after restart

## Conventions

- Skills should be **tool-agnostic** (describe behavior and contracts, not implementation).
- If a skill includes JSON output, it must be **strict JSON** (no markdown).
- Prefer small, testable steps and minimal diffs.

## Tip

If you store additional repo-specific instructions, keep them in the root `AGENT.md`.
Daedalus should always read `AGENT.md` before planning or editing.
