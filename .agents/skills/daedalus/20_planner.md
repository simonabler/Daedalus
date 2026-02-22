# Skill: Planner (Small steps + acceptance criteria)

## Goal
Create a plan with small, testable steps.

## Output contract (JSON)
{
  "needs_human": false,
  "questions": [],
  "plan": [
    {"id":"1","title":"...","owner":"mode:general|bugfix|refactor|tests|docs|infra","acceptance":"..."}
  ]
}

## Rules
- Max 8 steps
- Each step must have explicit acceptance criteria
- Prefer incremental diffs and early test runs
- If requirements unclear: needs_human=true + questions
