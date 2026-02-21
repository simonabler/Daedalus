# Skill: Reviewer (Correctness + safety + conventions)

## Goal
Review diff and results for:
- correctness
- security/safety
- adherence to AGENT.md and repo conventions
- test coverage / verification steps

## Output (JSON)
{
  "approved": true,
  "risk_level": "low|medium|high",
  "issues": [],
  "suggested_fixes": []
}

## Rules
- If risk_level=high → require human approval
- Flag large refactors or config/secrets changes
