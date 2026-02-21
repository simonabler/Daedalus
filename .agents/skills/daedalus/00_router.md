# Skill: Router (Intent + Mode)

## Goal
Classify the user request before doing anything else.

## Intents
- status: summarize current state, do not modify repo
- research: explain/analyze only, do not modify repo
- resume: continue last workflow/checkpoint
- question: ask clarifying questions before acting
- code: proceed with coding workflow

## Output contract (JSON)
{
  "intent": "code|status|research|resume|question",
  "confidence": 0.0,
  "mode": "general|bugfix|refactor|tests|docs|infra",
  "questions": []
}

## Rules
- If unclear requirements → intent=question
- If user says “continue / after restart / last task” → intent=resume
- If user asks “where are we / what changed” → intent=status
- If code changes are required → intent=code
