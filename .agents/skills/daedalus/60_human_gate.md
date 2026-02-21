# Skill: Human Gate (Approval required)

## Goal
Stop and ask for explicit approval before risky actions.

## Approval triggers
- commit / push
- file deletes
- large diffs (e.g., >400 LOC changed)
- secrets / CI / deployment changes
- unusual shell commands

## Ask format
- What I want to do
- Why
- Exact command / diff
- Risks and rollback plan

## Output
Set needs_human=true + human_question + optional payload.
