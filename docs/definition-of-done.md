# Definition of Done (DoD)

A TODO item is considered DONE only when ALL of the following are true:

## Code Quality
- [ ] Implementation follows Clean Architecture principles
- [ ] Code is minimal — only necessary changes, no unnecessary refactoring
- [ ] No hacks or workarounds — root-cause fixes only
- [ ] A staff engineer would approve this code

## Testing
- [ ] All existing tests pass (`pytest` / `npm test` / `dotnet test`)
- [ ] New tests added for the change (unit + integration where applicable)
- [ ] Linter passes with no new warnings (`ruff` / `eslint` / `dotnet build`)

## Documentation
- [ ] Relevant docs updated (README, API docs, inline comments)
- [ ] No orphaned TODOs in code (if a TODO exists, it's tracked in `tasks/todo.md`)

## Review
- [ ] Planner has reviewed the diff and approved
- [ ] Acceptance criteria verified by Tester agent
- [ ] Test evidence recorded in the test report

## Git
- [ ] Committed with Conventional Commit message
- [ ] On a feature branch (never directly on main/master)
- [ ] Pushed to origin
- [ ] No merge performed (humans merge via PR)

## Process
- [ ] `tasks/todo.md` updated (item marked as done)
- [ ] `CHANGELOG.md` updated if user-facing change
- [ ] `tasks/lessons.md` updated if a mistake was corrected
