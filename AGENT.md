You are modifying the Daedalus repository.

Goal: Make Daedalus behave like Codex in VSCode:
1) Router decides intent (code/status/research/resume/question)
2) Read repo + AGENT.md before planning/coding
3) Select subagent mode
4) Work in loop until done or human approval is needed
5) Human-in-the-loop gate before risky actions (commit/push/delete/large diffs)
6) Add safe search tool (search_in_repo)

Implementation tasks:
A) Extend app/core/state.py GraphState with intent, agent_instructions, repo_facts, context_files, needs_human, human_question, human_payload, checkpoint_id.
B) Add prompt library files under app/agents/prompts/: router.py, context_loader.py, planner.py, coder.py, reviewer.py, human_gate.py.
C) Add safe tool app/tools/search.py exposing search_in_repo(repo_path, pattern, glob=None, max_hits=50) and register it in the tool registry used by nodes.
D) Add router_node as the graph entry point in app/core/orchestrator.py, with conditional routing:
   - status/research -> status_node (non-coding)
   - question -> human_question_node
   - resume -> resume_node
   - code -> context_loader -> planner -> existing workflow
E) Add context_loader_node to read AGENT.md/CONTRIBUTING/README + stack files and extract repo_facts via LLM JSON prompt.
F) Update planner node to include agent_instructions + repo_facts and output plan JSON (max 8 steps).
G) Update coder node prompt to enforce read-first, patch-first, test-first using tools (list_files/read_file/search_in_repo/apply_patch/run_cmd/git_diff).
H) Add human_gate_node before commit/push and any destructive file ops; when triggered set state.needs_human and stop.
I) Add stagnation detection in evaluate/decide to ask human when stuck.

Constraints:
- Keep diffs small and safe.
- Do not remove existing functionality; integrate with existing phase/workflow.
- Ensure all new LLM outputs are strict JSON parsing.
- Add minimal unit tests for router JSON parsing and search_in_repo behavior if a tests folder exists.

Deliverables:
- Code changes + brief summary in commit message style.
- Ensure existing workflow still runs for normal coding tasks.