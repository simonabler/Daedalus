"""LangGraph node implementations — split into per-node modules.

All public node functions, helpers, and previously-accessible names are
re-exported here for backward compatibility.  Existing imports like
``from app.core.nodes import router_node`` continue to work unchanged.
"""

# -- Node functions --------------------------------------------------------

from .router import router_node  # noqa: F401
from .context_loader import context_loader_node  # noqa: F401
from .intelligence import code_intelligence_node  # noqa: F401
from .planner import (  # noqa: F401
    planner_decide_node,
    planner_env_fix_node,
    planner_plan_node,
    planner_review_node,
)
from .coder import coder_node  # noqa: F401
from .reviewer import learn_from_review_node, peer_review_node  # noqa: F401
from .tester import tester_node  # noqa: F401
from .gates import answer_gate_node, human_gate_node, plan_approval_gate_node  # noqa: F401
from .committer import committer_node  # noqa: F401
from .documenter import documenter_node  # noqa: F401
from .status import status_node  # noqa: F401
from .research import research_node  # noqa: F401
from .resume import resume_node  # noqa: F401

# -- Helpers re-exported for test & orchestrator compatibility -------------

from ._helpers import (  # noqa: F401
    CHECKBOX_RE,
    CODER_TOOLS,
    DOCUMENTER_TOOLS,
    PLANNER_TOOLS,
    REVIEWER_TOOLS,
    ROUTER_INTENTS,
    TESTER_TOOLS,
    _assign_coder_pair,
    _budget_dict,
    _candidate_agents_for_task,
    _coder_label,
    _get_budget,
    _invoke_agent,
    _invoke_with_budget,
    _model_name_for_role,
    _os_note,
    _parse_coder_question,
    _progress_meta,
    _reviewer_for_worker,
    _reviewer_label,
    _save_checkpoint_snapshot,
    _write_todo_file,
)

from ._streaming import _stream_llm_round, _STREAMING_ROLES, _TOKEN_BATCH_MIN  # noqa: F401

from .router import (  # noqa: F401
    _answer_question_directly,
    _classify_request_intent,
    _extract_repo_ref,
    _heuristic_router_intent,
    _owner_to_agent,
    _parse_router_json,
)

from .context_loader import (  # noqa: F401
    _build_context_listing,
    _heuristic_analysis,
    _hydrate_issue,
)

from ._context_format import (  # noqa: F401
    _extract_language,
    _extract_test_command,
    _format_call_graph_for_prompt,
    _format_code_smells_for_prompt,
    _format_context_summary,
    _format_dep_graph_for_prompt,
    _format_intelligence_summary_for_prompt,
    _format_intelligence_summary_reviewer,
    _format_intelligence_summary_tester,
    _format_repo_context_for_prompt,
    _format_static_issues_for_prompt,
    _truncate_context_text,
)

from .intelligence import _emit_intelligence_complete  # noqa: F401
from .planner import _compress_memory_file, _parse_plan_from_result  # noqa: F401
from .resume import _parse_todo_for_resume, _resume_from_saved_todo  # noqa: F401
from .reviewer import _parse_learnings  # noqa: F401
from .tester import _classify_test_output, _is_env_failure, _is_test_pass, _MAX_ENV_FIX_ATTEMPTS  # noqa: F401
from .gates import (  # noqa: F401
    _count_lines_in_numstat,
    _format_plan_for_human,
    _parse_changed_files_from_status,
)
from .committer import (  # noqa: F401
    _build_pr_repo_path,
    _create_pr_for_branch,
    _extract_commit_message,
    _try_post_pr_link_on_issue,
)
from .documenter import _diff_needs_docs  # noqa: F401

# -- Names that tests access as `nodes.X` via monkeypatch -----------------
# These were module-level imports in the original monolith.

from app.core.config import get_settings  # noqa: F401
from app.core.checkpoints import checkpoint_manager  # noqa: F401
from app.agents.models import get_llm, load_system_prompt  # noqa: F401
from app.core.state import GraphState  # noqa: F401
from app.core.events import (  # noqa: F401
    emit_node_end,
    emit_node_start,
    emit_plan,
    emit_plan_approval_needed,
    emit_status,
)
from app.core.memory import (  # noqa: F401
    ensure_memory_files,
    get_memory_stats,
    load_all_memory,
)
from app.core.task_routing import (  # noqa: F401
    history_summary,
    select_agent_thompson,
)
from app.tools.filesystem import read_file  # noqa: F401
from app.tools.git import git_command, git_create_branch  # noqa: F401

__all__ = [
    # Node functions (19)
    "answer_gate_node",
    "code_intelligence_node",
    "coder_node",
    "committer_node",
    "context_loader_node",
    "documenter_node",
    "human_gate_node",
    "learn_from_review_node",
    "peer_review_node",
    "plan_approval_gate_node",
    "planner_decide_node",
    "planner_env_fix_node",
    "planner_plan_node",
    "planner_review_node",
    "research_node",
    "resume_node",
    "router_node",
    "status_node",
    "tester_node",
    # Tool sets
    "CODER_TOOLS",
    "DOCUMENTER_TOOLS",
    "PLANNER_TOOLS",
    "REVIEWER_TOOLS",
    "TESTER_TOOLS",
]
