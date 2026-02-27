"""Split node package with backward-compatible re-exports for app.core.nodes."""

from __future__ import annotations

import functools

from . import _helpers as _m__helpers
from . import _streaming as _m__streaming
from . import router as _m_router
from . import resume as _m_resume
from . import _intelligence_helpers as _m__intelligence_helpers
from . import _prompt_enrichment as _m__prompt_enrichment
from . import context_loader as _m_context_loader
from . import intelligence as _m_intelligence
from . import status as _m_status
from . import research as _m_research
from . import planner as _m_planner
from . import planner_reviewing as _m_planner_reviewing
from . import coder as _m_coder
from . import reviewer as _m_reviewer
from . import tester as _m_tester
from . import gates as _m_gates
from . import committer as _m_committer
from . import documenter as _m_documenter

from ._helpers import *  # noqa: F401,F403
from ._streaming import *  # noqa: F401,F403
from .router import *  # noqa: F401,F403
from .resume import *  # noqa: F401,F403
from ._intelligence_helpers import *  # noqa: F401,F403
from ._prompt_enrichment import *  # noqa: F401,F403
from .tester import *  # noqa: F401,F403
from ._streaming import _STREAMING_ROLES, _TOKEN_BATCH_MIN
from .tester import _MAX_ENV_FIX_ATTEMPTS

from ._streaming import _model_name_for_role as _raw__model_name_for_role
from .coder import _parse_coder_question as _raw__parse_coder_question
from ._streaming import _stream_llm_round as _raw__stream_llm_round
from ._streaming import _invoke_agent as _raw__invoke_agent
from ._helpers import _assign_coder_pair as _raw__assign_coder_pair
from ._helpers import _reviewer_for_worker as _raw__reviewer_for_worker
from ._helpers import _coder_label as _raw__coder_label
from ._helpers import _reviewer_label as _raw__reviewer_label
from ._helpers import _candidate_agents_for_task as _raw__candidate_agents_for_task
from ._helpers import _get_budget as _raw__get_budget
from ._helpers import _budget_dict as _raw__budget_dict
from ._helpers import _invoke_with_budget as _raw__invoke_with_budget
from ._helpers import _os_note as _raw__os_note
from ._helpers import _progress_meta as _raw__progress_meta
from ._helpers import _write_todo_file as _raw__write_todo_file
from ._helpers import _save_checkpoint_snapshot as _raw__save_checkpoint_snapshot
from .router import _classify_request_intent as _raw__classify_request_intent
from .router import _heuristic_router_intent as _raw__heuristic_router_intent
from .router import _extract_repo_ref as _raw__extract_repo_ref
from .router import _parse_router_json as _raw__parse_router_json
from .router import _owner_to_agent as _raw__owner_to_agent
from .resume import _parse_todo_for_resume as _raw__parse_todo_for_resume
from .resume import _resume_from_saved_todo as _raw__resume_from_saved_todo
from .router import _answer_question_directly as _raw__answer_question_directly
from .router import router_node as _raw_router_node
from .context_loader import _hydrate_issue as _raw__hydrate_issue
from .context_loader import context_loader_node as _raw_context_loader_node
from .intelligence import code_intelligence_node as _raw_code_intelligence_node
from .intelligence import _emit_intelligence_complete as _raw__emit_intelligence_complete
from ._intelligence_helpers import _truncate_context_text as _raw__truncate_context_text
from ._intelligence_helpers import _build_context_listing as _raw__build_context_listing
from ._intelligence_helpers import _heuristic_analysis as _raw__heuristic_analysis
from ._prompt_enrichment import _format_intelligence_summary_for_prompt as _raw__format_intelligence_summary_for_prompt
from ._prompt_enrichment import _format_intelligence_summary_reviewer as _raw__format_intelligence_summary_reviewer
from ._prompt_enrichment import _format_intelligence_summary_tester as _raw__format_intelligence_summary_tester
from ._intelligence_helpers import _format_context_summary as _raw__format_context_summary
from ._intelligence_helpers import _extract_test_command as _raw__extract_test_command
from ._intelligence_helpers import _extract_language as _raw__extract_language
from ._prompt_enrichment import _format_call_graph_for_prompt as _raw__format_call_graph_for_prompt
from ._prompt_enrichment import _format_code_smells_for_prompt as _raw__format_code_smells_for_prompt
from ._prompt_enrichment import _format_dep_graph_for_prompt as _raw__format_dep_graph_for_prompt
from ._prompt_enrichment import _format_static_issues_for_prompt as _raw__format_static_issues_for_prompt
from ._prompt_enrichment import _format_repo_context_for_prompt as _raw__format_repo_context_for_prompt
from .status import status_node as _raw_status_node
from .research import research_node as _raw_research_node
from .resume import resume_node as _raw_resume_node
from .planner import planner_plan_node as _raw_planner_plan_node
from .planner import _compress_memory_file as _raw__compress_memory_file
from .planner import _parse_plan_from_result as _raw__parse_plan_from_result
from .coder import coder_node as _raw_coder_node
from .reviewer import peer_review_node as _raw_peer_review_node
from .reviewer import learn_from_review_node as _raw_learn_from_review_node
from .reviewer import _parse_learnings as _raw__parse_learnings
from .planner_reviewing import planner_review_node as _raw_planner_review_node
from .tester import _is_env_failure as _raw__is_env_failure
from .tester import _is_test_pass as _raw__is_test_pass
from .tester import _classify_test_output as _raw__classify_test_output
from .tester import tester_node as _raw_tester_node
from .planner_reviewing import planner_env_fix_node as _raw_planner_env_fix_node
from .planner_reviewing import planner_decide_node as _raw_planner_decide_node
from .gates import _count_lines_in_numstat as _raw__count_lines_in_numstat
from .gates import _parse_changed_files_from_status as _raw__parse_changed_files_from_status
from .gates import _format_plan_for_human as _raw__format_plan_for_human
from .gates import plan_approval_gate_node as _raw_plan_approval_gate_node
from .gates import answer_gate_node as _raw_answer_gate_node
from .gates import human_gate_node as _raw_human_gate_node
from .committer import _extract_commit_message as _raw__extract_commit_message
from .committer import _build_pr_repo_path as _raw__build_pr_repo_path
from .committer import _create_pr_for_branch as _raw__create_pr_for_branch
from .committer import _try_post_pr_link_on_issue as _raw__try_post_pr_link_on_issue
from .committer import committer_node as _raw_committer_node
from .documenter import _diff_needs_docs as _raw__diff_needs_docs
from .documenter import documenter_node as _raw_documenter_node

_PATCH_SYNC_MODULES = [
    _m__helpers,
    _m__streaming,
    _m_router,
    _m_resume,
    _m__intelligence_helpers,
    _m__prompt_enrichment,
    _m_context_loader,
    _m_intelligence,
    _m_status,
    _m_research,
    _m_planner,
    _m_planner_reviewing,
    _m_coder,
    _m_reviewer,
    _m_tester,
    _m_gates,
    _m_committer,
    _m_documenter,
]

_SYNC_SKIP = {
    'functools', '_PATCH_SYNC_MODULES', '_SYNC_SKIP', '_sync_patched_globals', '_wrap_with_patch_sync',
    '_m__helpers',
    '_m__streaming',
    '_m_router',
    '_m_resume',
    '_m__intelligence_helpers',
    '_m__prompt_enrichment',
    '_m_context_loader',
    '_m_intelligence',
    '_m_status',
    '_m_research',
    '_m_planner',
    '_m_planner_reviewing',
    '_m_coder',
    '_m_reviewer',
    '_m_tester',
    '_m_gates',
    '_m_committer',
    '_m_documenter',
    '_raw__model_name_for_role',
    '_raw__parse_coder_question',
    '_raw__stream_llm_round',
    '_raw__invoke_agent',
    '_raw__assign_coder_pair',
    '_raw__reviewer_for_worker',
    '_raw__coder_label',
    '_raw__reviewer_label',
    '_raw__candidate_agents_for_task',
    '_raw__get_budget',
    '_raw__budget_dict',
    '_raw__invoke_with_budget',
    '_raw__os_note',
    '_raw__progress_meta',
    '_raw__write_todo_file',
    '_raw__save_checkpoint_snapshot',
    '_raw__classify_request_intent',
    '_raw__heuristic_router_intent',
    '_raw__extract_repo_ref',
    '_raw__parse_router_json',
    '_raw__owner_to_agent',
    '_raw__parse_todo_for_resume',
    '_raw__resume_from_saved_todo',
    '_raw__answer_question_directly',
    '_raw_router_node',
    '_raw__hydrate_issue',
    '_raw_context_loader_node',
    '_raw_code_intelligence_node',
    '_raw__emit_intelligence_complete',
    '_raw__truncate_context_text',
    '_raw__build_context_listing',
    '_raw__heuristic_analysis',
    '_raw__format_intelligence_summary_for_prompt',
    '_raw__format_intelligence_summary_reviewer',
    '_raw__format_intelligence_summary_tester',
    '_raw__format_context_summary',
    '_raw__extract_test_command',
    '_raw__extract_language',
    '_raw__format_call_graph_for_prompt',
    '_raw__format_code_smells_for_prompt',
    '_raw__format_dep_graph_for_prompt',
    '_raw__format_static_issues_for_prompt',
    '_raw__format_repo_context_for_prompt',
    '_raw_status_node',
    '_raw_research_node',
    '_raw_resume_node',
    '_raw_planner_plan_node',
    '_raw__compress_memory_file',
    '_raw__parse_plan_from_result',
    '_raw_coder_node',
    '_raw_peer_review_node',
    '_raw_learn_from_review_node',
    '_raw__parse_learnings',
    '_raw_planner_review_node',
    '_raw__is_env_failure',
    '_raw__is_test_pass',
    '_raw__classify_test_output',
    '_raw_tester_node',
    '_raw_planner_env_fix_node',
    '_raw_planner_decide_node',
    '_raw__count_lines_in_numstat',
    '_raw__parse_changed_files_from_status',
    '_raw__format_plan_for_human',
    '_raw_plan_approval_gate_node',
    '_raw_answer_gate_node',
    '_raw_human_gate_node',
    '_raw__extract_commit_message',
    '_raw__build_pr_repo_path',
    '_raw__create_pr_for_branch',
    '_raw__try_post_pr_link_on_issue',
    '_raw_committer_node',
    '_raw__diff_needs_docs',
    '_raw_documenter_node',
}

def _sync_patched_globals() -> None:
    pkg = globals()
    for mod in _PATCH_SYNC_MODULES:
        md = mod.__dict__
        for name, value in pkg.items():
            if name.startswith('__') or name in _SYNC_SKIP:
                continue
            if name in md:
                md[name] = value

def _wrap_with_patch_sync(fn):
    @functools.wraps(fn)
    def _wrapped(*args, **kwargs):
        _sync_patched_globals()
        return fn(*args, **kwargs)
    return _wrapped

_model_name_for_role = _wrap_with_patch_sync(_raw__model_name_for_role)
_parse_coder_question = _wrap_with_patch_sync(_raw__parse_coder_question)
_stream_llm_round = _wrap_with_patch_sync(_raw__stream_llm_round)
_invoke_agent = _wrap_with_patch_sync(_raw__invoke_agent)
_assign_coder_pair = _wrap_with_patch_sync(_raw__assign_coder_pair)
_reviewer_for_worker = _wrap_with_patch_sync(_raw__reviewer_for_worker)
_coder_label = _wrap_with_patch_sync(_raw__coder_label)
_reviewer_label = _wrap_with_patch_sync(_raw__reviewer_label)
_candidate_agents_for_task = _wrap_with_patch_sync(_raw__candidate_agents_for_task)
_get_budget = _wrap_with_patch_sync(_raw__get_budget)
_budget_dict = _wrap_with_patch_sync(_raw__budget_dict)
_invoke_with_budget = _wrap_with_patch_sync(_raw__invoke_with_budget)
_os_note = _wrap_with_patch_sync(_raw__os_note)
_progress_meta = _wrap_with_patch_sync(_raw__progress_meta)
_write_todo_file = _wrap_with_patch_sync(_raw__write_todo_file)
_save_checkpoint_snapshot = _wrap_with_patch_sync(_raw__save_checkpoint_snapshot)
_classify_request_intent = _wrap_with_patch_sync(_raw__classify_request_intent)
_heuristic_router_intent = _wrap_with_patch_sync(_raw__heuristic_router_intent)
_extract_repo_ref = _wrap_with_patch_sync(_raw__extract_repo_ref)
_parse_router_json = _wrap_with_patch_sync(_raw__parse_router_json)
_owner_to_agent = _wrap_with_patch_sync(_raw__owner_to_agent)
_parse_todo_for_resume = _wrap_with_patch_sync(_raw__parse_todo_for_resume)
_resume_from_saved_todo = _wrap_with_patch_sync(_raw__resume_from_saved_todo)
_answer_question_directly = _wrap_with_patch_sync(_raw__answer_question_directly)
router_node = _wrap_with_patch_sync(_raw_router_node)
_hydrate_issue = _wrap_with_patch_sync(_raw__hydrate_issue)
context_loader_node = _wrap_with_patch_sync(_raw_context_loader_node)
code_intelligence_node = _wrap_with_patch_sync(_raw_code_intelligence_node)
_emit_intelligence_complete = _wrap_with_patch_sync(_raw__emit_intelligence_complete)
_truncate_context_text = _wrap_with_patch_sync(_raw__truncate_context_text)
_build_context_listing = _wrap_with_patch_sync(_raw__build_context_listing)
_heuristic_analysis = _wrap_with_patch_sync(_raw__heuristic_analysis)
_format_intelligence_summary_for_prompt = _wrap_with_patch_sync(_raw__format_intelligence_summary_for_prompt)
_format_intelligence_summary_reviewer = _wrap_with_patch_sync(_raw__format_intelligence_summary_reviewer)
_format_intelligence_summary_tester = _wrap_with_patch_sync(_raw__format_intelligence_summary_tester)
_format_context_summary = _wrap_with_patch_sync(_raw__format_context_summary)
_extract_test_command = _wrap_with_patch_sync(_raw__extract_test_command)
_extract_language = _wrap_with_patch_sync(_raw__extract_language)
_format_call_graph_for_prompt = _wrap_with_patch_sync(_raw__format_call_graph_for_prompt)
_format_code_smells_for_prompt = _wrap_with_patch_sync(_raw__format_code_smells_for_prompt)
_format_dep_graph_for_prompt = _wrap_with_patch_sync(_raw__format_dep_graph_for_prompt)
_format_static_issues_for_prompt = _wrap_with_patch_sync(_raw__format_static_issues_for_prompt)
_format_repo_context_for_prompt = _wrap_with_patch_sync(_raw__format_repo_context_for_prompt)
status_node = _wrap_with_patch_sync(_raw_status_node)
research_node = _wrap_with_patch_sync(_raw_research_node)
resume_node = _wrap_with_patch_sync(_raw_resume_node)
planner_plan_node = _wrap_with_patch_sync(_raw_planner_plan_node)
_compress_memory_file = _wrap_with_patch_sync(_raw__compress_memory_file)
_parse_plan_from_result = _wrap_with_patch_sync(_raw__parse_plan_from_result)
coder_node = _wrap_with_patch_sync(_raw_coder_node)
peer_review_node = _wrap_with_patch_sync(_raw_peer_review_node)
learn_from_review_node = _wrap_with_patch_sync(_raw_learn_from_review_node)
_parse_learnings = _wrap_with_patch_sync(_raw__parse_learnings)
planner_review_node = _wrap_with_patch_sync(_raw_planner_review_node)
_is_env_failure = _wrap_with_patch_sync(_raw__is_env_failure)
_is_test_pass = _wrap_with_patch_sync(_raw__is_test_pass)
_classify_test_output = _wrap_with_patch_sync(_raw__classify_test_output)
tester_node = _wrap_with_patch_sync(_raw_tester_node)
planner_env_fix_node = _wrap_with_patch_sync(_raw_planner_env_fix_node)
planner_decide_node = _wrap_with_patch_sync(_raw_planner_decide_node)
_count_lines_in_numstat = _wrap_with_patch_sync(_raw__count_lines_in_numstat)
_parse_changed_files_from_status = _wrap_with_patch_sync(_raw__parse_changed_files_from_status)
_format_plan_for_human = _wrap_with_patch_sync(_raw__format_plan_for_human)
plan_approval_gate_node = _wrap_with_patch_sync(_raw_plan_approval_gate_node)
answer_gate_node = _wrap_with_patch_sync(_raw_answer_gate_node)
human_gate_node = _wrap_with_patch_sync(_raw_human_gate_node)
_extract_commit_message = _wrap_with_patch_sync(_raw__extract_commit_message)
_build_pr_repo_path = _wrap_with_patch_sync(_raw__build_pr_repo_path)
_create_pr_for_branch = _wrap_with_patch_sync(_raw__create_pr_for_branch)
_try_post_pr_link_on_issue = _wrap_with_patch_sync(_raw__try_post_pr_link_on_issue)
committer_node = _wrap_with_patch_sync(_raw_committer_node)
_diff_needs_docs = _wrap_with_patch_sync(_raw__diff_needs_docs)
documenter_node = _wrap_with_patch_sync(_raw_documenter_node)

del functools
