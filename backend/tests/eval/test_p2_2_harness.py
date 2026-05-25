"""Cut P2.2-②a — eval harness unit tests."""
import json
from pathlib import Path

import pytest

from app.eval.p2_2_agent_ablation.matrix import RunSpec, expand_matrix
from app.eval.p2_2_agent_ablation.single_run import validate_record_schema


def test_matrix_expansion_main_run_count_matches_spec():
    """Main matrix (no appendix) over single-turn queries only."""
    specs = expand_matrix(
        models=["gemma3:4b", "qwen3.5:4b", "qwen2.5:7b", "gemma4:e4b"],
        modes=["deterministic", "agent_loop"],
        single_turn_queries=[{"id": f"q{i}", "message": f"m{i}"} for i in range(10)],
        multi_turn_queries=[
            {"id": "mt1", "messages": ["a", "b"]},
            {"id": "mt2", "messages": ["c", "d"]},
        ],
        runs=3,
        thinking=False,
    )

    # 10 × 4 × 2 × 3 single-turn = 240
    # 2 multi-turn × 2 turns × 4 × 2 × 3 = 96
    # Total = 336
    assert len(specs) == 336
    # Every spec has unique run_id
    assert len({s.run_id for s in specs}) == 336
    # The two multi-turn queries each produced "turn_idx" 0 and 1 records
    turn_indices = {(s.query_id, s.turn_idx) for s in specs if s.query_id.startswith("mt")}
    assert (("mt1", 0) in turn_indices and ("mt1", 1) in turn_indices)


def test_record_schema_validation_accepts_full_and_rejects_missing_keys():
    """One test, two assertions — schema validation is one contract surface."""
    full = {
        "run_id": "abc-123",
        "timestamp": "2026-05-23T12:00:00",
        "model": "qwen2.5:7b",
        "mode": "agent_loop",
        "query_id": "plan_hyde",
        "turn_idx": 0,
        "run_idx": 0,
        "operational": {
            "wall_time_s": 4.23,
            "iterations": 3,
            "tool_calls": [{"name": "retriever_search", "count": 1}],
            "tool_call_count": 1,
            "tool_errors": 0,
            "input_tokens": 1843,
            "output_tokens": 412,
            "exit_reason": "natural_stop",
        },
        "output": {
            "plan_action": "generate",
            "milestones_persisted": 5,
            "milestones_json": [],
            "final_text_excerpt": "Plan: ...",
        },
        "judge_local": {"score": 0.78, "weak_dims": [], "reasoning": "..."},
        "judge_cloud": {"score": 0.82, "weak_dims": [], "reasoning": "...", "model": "gpt-4o-mini"},
    }
    # Must NOT raise on a full record
    validate_record_schema(full)

    # Reject when a required top-level key is absent
    partial = {k: v for k, v in full.items() if k != "timestamp"}
    with pytest.raises(ValueError, match="missing required key"):
        validate_record_schema(partial)


def test_resumable_skips_runs_already_in_results_jsonl(tmp_path):
    from app.eval.p2_2_agent_ablation.run_eval import filter_pending_specs

    all_specs = expand_matrix(
        models=["m1", "m2"],
        modes=["deterministic"],
        single_turn_queries=[{"id": "q1", "message": "x"}],
        multi_turn_queries=[],
        runs=2,
        thinking=False,
    )
    # All specs: 2 models × 1 mode × 1 query × 2 runs = 4
    assert len(all_specs) == 4

    # Pretend 2 are already in results
    results_path = tmp_path / "results.jsonl"
    done_ids = [all_specs[0].run_id, all_specs[2].run_id]
    with results_path.open("w") as f:
        for rid in done_ids:
            f.write(json.dumps({"run_id": rid, "model": "m1", "mode": "deterministic"}) + "\n")

    pending = filter_pending_specs(all_specs, results_path)
    assert len(pending) == 2
    assert all_specs[0].run_id not in {s.run_id for s in pending}
    assert all_specs[1].run_id in {s.run_id for s in pending}
