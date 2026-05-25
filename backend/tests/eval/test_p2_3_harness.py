"""Cut P2.3-②a — unit tests for the eval harness."""
import pytest

from app.eval.p2_3_quiz_ablation.matrix import (
    expand_matrix,
)
from app.eval.p2_3_quiz_ablation.single_run import validate_record_schema


def test_expand_matrix_main_count():
    """4 models × 2 modes × 10 single-turn × 3 runs + 4 × 2 × 2 multi × 2 turns × 3 = 336."""
    specs = expand_matrix(
        models=["gemma3:4b", "qwen3.5:4b", "qwen2.5:7b", "gemma4:e4b"],
        modes=["deterministic", "agent_loop"],
        single_turn_queries=[{"id": f"q{i}", "message": f"m{i}"} for i in range(10)],
        multi_turn_queries=[
            {"id": "mt1", "messages": ["turn0", "turn1"]},
            {"id": "mt2", "messages": ["turn0", "turn1"]},
        ],
        runs=3,
        thinking_appendix=False,
    )
    assert len(specs) == 336


def test_expand_matrix_with_appendix_adds_60():
    """Appendix: gemma4:e4b thinking-ON × 2 modes × 10 single × 3 runs = 60."""
    specs = expand_matrix(
        models=["gemma3:4b", "qwen3.5:4b", "qwen2.5:7b", "gemma4:e4b"],
        modes=["deterministic", "agent_loop"],
        single_turn_queries=[{"id": f"q{i}", "message": f"m{i}"} for i in range(10)],
        multi_turn_queries=[
            {"id": "mt1", "messages": ["turn0", "turn1"]},
            {"id": "mt2", "messages": ["turn0", "turn1"]},
        ],
        runs=3,
        thinking_appendix=True,
    )
    assert len(specs) == 336 + 60


def test_validate_record_schema_rejects_missing_keys():
    record = {
        "run_id": "abc", "timestamp": "t", "model": "m", "mode": "deterministic",
        "query_id": "q", "turn_idx": 0, "run_idx": 0,
        "operational": {"wall_time_s": 1, "iterations": 0},  # missing keys
        "output": {}, "judge_local": {}, "judge_cloud": {},
    }
    with pytest.raises(ValueError, match="record.operational missing required key"):
        validate_record_schema(record)
