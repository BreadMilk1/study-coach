"""One row in results.jsonl, plus the function that produces it."""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from langchain_core.messages import HumanMessage


REQUIRED_TOP_LEVEL_KEYS = (
    "run_id", "timestamp", "model", "mode", "query_id",
    "turn_idx", "run_idx",
    "operational", "output", "judge_local", "judge_cloud",
)
REQUIRED_OPERATIONAL_KEYS = (
    "wall_time_s", "iterations", "tool_calls", "tool_call_count",
    "tool_errors", "input_tokens", "output_tokens", "exit_reason",
)


def validate_record_schema(record: dict) -> None:
    """Cheap structural check — DOES NOT validate types deeply."""
    for k in REQUIRED_TOP_LEVEL_KEYS:
        if k not in record:
            raise ValueError(f"record missing required key: {k}")
    op = record["operational"]
    for k in REQUIRED_OPERATIONAL_KEYS:
        if k not in op:
            raise ValueError(f"record.operational missing required key: {k}")


async def run_one(
    *,
    spec,                    # RunSpec
    graph,                   # compiled LangGraph
    judge_local,             # callable: (question, plan_text) -> {score, weak_dims, reasoning}
    judge_cloud,             # callable, may be None if budget exhausted
    config_extras: dict,     # planner/planner_agent/memory_* callables + judge_llm
    user_id: str,
) -> dict:
    """Execute one RunSpec end-to-end, build the record dict, return it."""
    input_state = {
        "messages": [HumanMessage(content=spec.message)],
        "user_id": user_id,
    }
    config = {
        "configurable": {
            **config_extras,
            "thread_id": spec.session_key,
            "planner_mode": spec.mode,
        }
    }
    t0 = datetime.utcnow()
    try:
        final_state = await graph.ainvoke(input_state, config=config)
        err = None
    except Exception as exc:
        final_state = {}
        err = f"{type(exc).__name__}: {exc}"
    elapsed = (datetime.utcnow() - t0).total_seconds()

    trace = final_state.get("agent_trace") or {
        "wall_time_s": elapsed,
        "total_iterations": 0,
        "total_tool_calls": 0,
        "tool_call_breakdown": {},
        "tool_errors": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "exit_reason": "error" if err else ("deterministic" if spec.mode == "deterministic" else "n/a"),
        "llm_error": err,
    }
    final_text = ""
    msgs = final_state.get("messages") or []
    if msgs:
        last = msgs[-1]
        final_text = getattr(last, "content", "") or ""

    plan_action = final_state.get("plan_action")
    persisted = (
        len(final_state.get("active_plan_id") or "") > 0
    )

    judge_local_out = await judge_local(spec.message, final_text) if judge_local else {}
    judge_cloud_out = await judge_cloud(spec.message, final_text) if judge_cloud else {}

    record = {
        "run_id": spec.run_id,
        "timestamp": t0.isoformat(),
        "model": spec.model,
        "mode": spec.mode,
        "query_id": spec.query_id,
        "turn_idx": spec.turn_idx,
        "run_idx": spec.run_idx,
        "operational": {
            "wall_time_s": trace.get("wall_time_s", elapsed),
            "iterations": trace.get("total_iterations", 0),
            "tool_calls": [
                {"name": name, "count": count}
                for name, count in (trace.get("tool_call_breakdown") or {}).items()
            ],
            "tool_call_count": trace.get("total_tool_calls", 0),
            "tool_errors": trace.get("tool_errors", 0),
            "input_tokens": trace.get("input_tokens", 0),
            "output_tokens": trace.get("output_tokens", 0),
            "exit_reason": trace.get("exit_reason", "unknown"),
        },
        "output": {
            "plan_action": plan_action,
            "milestones_persisted": 1 if persisted else 0,
            "milestones_json": [],   # populated by reading PlanRepository if needed
            "final_text_excerpt": final_text[:500],
        },
        "judge_local": judge_local_out,
        "judge_cloud": judge_cloud_out,
    }
    validate_record_schema(record)
    return record
