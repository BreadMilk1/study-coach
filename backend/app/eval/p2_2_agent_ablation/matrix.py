"""Matrix expansion for the P2.2 agent-loop ablation eval."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    model: str
    mode: Literal["deterministic", "agent_loop"]
    query_id: str
    turn_idx: int            # 0 for single-turn or first turn of multi-turn
    run_idx: int             # which repeat run (0..runs-1)
    thinking: bool           # appendix axis only; main matrix all False
    message: str             # the user message for this turn
    is_multi_turn: bool
    session_key: str         # shared across turns of one multi-turn run


def _run_id(*parts: str) -> str:
    """Deterministic ID so resumability key is reproducible across processes."""
    return hashlib.md5("|".join(parts).encode()).hexdigest()[:16]


def expand_matrix(
    *,
    models: list[str],
    modes: list[str],
    single_turn_queries: list[dict],
    multi_turn_queries: list[dict],
    runs: int,
    thinking: bool = False,
) -> list[RunSpec]:
    """Cartesian product of (model × mode × query × run), with multi-turn
    queries unrolled into one RunSpec per (query, turn). Each multi-turn run
    shares a session_key so the harness can replay turns against the same
    LangGraph checkpointer thread.
    """
    out: list[RunSpec] = []
    for model in models:
        for mode in modes:
            for run_idx in range(runs):
                # Single-turn
                for q in single_turn_queries:
                    sk = _run_id(model, mode, q["id"], str(run_idx), str(thinking), "ST")
                    rid = _run_id(model, mode, q["id"], "0", str(run_idx), str(thinking))
                    out.append(RunSpec(
                        run_id=rid, model=model, mode=mode,
                        query_id=q["id"], turn_idx=0, run_idx=run_idx,
                        thinking=thinking, message=q["message"],
                        is_multi_turn=False, session_key=sk,
                    ))
                # Multi-turn
                for q in multi_turn_queries:
                    sk = _run_id(model, mode, q["id"], str(run_idx), str(thinking), "MT")
                    for ti, msg in enumerate(q["messages"]):
                        rid = _run_id(model, mode, q["id"], str(ti), str(run_idx), str(thinking))
                        out.append(RunSpec(
                            run_id=rid, model=model, mode=mode,
                            query_id=q["id"], turn_idx=ti, run_idx=run_idx,
                            thinking=thinking, message=msg,
                            is_multi_turn=True, session_key=sk,
                        ))
    return out
