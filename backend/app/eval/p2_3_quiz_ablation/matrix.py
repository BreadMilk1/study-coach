"""Matrix expansion: build RunSpec list from models × modes × queries × runs."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class RunSpec:
    run_id: str           # deterministic md5 hash of all other fields
    model: str
    mode: Literal["deterministic", "agent_loop"]
    thinking: bool        # reasoning=spec.thinking forwarded to ChatOllama
    query_id: str
    turn_idx: int         # 0 for single-turn or first of multi-turn; 1 for grade
    run_idx: int          # 0..runs-1 — repetitions for statistical power
    message: str          # the prompt text for this turn
    session_key: str      # langgraph thread_id (same across multi-turn turns)


def _run_id(model: str, mode: str, thinking: bool, query_id: str, turn_idx: int, run_idx: int) -> str:
    raw = f"{model}|{mode}|{thinking}|{query_id}|{turn_idx}|{run_idx}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def expand_matrix(
    *,
    models: list[str],
    modes: list[Literal["deterministic", "agent_loop"]],
    single_turn_queries: list[dict],
    multi_turn_queries: list[dict],
    runs: int = 3,
    thinking_appendix: bool = False,
) -> list[RunSpec]:
    """Expand matrix. Appendix is gemma4:e4b thinking-ON, single-turn only."""
    specs: list[RunSpec] = []
    for model in models:
        for mode in modes:
            for q in single_turn_queries:
                for r in range(runs):
                    specs.append(RunSpec(
                        run_id=_run_id(model, mode, False, q["id"], 0, r),
                        model=model, mode=mode, thinking=False,
                        query_id=q["id"], turn_idx=0, run_idx=r,
                        message=q["message"],
                        session_key=f"{model}|{mode}|{q['id']}|r{r}",
                    ))
            for q in multi_turn_queries:
                for r in range(runs):
                    for turn, msg in enumerate(q["messages"]):
                        specs.append(RunSpec(
                            run_id=_run_id(model, mode, False, q["id"], turn, r),
                            model=model, mode=mode, thinking=False,
                            query_id=q["id"], turn_idx=turn, run_idx=r,
                            message=msg,
                            session_key=f"{model}|{mode}|{q['id']}|r{r}",
                        ))
    if thinking_appendix:
        for mode in modes:
            for q in single_turn_queries:
                for r in range(runs):
                    specs.append(RunSpec(
                        run_id=_run_id("gemma4:e4b", mode, True, q["id"], 0, r),
                        model="gemma4:e4b", mode=mode, thinking=True,
                        query_id=q["id"], turn_idx=0, run_idx=r,
                        message=q["message"],
                        session_key=f"gemma4:e4b|{mode}|{q['id']}|r{r}|think",
                    ))
    return specs
