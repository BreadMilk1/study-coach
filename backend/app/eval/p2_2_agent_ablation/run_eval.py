"""CLI entry point for the P2.2 ablation eval.

Usage:
  python -m app.eval.p2_2_agent_ablation.run_eval \\
      --queries app/eval/p2_2_agent_ablation/queries.json \\
      --output app/eval/p2_2_agent_ablation/output/results.jsonl \\
      [--runs 3] [--models gemma3:4b,qwen3.5:4b,qwen2.5:7b,gemma4:e4b]
      [--modes deterministic,agent_loop] [--thinking-appendix]

Resumable: re-running with the same --output skips run_ids that already have
a row in the file. Failures are written as error rows (operational.exit_reason
== "harness_error") rather than retried — failure IS data per spec §6.4.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from .matrix import RunSpec, expand_matrix


def filter_pending_specs(specs: list[RunSpec], results_path: Path) -> list[RunSpec]:
    if not results_path.exists():
        return list(specs)
    done = set()
    with results_path.open() as f:
        for line in f:
            try:
                done.add(json.loads(line)["run_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return [s for s in specs if s.run_id not in done]


async def main_async(args):
    """Wire up the graph + planner factories + judges, then iterate specs."""
    # Imports kept local so unit tests can import filter_pending_specs / matrix
    # without spinning up SQLite etc.
    from langchain_ollama import ChatOllama
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    from sqlalchemy.pool import StaticPool

    from app.agent.graph import build_graph
    from app.agent.memory_updater import build_memory_hydrator, build_memory_writer
    from app.agent.planner import build_planner
    from app.agent.planner_agent import build_planner_agent
    from app.db.models import Base
    from app.db.repositories import (
        GoalRepository, MasteryRepository, MistakeRepository,
        PlanRepository, TopicRepository, UserRepository,
    )
    from langgraph.checkpoint.memory import InMemorySaver

    from .judges import make_cloud_judge, make_local_judge
    from .single_run import run_one

    with Path(args.queries).open() as f:
        queries_doc = json.load(f)

    models = args.models.split(",")
    modes = args.modes.split(",")
    specs = expand_matrix(
        models=models, modes=modes,
        single_turn_queries=queries_doc.get("single_turn_plan", []),
        multi_turn_queries=queries_doc.get("multi_turn_check_in", []),
        runs=args.runs, thinking=False,
    )
    if args.thinking_appendix:
        specs += expand_matrix(
            models=["gemma4:e4b"], modes=modes,
            single_turn_queries=queries_doc.get("single_turn_plan", []),
            multi_turn_queries=[], runs=args.runs, thinking=True,
        )

    results_path = Path(args.output)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    pending = filter_pending_specs(specs, results_path)
    print(f"Total specs: {len(specs)}; pending: {len(pending)}", file=sys.stderr)

    # cloud-adapt: production deploy would key the judge_llm off x-judge-model; eval pins it to qwen2.5:7b for cross-mode score comparability.
    local_judge_llm = ChatOllama(model="qwen2.5:7b", temperature=0.0)
    judge_local = make_local_judge(local_judge_llm)
    judge_cloud = make_cloud_judge()

    # One DB/graph per model loop iteration so plans persist across multi-turn
    # turns but not across runs of different models (memory_writer would otherwise
    # carry mastery across).
    # Checkpointer must persist ACROSS turns of the same session_key, else
    # multi_turn_check_in turn 2 cannot see turn 1's plan state.
    savers: dict[str, InMemorySaver] = {}
    for spec in pending:
        engine = create_engine(
            f"sqlite:///{results_path.parent}/eval_{spec.session_key}.db",
            connect_args={"check_same_thread": False}, poolclass=StaticPool,
        )
        Base.metadata.create_all(engine)
        with Session(engine) as session:
            user = UserRepository(session).get_or_create(f"eval-{spec.session_key}")
            planner_llm = ChatOllama(
                model=spec.model, temperature=0.7,
                reasoning=spec.thinking,
                # cloud-adapt: ChatOllama 1.1 `reasoning` kwarg maps to Ollama API `think`
                # field (verified Cut ①f Phase B: qwen3.5:4b 813s → 7.3s with reasoning=False).
                # Main matrix specs have thinking=False; appendix gemma4:e4b runs use True.
            )

            class _NullRetriever:
                def search(self, q, top_k=5): return []
                def add_chunks(self, _): pass
            retriever = _NullRetriever()

            saver = savers.setdefault(spec.session_key, InMemorySaver())
            graph = build_graph(retriever=retriever, llm=planner_llm,
                                checkpointer=saver)
            config_extras = {
                "planner": build_planner(
                    llm=planner_llm, plan_repo=PlanRepository(session),
                    goal_repo=GoalRepository(session),
                    mastery_repo=MasteryRepository(session),
                    mistake_repo=MistakeRepository(session),
                    retriever=retriever,
                ),
                "planner_agent": build_planner_agent(
                    llm=planner_llm, plan_repo=PlanRepository(session),
                    goal_repo=GoalRepository(session),
                    mastery_repo=MasteryRepository(session),
                    mistake_repo=MistakeRepository(session),
                    retriever=retriever, max_iter=10,
                ),
                "memory_hydrator": build_memory_hydrator(
                    mastery_repo=MasteryRepository(session),
                    mistake_repo=MistakeRepository(session),
                ),
                "memory_writer": build_memory_writer(
                    mastery_repo=MasteryRepository(session),
                    mistake_repo=MistakeRepository(session),
                ),
                "judge_llm": None,  # eval judges run OUT of graph
                "quiz_master": None,
            }
            try:
                record = await run_one(
                    spec=spec, graph=graph,
                    judge_local=judge_local, judge_cloud=judge_cloud,
                    config_extras=config_extras, user_id=user.id,
                )
            except Exception as exc:
                record = {
                    "run_id": spec.run_id, "timestamp": "",
                    "model": spec.model, "mode": spec.mode,
                    "query_id": spec.query_id, "turn_idx": spec.turn_idx,
                    "run_idx": spec.run_idx,
                    "operational": {
                        "wall_time_s": 0.0, "iterations": 0,
                        "tool_calls": [], "tool_call_count": 0,
                        "tool_errors": 0, "input_tokens": 0, "output_tokens": 0,
                        "exit_reason": "harness_error",
                    },
                    "output": {"plan_action": None, "milestones_persisted": 0,
                               "milestones_json": [], "final_text_excerpt": str(exc)[:500]},
                    "judge_local": {}, "judge_cloud": {},
                }

        with results_path.open("a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"[{spec.run_id}] model={spec.model} mode={spec.mode} "
              f"exit={record['operational']['exit_reason']}", file=sys.stderr)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--queries", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--models", default="gemma3:4b,qwen3.5:4b,qwen2.5:7b,gemma4:e4b")
    p.add_argument("--modes", default="deterministic,agent_loop")
    p.add_argument("--thinking-appendix", action="store_true")
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
