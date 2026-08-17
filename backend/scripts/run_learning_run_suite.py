"""Execute the frozen 12x2 Learning Run suite against the local evaluation model."""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agent.tutor_attempt import TutorAttemptEngine
from app.api.eval_routes import connection_fingerprint
from app.db.models import Base, EvalRun, EvalScoreSet, EvalScorerExecution
from app.db.session import make_engine
from app.eval.learning_run.corpus import CorpusMaterializerController, CorpusSnapshotLoader
from app.eval.learning_run.registry import TaskRegistry
from app.eval.learning_run.repositories import (
    EvalExecutionClaimRepository,
    EvalExecutionControlRepository,
    EvalRunRepository,
    EvalScoreSetRepository,
    EvalScorerExecutionRepository,
)
from app.eval.learning_run.runner import TutorRunner
from app.eval.learning_run.service import EvalModelConnection, RunService
from app.llm.provider import LLMConfig, get_chat_model


DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "app"
    / "eval"
    / "learning_run"
    / "output"
    / "tutor-prompt-regression-v1.jsonl"
)


def _code_revision() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).resolve().parents[2],
                text=True,
            ).strip()
            or "unknown"
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _new_runner() -> TutorRunner:
    loader = CorpusSnapshotLoader()
    return TutorRunner(
        corpus_loader=loader,
        attempt_engine=TutorAttemptEngine(),
        materializer_controller=CorpusMaterializerController(loader),
    )


def _close_runner(runner: TutorRunner) -> None:
    controller = getattr(runner, "materializer_controller", None)
    shutdown = getattr(controller, "shutdown", None)
    if callable(shutdown):
        shutdown(wait=True)


def _connection(registry: TaskRegistry, *, base_url: str | None):
    # Fresh client per cell. reasoning=False is an inference flag for
    # thinking models (gemma4 / qwen3.5); it is not a frozen experiment axis.
    controls = registry.experiment.variants["tutor-v2"]
    scorer_config = dict(registry.scorer.model_config)
    tutor_parameters = dict(controls["parameters"])
    scorer_parameters = {
        key: value for key, value in scorer_config.items() if key not in {"provider", "model"}
    }
    llm_kwargs = {"reasoning": False}
    tutor_llm = get_chat_model(
        LLMConfig(provider="ollama", model="llama3.2", api_key=None, base_url=base_url),
        **tutor_parameters,
        **llm_kwargs,
    )
    scorer_llm = get_chat_model(
        LLMConfig(
            provider=str(scorer_config["provider"]),
            model=str(scorer_config["model"]),
            api_key=None,
            base_url=base_url,
        ),
        **scorer_parameters,
        **llm_kwargs,
    )
    return EvalModelConnection(
        tutor_provider="ollama",
        tutor_model="llama3.2",
        tutor_parameters=tutor_parameters,
        tutor_llm=tutor_llm,
        scorer_provider=str(scorer_config["provider"]),
        scorer_model=str(scorer_config["model"]),
        scorer_parameters=scorer_parameters,
        scorer_llm=scorer_llm,
        connection_fingerprint=connection_fingerprint("ollama", base_url),
    )


def _service(session: Session, registry: TaskRegistry, runner: TutorRunner) -> RunService:
    return RunService(
        registry=registry,
        tutor_runner=runner,
        runs=EvalRunRepository(session),
        score_sets=EvalScoreSetRepository(session),
        scorer_executions=EvalScorerExecutionRepository(session),
        claim_repository=EvalExecutionClaimRepository(session),
        code_revision=_code_revision(),
    )


def _record_for(session: Session, run: EvalRun) -> dict[str, Any]:
    score_sets = list(
        session.scalars(
            select(EvalScoreSet)
            .where(EvalScoreSet.run_id == run.id)
            .order_by(EvalScoreSet.created_at, EvalScoreSet.id)
        )
    )
    score_set_ids = [item.id for item in score_sets]
    executions = []
    if score_set_ids:
        executions = list(
            session.scalars(
                select(EvalScorerExecution)
                .where(EvalScorerExecution.score_set_id.in_(score_set_ids))
                .order_by(EvalScorerExecution.created_at, EvalScorerExecution.id)
            )
        )
    return {
        "run": {
            "id": run.id,
            "experiment_id": run.experiment_id,
            "task_case_id": run.task_case_id,
            "task_case_version": run.task_case_version,
            "variant_id": run.variant_id,
            "run_profile": run.run_profile,
            "lifecycle": run.lifecycle,
            "outcome": run.outcome,
            "suite_execution_id": run.suite_execution_id,
            "manifest": run.manifest_json,
            "manifest_hash": run.manifest_hash,
            "candidate_artifact": run.candidate_artifact_json,
            "artifact_hash": run.artifact_hash,
            "operational_error_json": run.operational_error_json,
        },
        "score_sets": [
            {
                "id": item.id,
                "scorer_id": item.scorer_id,
                "scorer_version": item.scorer_version,
                "scorer_snapshot": item.scorer_snapshot_json,
                "scorer_definition_hash": item.scorer_definition_hash,
                "artifact_input_hash": item.artifact_input_hash,
                "status": item.status,
                "quality_verdict": item.quality_verdict,
                "aggregate_scores": item.aggregate_scores_json,
                "findings": item.findings_json,
                "operational_error_code": item.operational_error_code,
                "operational_error_message": item.operational_error_message,
            }
            for item in score_sets
        ],
        "executions": [
            {
                "id": item.id,
                "score_set_id": item.score_set_id,
                "scorer_id": item.scorer_id,
                "scorer_version": item.scorer_version,
                "status": item.status,
                "input_hash": item.input_hash,
                "output": item.output_json,
                "error_code": item.operational_error_code,
                "error_message": item.operational_error_message,
                "latency_ms": item.latency_ms,
                "usage": item.usage_json,
            }
            for item in executions
        ],
    }


def export_suite(session: Session, destination: Path) -> int:
    runs = list(session.scalars(select(EvalRun).order_by(EvalRun.task_case_id, EvalRun.variant_id, EvalRun.id)))
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(
            _record_for(session, run),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        for run in runs
        if run.lifecycle == "finished" and run.artifact_hash
    ]
    destination.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def _delete_run(session: Session, run: EvalRun) -> None:
    score_sets = list(session.scalars(select(EvalScoreSet).where(EvalScoreSet.run_id == run.id)))
    score_set_ids = [item.id for item in score_sets]
    if score_set_ids:
        executions = list(
            session.scalars(
                select(EvalScorerExecution).where(EvalScorerExecution.score_set_id.in_(score_set_ids))
            )
        )
        for execution in executions:
            session.delete(execution)
    for score_set in score_sets:
        session.delete(score_set)
    session.delete(run)
    session.commit()


def _suite_id(session: Session) -> str:
    existing = session.scalars(
        select(EvalRun.suite_execution_id).where(EvalRun.suite_execution_id.is_not(None))
    ).first()
    return existing or str(uuid.uuid4())


async def _run_one(
    *,
    registry: TaskRegistry,
    runner: TutorRunner,
    engine,
    connection,
    task_case_id: str,
    variant_id: str,
    suite_id: str,
) -> EvalRun:
    with Session(engine) as session:
        service = _service(session, registry, runner)
        prepared = service.prepare(
            experiment_id=registry.experiment.experiment_id,
            task_case_id=task_case_id,
            variant_id=variant_id,
            run_profile=registry.experiment.run_profile,
            connection=connection,
        )
        prepared.run.suite_execution_id = suite_id
        session.commit()
        result = await service.execute_prepared(prepared)
        session.refresh(result.run)
        return result.run


async def _rescore_one(
    *,
    registry: TaskRegistry,
    runner: TutorRunner,
    engine,
    connection,
    run_id: str,
    scorer_version: str,
) -> None:
    with Session(engine) as session:
        service = _service(session, registry, runner)
        prepared = service.prepare_rescore(
            run_id=run_id,
            scorer_version=scorer_version,
            connection=connection,
        )
        await service.execute_rescore(prepared)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the frozen Learning Run 12x2 suite")
    parser.add_argument("--database-url", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--only", help="Optional case,variant selector for a smoke cell")
    parser.add_argument("--skip-rescore", action="store_true")
    parser.add_argument("--base-url", default=None)
    args = parser.parse_args(argv)

    registry = TaskRegistry.load_default()
    engine = make_engine(args.database_url)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        EvalExecutionControlRepository(session).reconcile(
            started_before=datetime.utcnow() + timedelta(seconds=1)
        )
        suite_id = _suite_id(session)
        existing = list(session.scalars(select(EvalRun)))
        by_pair: dict[tuple[str, str], list[EvalRun]] = {}
        for run in existing:
            by_pair.setdefault((run.task_case_id, run.variant_id), []).append(run)

    targets = [
        (case_id, variant_id)
        for case_id in sorted(registry.task_case_ids)
        for variant_id in registry.experiment.variants
    ]
    if args.only:
        case_id, variant_id = args.only.split(",", 1)
        targets = [(case_id.strip(), variant_id.strip())]

    for case_id, variant_id in targets:
        pair = (case_id, variant_id)
        current = by_pair.get(pair, [])
        usable = [
            run
            for run in current
            if run.lifecycle == "finished" and run.artifact_hash
        ]
        if usable:
            print(f"skip finished {case_id}/{variant_id} {usable[0].id}", flush=True)
            continue
        with Session(engine) as session:
            for run in list(session.scalars(
                select(EvalRun).where(
                    EvalRun.task_case_id == case_id,
                    EvalRun.variant_id == variant_id,
                )
            )):
                print(f"replace incomplete {case_id}/{variant_id} {run.id}", flush=True)
                _delete_run(session, run)
        print(f"start {case_id}/{variant_id}", flush=True)
        runner = _new_runner()
        try:
            run = asyncio.run(
                _run_one(
                    registry=registry,
                    runner=runner,
                    engine=engine,
                    connection=_connection(registry, base_url=args.base_url),
                    task_case_id=case_id,
                    variant_id=variant_id,
                    suite_id=suite_id,
                )
            )
            print(
                f"finish {case_id}/{variant_id} {run.id} lifecycle={run.lifecycle} outcome={run.outcome}",
                flush=True,
            )
        finally:
            _close_runner(runner)
        with Session(engine) as session:
            export_suite(session, args.output)

    if not args.skip_rescore:
        with Session(engine) as session:
            finished = [
                run
                for run in session.scalars(select(EvalRun)).all()
                if run.lifecycle == "finished" and run.artifact_hash
            ]
            for run in finished:
                versions = {
                    item.scorer_version
                    for item in session.scalars(
                        select(EvalScoreSet).where(EvalScoreSet.run_id == run.id)
                    )
                    if item.status in {"completed", "partial", "failed", "cancelled"}
                }
                if "hybrid-v2" in versions:
                    print(f"skip rescore {run.task_case_id}/{run.variant_id}", flush=True)
                    continue
                print(f"rescore {run.task_case_id}/{run.variant_id} {run.id}", flush=True)
                runner = _new_runner()
                try:
                    asyncio.run(
                        _rescore_one(
                            registry=registry,
                            runner=runner,
                            engine=engine,
                            connection=_connection(registry, base_url=args.base_url),
                            run_id=run.id,
                            scorer_version="hybrid-v2",
                        )
                    )
                finally:
                    _close_runner(runner)
                with Session(engine) as export_session:
                    export_suite(export_session, args.output)

    with Session(engine) as session:
        count = export_suite(session, args.output)
    print(f"exported {count} runs -> {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
