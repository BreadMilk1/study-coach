"""Import a frozen Learning Run suite export in one transaction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sqlalchemy.orm import Session

from app.db.session import session_scope
from app.eval.learning_run.registry import TaskRegistry
from app.eval.learning_run.repositories import EvalSuiteImportRepository


def parse_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at line {line_number}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"invalid JSONL at line {line_number}: expected object")
        records.append(payload)
    return records


def import_suite(path: Path, *, session: Session | None = None) -> int:
    records = parse_jsonl(path)
    registry = TaskRegistry.load_default()
    if session is not None:
        return EvalSuiteImportRepository(session, registry=registry).import_records(records)
    with session_scope() as db:
        return EvalSuiteImportRepository(db, registry=registry).import_records(records)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Import a Learning Run suite JSONL export")
    parser.add_argument("path", nargs="?", type=Path, help="JSONL export path")
    args = parser.parse_args(argv)
    if args.path is None:
        parser.print_help()
        return 0
    count = import_suite(args.path)
    print(f"imported {count} runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
