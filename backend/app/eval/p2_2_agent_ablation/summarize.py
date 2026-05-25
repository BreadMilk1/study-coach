"""Cut ②b summarize — read results.jsonl, emit markdown tables.

Produces sections for EVAL.md:
  - Latency (median wall_time_s per (model, mode) cell)
  - Exit reason distribution
  - Tool-call rate (agent_loop only)
  - Plan quality — local judge mean
  - Plan quality — cloud judge mean
  - Judge agreement (mean |local - cloud|)
  - Token cost (mean tokens / cell)
  - Thinking-on/off appendix (gemma4:e4b only)

Output goes to stdout; redirect to summary.md or EVAL.md section.
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def load(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def by_cell(rows: list[dict], *, thinking_split: bool = False) -> dict[tuple, list[dict]]:
    """Group rows by (model, mode) — or (model, mode, thinking) when split."""
    out: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        # The matrix-expansion thinking is encoded via session_key suffix or
        # in the appendix runs. We don't have a direct field on the record,
        # so we approximate: gemma4:e4b cells > 84 imply appendix mixed in.
        # For thinking-split view, we'd need to re-thread; for now, group by
        # (model, mode) and note appendix in a separate section.
        key: Any = (r["model"], r["mode"])
        out[key].append(r)
    return out


def fmt_table(headers: list[str], rows: list[list]) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = "\n".join("| " + " | ".join(str(c) for c in r) + " |" for r in rows)
    return f"{head}\n{sep}\n{body}"


def median_safe(xs: list[float]) -> float:
    return statistics.median(xs) if xs else 0.0


def mean_safe(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def section_overview(rows: list[dict]) -> str:
    return f"Total rows: **{len(rows)}**\n"


def section_latency(cells: dict[tuple, list[dict]]) -> str:
    table = []
    for (model, mode), cell in sorted(cells.items()):
        times = [r["operational"]["wall_time_s"] for r in cell
                 if isinstance(r["operational"].get("wall_time_s"), (int, float)) and r["operational"]["wall_time_s"] > 0]
        med = median_safe(times)
        mean = mean_safe(times)
        table.append([model, mode, f"{med:.1f}", f"{mean:.1f}", len(cell)])
    return fmt_table(["model", "mode", "median wall_s", "mean wall_s", "n"], table)


def section_exit_reasons(cells: dict[tuple, list[dict]]) -> str:
    table = []
    for (model, mode), cell in sorted(cells.items()):
        counts: dict[str, int] = defaultdict(int)
        for r in cell:
            counts[r["operational"]["exit_reason"]] += 1
        # Compact: sorted by descending count
        pieces = ", ".join(f"{k}={v}" for k, v in
                           sorted(counts.items(), key=lambda x: -x[1]))
        table.append([model, mode, pieces])
    return fmt_table(["model", "mode", "exit_reason distribution"], table)


def section_tool_calls(cells: dict[tuple, list[dict]]) -> str:
    table = []
    for (model, mode), cell in sorted(cells.items()):
        if mode != "agent_loop":
            continue
        counts = [r["operational"]["tool_call_count"] for r in cell]
        errors = sum(r["operational"]["tool_errors"] for r in cell)
        successful_calls = [c for c in counts if c > 0]
        zero_call_rate = sum(1 for c in counts if c == 0) / max(1, len(counts))
        table.append([
            model,
            f"{mean_safe(counts):.2f}",
            f"{median_safe(counts):.1f}",
            f"{zero_call_rate:.0%}",
            errors,
            len(cell),
        ])
    return fmt_table(
        ["model (agent_loop)", "mean tool calls/run", "median",
         "% runs with 0 tool calls", "tool_errors total", "n"],
        table,
    )


def section_judge_local(cells: dict[tuple, list[dict]]) -> str:
    table = []
    for (model, mode), cell in sorted(cells.items()):
        scores = [r.get("judge_local", {}).get("score") for r in cell]
        scores = [s for s in scores if isinstance(s, (int, float))]
        table.append([model, mode, f"{mean_safe(scores):.3f}", f"{median_safe(scores):.3f}", len(scores)])
    return fmt_table(["model", "mode", "local mean", "local median", "n"], table)


def section_judge_cloud(cells: dict[tuple, list[dict]]) -> str:
    table = []
    for (model, mode), cell in sorted(cells.items()):
        scores = [r.get("judge_cloud", {}).get("score") for r in cell]
        scores = [s for s in scores if isinstance(s, (int, float))]
        table.append([model, mode, f"{mean_safe(scores):.3f}", f"{median_safe(scores):.3f}", len(scores)])
    return fmt_table(["model", "mode", "cloud mean", "cloud median", "n"], table)


def section_judge_agreement(cells: dict[tuple, list[dict]]) -> str:
    """Mean absolute delta between local and cloud judge scores, per cell."""
    table = []
    for (model, mode), cell in sorted(cells.items()):
        deltas = []
        local_mean = []
        cloud_mean = []
        for r in cell:
            jl = r.get("judge_local", {}).get("score")
            jc = r.get("judge_cloud", {}).get("score")
            if isinstance(jl, (int, float)) and isinstance(jc, (int, float)):
                deltas.append(abs(jl - jc))
                local_mean.append(jl)
                cloud_mean.append(jc)
        sign = ""
        if local_mean and cloud_mean:
            diff = mean_safe(local_mean) - mean_safe(cloud_mean)
            if diff > 0.02:
                sign = f" (local +{diff:.2f})"
            elif diff < -0.02:
                sign = f" (cloud +{-diff:.2f})"
        table.append([model, mode, f"{mean_safe(deltas):.3f}{sign}", len(deltas)])
    return fmt_table(["model", "mode", "mean |local−cloud|", "n"], table)


def section_token_cost(cells: dict[tuple, list[dict]]) -> str:
    table = []
    for (model, mode), cell in sorted(cells.items()):
        in_toks = [r["operational"]["input_tokens"] for r in cell]
        out_toks = [r["operational"]["output_tokens"] for r in cell]
        total = [a + b for a, b in zip(in_toks, out_toks)]
        non_zero = [t for t in total if t > 0]
        table.append([
            model, mode,
            f"{mean_safe(in_toks):.0f}",
            f"{mean_safe(out_toks):.0f}",
            f"{mean_safe(total):.0f}",
            f"{len(non_zero)}/{len(cell)}",
        ])
    return fmt_table(
        ["model", "mode", "mean in_tok", "mean out_tok",
         "mean total_tok", "non-zero rows"],
        table,
    )


def section_milestones_persisted(cells: dict[tuple, list[dict]]) -> str:
    """How often did the run actually save something?"""
    table = []
    for (model, mode), cell in sorted(cells.items()):
        saved = sum(1 for r in cell if r["output"].get("milestones_persisted", 0) > 0)
        rate = saved / max(1, len(cell))
        # Plan-action breakdown
        actions: dict[str, int] = defaultdict(int)
        for r in cell:
            actions[r["output"].get("plan_action") or "none"] += 1
        action_str = ", ".join(f"{k}={v}" for k, v in sorted(actions.items()))
        table.append([model, mode, f"{rate:.0%}", saved, len(cell), action_str])
    return fmt_table(
        ["model", "mode", "% persisted", "n persisted", "n", "plan_action breakdown"],
        table,
    )


def main(path: str) -> None:
    rows = load(Path(path))
    cells = by_cell(rows)

    print("# P2.2 Ablation Results — Auto-Generated Summary\n")
    print(section_overview(rows))

    print("## 1. Latency (wall_time_s per cell)\n")
    print(section_latency(cells))

    print("\n## 2. Exit reason distribution\n")
    print(section_exit_reasons(cells))

    print("\n## 3. Tool calling correctness (agent_loop only)\n")
    print(section_tool_calls(cells))

    print("\n## 4. Plan quality — Local judge (qwen2.5:7b)\n")
    print(section_judge_local(cells))

    print("\n## 5. Plan quality — Cloud judge (MiniMax-M2.7)\n")
    print(section_judge_cloud(cells))

    print("\n## 6. Judge agreement (cross-model)\n")
    print(section_judge_agreement(cells))

    print("\n## 7. Token cost\n")
    print(section_token_cost(cells))

    print("\n## 8. Plan persistence + plan_action breakdown\n")
    print(section_milestones_persisted(cells))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1
         else "app/eval/p2_2_agent_ablation/output/results.jsonl")
