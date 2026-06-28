"""Shared agent-loop instrumentation dataclasses.

Extracted from `planner_agent.py` in P2.3 Cut ①a so the Quiz agent (and any
future agent) can use the same trace shape without cross-module reach.

Per-run instrumentation is the only structured record the eval harness pulls,
so the `serialize()` output shape is contractual. Anything tightly coupled to
a specific tool name (`last_persisted_plan_id`, `get_existing_plan_returned_nonnull`,
`last_persisted_question_id`) lives on this class as helpers — they're parallel
small methods, not abstractions. Refactor to a single `last_persisted_id(tool_name,
key)` when a 3rd consumer arrives (YAGNI).
"""
from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass, field


def _compact_json(value: dict, *, limit: int = 200) -> str:
    raw = json.dumps(value or {}, ensure_ascii=False, separators=(",", ":"), default=str)
    if len(raw) <= limit:
        return raw
    return raw[: max(0, limit - 3)] + "..."


def _preview_output(tool_name: str, output: str, *, error: bool, limit: int = 220) -> str:
    text = str(output or "")
    if not error:
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            parsed = None
        if tool_name == "retriever_search" and isinstance(parsed, list):
            return f"{len(parsed)} chunks"
        if tool_name == "retriever_search" and text.lstrip().startswith("["):
            return "retriever chunks"
        if isinstance(parsed, dict):
            safe = {
                k: parsed.get(k)
                for k in ("plan_id", "question_id", "topic_id", "persisted")
                if k in parsed
            }
            if safe:
                return _compact_json(safe, limit=limit)
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


@dataclass
class IterationRecord:
    iteration: int
    has_tool_calls: bool
    tool_call_count: int
    input_tokens: int
    output_tokens: int


@dataclass
class ToolCallRecord:
    name: str
    args: dict
    output: str
    error: bool


@dataclass
class AgentTrace:
    """Per-run instrumentation. Serialized into eval results.jsonl rows.

    Field choices are deliberately minimal — anything tightly coupled to a
    specific schema (per-call latency breakdown, full tool output) is out
    because matrices have hundreds of runs and the file should stay grep-able.
    """
    t_start: float
    iterations: list[IterationRecord] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    exit_reason: str = "in_flight"
    llm_error: str | None = None

    def record_iteration(self, response, iteration: int) -> None:
        tcs = getattr(response, "tool_calls", None) or []
        usage = getattr(response, "usage_metadata", None) or {}
        self.iterations.append(IterationRecord(
            iteration=iteration,
            has_tool_calls=bool(tcs),
            tool_call_count=len(tcs),
            input_tokens=int(usage.get("input_tokens", 0) or 0),
            output_tokens=int(usage.get("output_tokens", 0) or 0),
        ))

    def record_tool_call(self, name: str, args: dict, output: str, *, error: bool) -> None:
        self.tool_calls.append(ToolCallRecord(
            name=name, args=dict(args or {}), output=str(output)[:500], error=error,
        ))
        # cloud-adapt: production deploy should redact output entirely; truncating
        # to 500 chars keeps it tractable for eval but still leaks user content.

    def record_budget_exhaustion(self, max_iter: int) -> None:
        self.exit_reason = "budget_exhausted"

    def record_llm_error(self, exc: str) -> None:
        self.exit_reason = "llm_call_failed"
        self.llm_error = exc

    def tool_names_called(self) -> list[str]:
        return [tc.name for tc in self.tool_calls if not tc.error]

    def get_existing_plan_returned_nonnull(self) -> bool:
        for tc in self.tool_calls:
            if tc.name == "get_existing_plan" and tc.output != "null":
                return True
        return False

    def last_persisted_plan_id(self) -> str | None:
        for tc in reversed(self.tool_calls):
            if tc.name == "update_study_plan" and not tc.error:
                try:
                    return json.loads(tc.output).get("plan_id")
                except (json.JSONDecodeError, AttributeError):
                    return None
        return None

    def last_persisted_question_id(self) -> str | None:
        """Return the question_id from the most recent successful
        persist_quiz_question tool call. Returns None if no successful
        persistence happened in this trace."""
        for tc in reversed(self.tool_calls):
            if tc.name == "persist_quiz_question" and not tc.error:
                try:
                    return json.loads(tc.output).get("question_id")
                except (json.JSONDecodeError, AttributeError):
                    return None
        return None

    def aggregated_retriever_context(self) -> str:
        """Flatten all retriever_search outputs into a single context string for
        consumers (memory_writer / future analytics) that expect the same
        `last_context` shape as the deterministic planner emits."""
        parts: list[str] = []
        for tc in self.tool_calls:
            if tc.name == "retriever_search" and not tc.error:
                try:
                    chunks = json.loads(tc.output)
                except (json.JSONDecodeError, TypeError):
                    continue
                for i, c in enumerate(chunks, start=1):
                    parts.append(f"[{i}] {c.get('content', '')}")
        return "\n".join(parts)

    def serialize(self) -> dict:
        return {
            "total_iterations": len(self.iterations),
            "total_tool_calls": len(self.tool_calls),
            "tool_call_breakdown": dict(Counter(tc.name for tc in self.tool_calls)),
            "tool_errors": sum(1 for tc in self.tool_calls if tc.error),
            "input_tokens": sum(it.input_tokens for it in self.iterations),
            "output_tokens": sum(it.output_tokens for it in self.iterations),
            "wall_time_s": time.monotonic() - self.t_start,
            "exit_reason": self.exit_reason,
            "llm_error": self.llm_error,
        }

    def serialize_public(self, *, node: str, mode: str = "agent_loop") -> dict:
        return {
            "node": node,
            "mode": mode,
            "total_iterations": len(self.iterations),
            "total_tool_calls": len(self.tool_calls),
            "tool_call_breakdown": dict(Counter(tc.name for tc in self.tool_calls)),
            "tool_errors": sum(1 for tc in self.tool_calls if tc.error),
            "input_tokens": sum(it.input_tokens for it in self.iterations),
            "output_tokens": sum(it.output_tokens for it in self.iterations),
            "wall_time_s": time.monotonic() - self.t_start,
            "exit_reason": self.exit_reason,
            "llm_error": self.llm_error,
            "tool_calls": [
                {
                    "name": tc.name,
                    "error": tc.error,
                    "args_preview": _compact_json(tc.args),
                    "output_preview": _preview_output(tc.name, tc.output, error=tc.error),
                }
                for tc in self.tool_calls
            ],
        }
