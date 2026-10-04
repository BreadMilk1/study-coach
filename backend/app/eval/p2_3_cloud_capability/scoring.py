from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence

from app.agent.judge import QUIZ_DIMENSIONS, load_quiz_rubric
from app.eval.p2_3_cloud_capability.budget import BudgetExceeded, BudgetLedgerCorrupt, MissingUsage
from app.eval.p2_3_cloud_capability.protocol import (
    SmokeAbort,
    extract_usage,
    failure_class_for_exception,
    flatten_text_content,
    normalize_finish_status,
)

_ABORT = (SmokeAbort, BudgetExceeded, MissingUsage, BudgetLedgerCorrupt)

QUALITY_SCORER_IDS = (
    "legacy-visible-v1/qwen2.5:7b",
    "quiz-artifact-v1/qwen2.5:7b",
    "quiz-artifact-v1/deepseek-v4-pro",
    "quiz-artifact-v1/MiniMax-M2.7",
)
_ARTIFACT_JUDGES = (
    ("quiz-artifact-v1/qwen2.5:7b", "qwen", False),
    ("quiz-artifact-v1/deepseek-v4-pro", "deepseek", False),
    ("quiz-artifact-v1/MiniMax-M2.7", "m27", True),
)


def expected_scorer_ids(candidate: Mapping[str, Any] | None = None) -> list[str]:
    del candidate
    return list(QUALITY_SCORER_IDS)


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


class RecordingJudgeLLM:
    def __init__(self, inner):
        self._inner = inner
        self.last_text = ""
        self.last_response = None

    async def ainvoke(self, messages, **kwargs):
        resp = await self._inner.ainvoke(messages, **kwargs)
        self.last_response = resp
        self.last_text = flatten_text_content(getattr(resp, "content", None))
        return resp


def prepare_judge_content(raw: str, *, allow_strip_think: bool = False) -> tuple[str, dict[str, Any]]:
    text = raw or ""
    stripped = False
    if allow_strip_think and _THINK_RE.search(text):
        text = _THINK_RE.sub("", text).strip()
        stripped = True
    return text, {"think_stripped": stripped}


def _as_mapping(payload: Any) -> Mapping[str, Any] | None:
    if isinstance(payload, Mapping):
        return payload
    if isinstance(payload, str):
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, Mapping) else None
    return None


def validate_quiz_judge_payload(payload: Any) -> dict[str, Any] | None:
    data = _as_mapping(payload)
    if data is None:
        return None
    scores: list[float] = []
    dims: dict[str, float] = {}
    for dim in QUIZ_DIMENSIONS:
        if dim not in data:
            return None
        raw = data[dim]
        if type(raw) is bool or type(raw) not in (int, float):
            return None
        # Range-check the raw value before float(): 10**400 is an exact int and
        # float() would raise OverflowError before the check below could run.
        if raw < 1 or raw > 5:
            return None
        value = float(raw)
        if not math.isfinite(value) or value < 1 or value > 5:
            return None
        dims[dim] = value
        scores.append(value)
    return {
        "score": sum(scores) / len(scores) / 5.0,
        "dims": dims,
        "reasoning": str(data.get("reasoning", "")),
    }


def rubric_hash() -> str:
    return hashlib.sha256(load_quiz_rubric().encode("utf-8")).hexdigest()


def format_quiz_judge_prompt(*, question: str, answer: str, context: str) -> str:
    def _escape(value: str) -> str:
        return str(value).replace("{", "{{").replace("}", "}}")

    return load_quiz_rubric().format(
        question=_escape(question),
        answer=_escape(answer),
        context=_escape(context),
    )


def evidence_context(captures: Any) -> str:
    lines: list[str] = []
    for cap in captures or []:
        if not isinstance(cap, Mapping):
            continue
        for ev in cap.get("evidence") or []:
            if not isinstance(ev, Mapping):
                continue
            source = str(ev.get("source") or "")
            page = ev.get("page")
            content = str(ev.get("content") or "")
            loc = source if page is None else f"{source} p.{page}"
            lines.append(f"[{loc}] {content}".strip())
    return "\n".join(lines)


def _blocks_meta(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, list):
        blocks = []
        for block in content:
            if isinstance(block, Mapping):
                block_type = str(block.get("type") or "text")
                length = len(str(block.get("text") or block.get("thinking") or ""))
            else:
                block_type = str(getattr(block, "type", "text"))
                length = len(str(getattr(block, "text", "") or ""))
            blocks.append({"type": block_type, "length": length})
        return blocks
    text = flatten_text_content(content)
    return [{"type": "text", "length": len(text)}]


def _exec(**fields: Any) -> dict[str, Any]:
    base = {
        "scorer_id": fields.get("scorer_id"),
        "status": fields.get("status"),
        "error_code": fields.get("error_code"),
        "output": fields.get("output"),
        "role": fields.get("role", "quality"),
        "raw_response_sha256": fields.get("raw_response_sha256"),
        "blocks": fields.get("blocks"),
        "finish_status": fields.get("finish_status"),
        "usage": fields.get("usage"),
        "model": fields.get("model"),
        "protocol": fields.get("protocol"),
        "thinking": fields.get("thinking"),
        "rubric_hash": fields.get("rubric_hash"),
        "failure_class": fields.get("failure_class"),
    }
    return base


def _capture_ok(candidate: Mapping[str, Any]) -> bool:
    captures = candidate.get("retrieval_captures")
    if not isinstance(captures, list) or not captures:
        return False
    for item in captures:
        if not isinstance(item, Mapping):
            return False
        if item.get("capture_status") != "ok":
            return False
    return True


def _artifact_text(question: Mapping[str, Any]) -> str:
    return (
        f"prompt: {question.get('prompt')}\n"
        f"options: {question.get('options')}\n"
        f"answer: {question.get('answer')}\n"
        f"explanation: {question.get('explanation')}"
    )


async def _invoke_judge(judge: Any, prompt: str) -> Any:
    if judge is None:
        return None
    ainvoke = getattr(judge, "ainvoke", None)
    if callable(ainvoke):
        from langchain_core.messages import HumanMessage

        return await ainvoke([HumanMessage(content=prompt)])
    if callable(judge):
        result = judge(prompt)
        if hasattr(result, "__await__"):
            return await result
        return result
    return None


async def _score_one(
    *,
    judge: Any,
    scorer_id: str,
    question: str,
    answer: str,
    context: str,
    allow_strip_think: bool,
) -> dict[str, Any]:
    if judge is None:
        return _exec(scorer_id=scorer_id, status="failed", error_code="missing_judge", role="quality")
    prompt = format_quiz_judge_prompt(question=question, answer=answer, context=context)
    applied_hash = rubric_hash()
    try:
        response = await _invoke_judge(judge, prompt)
    except _ABORT:
        raise
    except Exception as exc:
        return _exec(
            scorer_id=scorer_id,
            status="failed",
            error_code=type(exc).__name__,
            role="quality",
            failure_class=failure_class_for_exception(exc),
            rubric_hash=applied_hash,
        )
    if response is None:
        return _exec(
            scorer_id=scorer_id,
            status="failed",
            error_code="missing_judge",
            role="quality",
        )
    raw_content = getattr(response, "content", response)
    raw_text = flatten_text_content(raw_content) if not isinstance(raw_content, str) else raw_content
    raw_bytes = raw_text.encode("utf-8")
    text, meta = prepare_judge_content(raw_text, allow_strip_think=False)
    parsed = validate_quiz_judge_payload(text)
    think_stripped = False
    if parsed is None and allow_strip_think:
        text, meta = prepare_judge_content(raw_text, allow_strip_think=True)
        parsed = validate_quiz_judge_payload(text)
        think_stripped = bool(meta.get("think_stripped"))
    protocol = getattr(judge, "protocol", "openai_compatible")
    stop = None
    metadata = getattr(response, "response_metadata", None) or {}
    if isinstance(metadata, Mapping):
        stop = metadata.get("stop_reason") or metadata.get("finish_reason")
    finish = normalize_finish_status(protocol=protocol, raw_stop=stop)
    if finish == "failed" and stop in {"stop", "end_turn"}:
        finish = "completed"
    if parsed is not None and finish == "failed" and stop is None:
        finish = "completed"
    usage = extract_usage(response)
    fields = dict(
        scorer_id=scorer_id,
        role="quality",
        raw_response_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        blocks=_blocks_meta(raw_content),
        finish_status=finish,
        usage=usage if usage is not None else "unavailable",
        model=getattr(judge, "model", None),
        protocol=protocol,
        thinking={
            "config": getattr(judge, "thinking_config", None),
            "stripped": think_stripped,
        },
        rubric_hash=applied_hash,
    )
    if parsed is None:
        return _exec(
            status="failed",
            error_code="parse",
            output=None,
            failure_class="model",
            **fields,
        )
    return _exec(status="success", output=parsed, **fields)


async def score_candidate(
    *,
    candidate: Mapping[str, Any],
    judges: Mapping[str, Any],
    scorer_ids: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    wanted = None if scorer_ids is None else set(scorer_ids)

    def include(scorer_id: str) -> bool:
        return wanted is None or scorer_id in wanted

    if candidate.get("quiz_action") == "grade":
        execs = [
            _exec(scorer_id="legacy-visible-v1/qwen2.5:7b", status="skipped", role="quality"),
            *[_exec(scorer_id=sid, status="skipped", role="quality") for sid, _, _ in _ARTIFACT_JUDGES],
        ]
        return [item for item in execs if include(str(item["scorer_id"]))]

    persisted = bool(candidate.get("question_persisted") and candidate.get("question"))
    capture_ok = _capture_ok(candidate)
    visible = str(candidate.get("final_text") or "")
    context = evidence_context(candidate.get("retrieval_captures"))
    query_label = str(candidate.get("query_id") or "visible quiz")
    if not persisted or not capture_ok:
        execs: list[dict[str, Any]] = []
        if include("legacy-visible-v1/qwen2.5:7b"):
            execs.append(
                await _score_one(
                    judge=judges.get("qwen"),
                    scorer_id="legacy-visible-v1/qwen2.5:7b",
                    question=query_label,
                    answer=visible,
                    context=context,
                    allow_strip_think=False,
                )
            )
        execs.extend(
            _exec(
                scorer_id=sid,
                status="failed",
                error_code="structured_artifact_failure",
                role="quality",
            )
            for sid, _, _ in _ARTIFACT_JUDGES
            if include(sid)
        )
        return execs
    question = candidate["question"]
    execs = []
    if include("legacy-visible-v1/qwen2.5:7b"):
        execs.append(
            await _score_one(
                judge=judges.get("qwen"),
                scorer_id="legacy-visible-v1/qwen2.5:7b",
                question=query_label,
                answer=visible,
                context=context,
                allow_strip_think=False,
            )
        )
    artifact_answer = _artifact_text(question)
    for scorer_id, key, strip in _ARTIFACT_JUDGES:
        if not include(scorer_id):
            continue
        execs.append(
            await _score_one(
                judge=judges.get(key),
                scorer_id=scorer_id,
                question=str(question.get("prompt") or "quiz artifact"),
                answer=artifact_answer,
                context=context,
                allow_strip_think=strip,
            )
        )
    return execs
