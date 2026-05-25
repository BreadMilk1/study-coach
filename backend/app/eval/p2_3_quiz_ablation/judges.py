"""Dual judges for the P2.3 Quiz eval.

Local: qwen2.5:7b via ChatOllama using app.agent.judge.judge_response with the
       quiz rubric (QUIZ_DIMENSIONS) so scores are directly comparable to
       production judge_node behavior on quiz turns.

Cloud: BYOK MiniMax-M2.7 via OpenAI-compatible endpoint (api.minimaxi.com/v1).
       Reads MINIMAX_API_KEY from the environment; falls back to None if missing
       so the harness still runs on machines without cloud access (entries just
       record judge_cloud: {} for those rows).

Forked from p2_2_agent_ablation/judges.py — swapped PLAN_DIMENSIONS/load_plan_rubric
for QUIZ_DIMENSIONS/load_quiz_rubric.
"""
from __future__ import annotations

import json
import os
from typing import Any

from app.agent.judge import (
    QUIZ_DIMENSIONS,
    judge_response,
    load_quiz_rubric,
)


def make_local_judge(judge_llm) -> Any:
    """Return an async callable (question, answer_text) -> judge dict."""
    rubric = load_quiz_rubric()

    async def judge(question: str, answer_text: str) -> dict:
        result = await judge_response(
            question=question, answer=answer_text, context="",
            rubric=rubric, judge_llm=judge_llm,
            dimensions=QUIZ_DIMENSIONS,
        )
        return {
            "score": result["score"],
            "weak_dims": result["weak_dims"],
            "reasoning": result["reasoning"],
        }

    return judge


def make_cloud_judge(model_id: str = "MiniMax-M2.7"):
    """Return an async callable, or None if MINIMAX_API_KEY is not set.

    cloud-adapt: this is the eval-side BYOK via MiniMax's OpenAI-compatible
    endpoint; production path uses x-judge-model header. Swapping providers
    only needs base_url + env var name changed.
    """
    api_key = os.getenv("MINIMAX_API_KEY")
    if not api_key:
        return None
    from openai import AsyncOpenAI
    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.minimaxi.com/v1",
    )
    rubric = load_quiz_rubric()

    async def judge(question: str, answer_text: str) -> dict:
        prompt = rubric.format(question=question, answer=answer_text, context="")
        resp = await client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.01,  # MiniMax-M2.7 temperature range (0, 1]; 0.0 rejected
        )
        text = resp.choices[0].message.content or ""
        parsed: dict = {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            # Naive fallback: search for first {...}
            import re
            m = re.search(r"\{[\s\S]*\}", text)
            if m:
                try:
                    parsed = json.loads(m.group(0))
                except json.JSONDecodeError:
                    parsed = {}
        scores = []
        weak = []
        for dim in QUIZ_DIMENSIONS:
            v = parsed.get(dim, 3)
            try:
                fv = float(v)
            except (TypeError, ValueError):
                fv = 3.0
            fv = max(1.0, min(5.0, fv))
            scores.append(fv)
            if fv <= 3:
                weak.append(dim)
        avg = sum(scores) / len(scores) / 5.0
        return {
            "score": round(avg, 4),
            "weak_dims": weak,
            "reasoning": str(parsed.get("reasoning", ""))[:500],
            "model": model_id,
        }

    return judge
