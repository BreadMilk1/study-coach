"""Phase B smoke — verify ChatOllama(reasoning=False) is fast on qwen3.5:4b.

This is the gate-check before running the P2.3 matrix (Cut ②b). P2.2 Cut ①f
discovered that ChatOllama(reasoning=False) is the only kwarg that forwards
to the Ollama API `think` field; without it, qwen3.5:4b takes ~813s per call
(thinking-ON), which would make the 396-record matrix take ~30 hours instead
of ~5 hours.

Expected output: elapsed < 15s (typical 5-10s on 16GB Apple Silicon Mac).

If elapsed > 30s, either:
  - Ollama isn't responding (check `ollama list` + `ollama serve`)
  - langchain-ollama upgraded and renamed the kwarg
  - the model needs to be re-pulled

In any of those cases, STOP and investigate before running Cut ②b. A 30x
regression here means the matrix takes ~30h instead of ~5h.

Usage:
    cd backend && uv run python scripts/smoke_quiz_reasoning_off.py
"""
import asyncio
import sys
import time

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama


async def main() -> int:
    print("[smoke] qwen3.5:4b reasoning=False — single prompt", flush=True)

    llm = ChatOllama(
        model="qwen3.5:4b",
        temperature=0.7,
        reasoning=False,  # P2.2 critical finding — the only kwarg that forwards `think=False`
    )

    t0 = time.monotonic()
    try:
        response = await llm.ainvoke(
            [HumanMessage(content="What is 2+2? Answer in one word.")]
        )
    except Exception as exc:
        print(f"[smoke] FAIL — LLM call raised: {type(exc).__name__}: {exc}")
        print("[smoke] Is Ollama running? `ollama serve` + `ollama list`")
        return 2

    elapsed = time.monotonic() - t0

    raw = getattr(response, "content", "") or ""
    if not isinstance(raw, str):
        raw = str(raw)

    print(f"[smoke] elapsed: {elapsed:.1f}s")
    print(f"[smoke] response: {raw[:120]!r}")

    if elapsed >= 30.0:
        print(
            f"[smoke] FAIL — reasoning=False mechanism appears BROKEN. "
            f"Expected <15s, got {elapsed:.1f}s. P2.2's 32× speedup has "
            f"regressed. Do NOT proceed to Cut ②b matrix run. Investigate:"
        )
        print("  1. langchain-ollama version: `uv pip show langchain-ollama`")
        print("  2. qwen3.5:4b model pulled: `ollama list | grep qwen3.5`")
        print("  3. raw-API test: `ollama run qwen3.5:4b --think=false 'hi'`")
        return 1

    print(f"[smoke] OK — reasoning=False mechanism holds (elapsed {elapsed:.1f}s < 30s gate)")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
