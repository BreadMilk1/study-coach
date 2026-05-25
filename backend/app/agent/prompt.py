"""Shared prompt construction for the Tutor RAG flow.

Used by both:
- `agent/graph.py` retrieve_and_answer node (sync LLM.invoke path)
- `api/routes.py` chat SSE handler (async LLM.astream path)

Keeping prompt + citation shape in one place ensures graph.py and routes.py
stay in sync. Future P2.1+ multi-node graph nodes can compose this with their
own system instructions.
"""

SYSTEM_INSTRUCTION = (
    "You are a study coach answering questions based ONLY on the provided sources. "
    "Cite each fact you use with [N] referring to the source list. "
    "If the sources do not contain the answer, say you don't know — do not fabricate."
)


def format_context(chunks: list[dict]) -> str:
    if not chunks:
        return "(no relevant sources retrieved)"
    return "\n\n".join(
        f"[{i + 1}] {c['source']} p.{c['page']}: {c['content']}"
        for i, c in enumerate(chunks)
    )


def build_prompt(query: str, chunks: list[dict]) -> str:
    return (
        f"{SYSTEM_INSTRUCTION}\n\n"
        f"Sources:\n{format_context(chunks)}\n\n"
        f"Question: {query}"
    )


def build_citations(chunks: list[dict]) -> list[dict]:
    return [
        {
            "chunk_id": c["chunk_id"],
            "source": c["source"],
            "page": c["page"],
            "span_start": 0,
            "span_end": len(c["content"]),
        }
        for c in chunks
    ]
