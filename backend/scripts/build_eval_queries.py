"""One-shot helper: build retrieval-eval ground-truth fixture.

Combines HKBU's hyde_test_queries.json (12 queries, 3 classes) with three
manually-curated cross-language queries that directly probe the
Chinese-question / English-PDF mismatch the user reported in Phase 1.

Auto-labels each query's expected (source, pages) by keyword grep over
the HKBU corpus PDFs. The output is a starter set — human review of the
_grep_keywords field is expected before the fixture is trusted for eval.

Run once:
    cd backend && uv run python scripts/build_eval_queries.py

Output: tests/fixtures/retrieval_eval_queries.json
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader

BACKEND_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = BACKEND_DIR.parent.parent
CORPUS_DIR = ROOT_DIR / "HKBU_StudyCompanion 2" / "data" / "hkbu_corpus"
HKBU_QUERIES = (
    ROOT_DIR
    / "HKBU_StudyCompanion 2"
    / "reports"
    / "project_report"
    / "hyde_test_queries.json"
)
OUTPUT = BACKEND_DIR / "tests" / "fixtures" / "retrieval_eval_queries.json"


HKBU_KEYWORD_HINTS: dict[str, list[str]] = {
    # short_kw — stems catch word-form variation
    "hallucinations": ["hallucinat"],
    "token budget": ["token budget"],
    "ReAct": ["ReAct"],
    "temperature": ["temperature"],
    # vocab_gap — dominant noun phrase, not the high-frequency filler words
    "What is prompt engineering and why is it important?": ["prompt engineering"],
    "How does a large language model work?": ["large language model"],
    "What is context engineering and how does it improve LLM applications?": [
        "context engineering"
    ],
    "What are the key methods for evaluating LLM applications?": [
        r"evaluat\w*",
        r"LLM applications?",
    ],
    # long_specific — pull the most distinctive proper term from each
    "Can you explain the autoregressive nature of text generation and what specific limitations it has regarding error correction?": [
        "autoregressive"
    ],
    "According to the text, what are the first three steps in the Cycle of Quality?": [
        "Cycle of Quality"
    ],
    "What are the different tiers of importance listed for prioritizing items within the Token Budget?": [
        r"tiers of importance|Token Budget"
    ],
    "Given that IT support assistance in this scenario involves a voice-over-the-phone application and is constrained by available documentation, what type of problem is the application *not* designed to handle?": [
        "IT support",
        r"voice[- ]over[- ]the[- ]phone",
    ],
}


MANUAL_QUERIES: list[dict] = [
    {
        "query": "什么是 prompt engineering 提示工程?",
        "class": "cross_lang",
        "_kw": ["prompt engineering"],
    },
    {
        "query": "解释少样本学习 few-shot learning",
        "class": "cross_lang",
        "_kw": ["few-shot", "few shot"],
    },
    {
        "query": "如何评估 LLM 应用?",
        "class": "cross_lang",
        "_kw": [r"evaluat\w*", r"LLM applications?"],
    },
]


def load_corpus() -> dict[str, list[dict]]:
    corpus: dict[str, list[dict]] = {}
    for pdf_path in sorted(CORPUS_DIR.glob("*.pdf")):
        loader = PyPDFLoader(str(pdf_path))
        pages = []
        for doc in loader.load():
            raw = doc.metadata.get("page", -1)
            page = raw + 1 if isinstance(raw, int) and raw >= 0 else -1
            pages.append({"page": page, "text": doc.page_content or ""})
        corpus[pdf_path.name] = pages
    return corpus


def grep_pages(corpus: dict, pattern: str) -> dict[str, set[int]]:
    flags = re.IGNORECASE
    result: dict[str, set[int]] = defaultdict(set)
    for source, pages in corpus.items():
        for p in pages:
            if re.search(pattern, p["text"], flags):
                result[source].add(p["page"])
    return result


def extract_keywords(
    query: str, classification: str, hint: list[str] | None = None
) -> list[str]:
    if hint:
        return hint
    if classification == "short_kw":
        return [re.escape(query.strip())]
    # vocab_gap & long_specific: prefer capitalized multi-word phrases,
    # fall back to longest content words.
    phrases = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", query)
    words = [w for w in re.findall(r"\b[A-Za-z]+\b", query) if len(w) >= 5]
    seen: set[str] = set()
    out: list[str] = []
    for p in phrases + sorted(words, key=len, reverse=True):
        if p.lower() in seen:
            continue
        seen.add(p.lower())
        out.append(re.escape(p))
        if len(out) >= 3:
            break
    return out


def main() -> None:
    if not CORPUS_DIR.exists():
        raise SystemExit(f"Corpus dir not found: {CORPUS_DIR}")
    if not HKBU_QUERIES.exists():
        raise SystemExit(f"HKBU queries not found: {HKBU_QUERIES}")

    print(f"Loading corpus from {CORPUS_DIR}…")
    corpus = load_corpus()
    total_pages = sum(len(p) for p in corpus.values())
    print(f"  {len(corpus)} PDFs, {total_pages} pages total.\n")

    hkbu_raw = json.loads(HKBU_QUERIES.read_text(encoding="utf-8"))
    hkbu = [
        {**q, "_kw": HKBU_KEYWORD_HINTS[q["query"]]}
        if q["query"] in HKBU_KEYWORD_HINTS
        else q
        for q in hkbu_raw
    ]
    all_queries = hkbu + MANUAL_QUERIES

    output = []
    for q in all_queries:
        kws = extract_keywords(q["query"], q.get("class", ""), q.get("_kw"))
        merged: dict[str, set[int]] = defaultdict(set)
        for kw in kws:
            for source, pages in grep_pages(corpus, kw).items():
                merged[source].update(pages)

        ranked = [
            {"source": s, "pages": sorted(p), "_match_count": len(p)}
            for s, p in merged.items()
        ]
        ranked.sort(key=lambda e: e["_match_count"], reverse=True)
        expected = [
            {"source": e["source"], "pages": e["pages"]} for e in ranked[:3]
        ]

        output.append(
            {
                "query": q["query"],
                "classification": q.get("class", ""),
                "expected": expected,
                "_grep_keywords": kws,
                "_candidates_truncated": len(ranked) > 3,
            }
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote {OUTPUT}: {len(output)} queries\n")
    for o in output:
        n_exp = len(o["expected"])
        first = o["expected"][0]["source"][:35] if o["expected"] else "—"
        print(
            f"  [{o['classification']:14}] {o['query'][:55]:55}"
            f" → {n_exp} sources (top: {first})"
        )


if __name__ == "__main__":
    main()
