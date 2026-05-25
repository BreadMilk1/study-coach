"""Retrieval eval runner — supports multiple retriever variants.

Per-variant results are saved as JSON under tests/fixtures/eval_results/, and
docs/RETRIEVAL_EVAL.md is regenerated from all results on each run so deltas
between variants are visible side-by-side.

Usage:
    cd backend && uv run python scripts/run_baseline_eval.py --variant baseline
    cd backend && uv run python scripts/run_baseline_eval.py --variant hybrid_bm25_rrf
    cd backend && uv run python scripts/run_baseline_eval.py --variant baseline --reindex
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

import chromadb  # noqa: E402

from app.eval.retrieval import EvalQuery, evaluate  # noqa: E402
from app.rag.document_processor import DocumentProcessor  # noqa: E402
from app.rag.embedder import OllamaEmbedder  # noqa: E402
from app.rag.hybrid_retriever import BM25Index, HybridRetriever  # noqa: E402
from app.rag.reranking_retriever import FastembedReranker, RerankingRetriever  # noqa: E402
from app.rag.retriever import Retriever  # noqa: E402

ROOT_DIR = BACKEND_DIR.parent.parent
CORPUS_DIR = ROOT_DIR / "HKBU_StudyCompanion 2" / "data" / "hkbu_corpus"
FIXTURE = BACKEND_DIR / "tests" / "fixtures" / "retrieval_eval_queries.json"
EVAL_CHROMA_PATH = BACKEND_DIR / "chroma_eval_data"
RESULTS_DIR = BACKEND_DIR / "tests" / "fixtures" / "eval_results"
REPORT_PATH = BACKEND_DIR.parent / "docs" / "RETRIEVAL_EVAL.md"

VARIANT_LABELS = {
    "baseline": "Baseline — Dense only (nomic-embed-text + Chroma)",
    "hybrid_bm25_rrf": "+B — Hybrid (BM25 + Dense, RRF k=60)",
    "hybrid_bm25_rrf_reranker": "+C — Hybrid + jina-reranker-v2 (depth=20→top-K)",
}
VARIANT_ORDER = ["baseline", "hybrid_bm25_rrf", "hybrid_bm25_rrf_reranker"]


def build_chroma_collection(reindex: bool):
    if reindex and EVAL_CHROMA_PATH.exists():
        print(f"Removing existing eval index at {EVAL_CHROMA_PATH}…")
        shutil.rmtree(EVAL_CHROMA_PATH)
    client = chromadb.PersistentClient(path=str(EVAL_CHROMA_PATH))
    return client.get_or_create_collection("eval_baseline")


def ingest_if_empty(collection, retriever) -> None:
    if collection.count() > 0:
        print(f"Using existing index: {collection.count()} chunks")
        return
    print(f"Indexing {CORPUS_DIR} into eval Chroma…")
    dp = DocumentProcessor()
    for pdf in sorted(CORPUS_DIR.glob("*.pdf")):
        chunks = dp.process_pdf(pdf)
        retriever.add_chunks(chunks)
        print(f"  {pdf.name}: +{len(chunks)} chunks "
              f"(running total: {collection.count()})")
    print(f"Total chunks indexed: {collection.count()}")


def load_chunks_from_chroma(collection) -> list[dict]:
    data = collection.get(include=["documents", "metadatas"])
    return [
        {
            "chunk_id": data["ids"][i],
            "content": data["documents"][i],
            "source": data["metadatas"][i].get("source", ""),
            "page": data["metadatas"][i].get("page", -1),
        }
        for i in range(len(data["ids"]))
    ]


def build_baseline_retriever(reindex: bool) -> Retriever:
    collection = build_chroma_collection(reindex)
    embedder = OllamaEmbedder(model="nomic-embed-text")
    retriever = Retriever(collection=collection, embedder=embedder)
    ingest_if_empty(collection, retriever)
    return retriever


def build_hybrid_retriever(reindex: bool) -> HybridRetriever:
    collection = build_chroma_collection(reindex)
    embedder = OllamaEmbedder(model="nomic-embed-text")
    dense = Retriever(collection=collection, embedder=embedder)
    bm25 = BM25Index()
    hybrid = HybridRetriever(dense=dense, bm25=bm25)

    if collection.count() == 0:
        ingest_if_empty(collection, hybrid)
    else:
        print(f"Using existing Chroma: {collection.count()} chunks")
        chunks = load_chunks_from_chroma(collection)
        bm25.add_chunks(chunks)
        print(f"BM25 (in-memory) rebuilt over {len(chunks)} chunks.")
    return hybrid


def build_reranked_retriever(reindex: bool) -> RerankingRetriever:
    hybrid = build_hybrid_retriever(reindex)
    reranker = FastembedReranker()
    return RerankingRetriever(base=hybrid, reranker=reranker, retrieval_depth=20)


def load_queries() -> list[EvalQuery]:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return [
        EvalQuery(
            query=item["query"],
            expected=item["expected"],
            classification=item["classification"],
        )
        for item in data
    ]


def save_results(variant: str, report, top_k: int) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "variant": variant,
        "label": VARIANT_LABELS.get(variant, variant),
        "top_k": top_k,
        "aggregate": report.aggregate,
        "per_query": report.per_query,
    }
    (RESULTS_DIR / f"{variant}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_all_results() -> list[dict]:
    if not RESULTS_DIR.exists():
        return []
    available = {f.stem: f for f in RESULTS_DIR.glob("*.json")}
    ordered = [available[v] for v in VARIANT_ORDER if v in available]
    return [json.loads(f.read_text(encoding="utf-8")) for f in ordered]


def regenerate_markdown(all_results: list[dict]) -> None:
    if not all_results:
        return
    lines = [
        "# Retrieval Evaluation",
        "",
        "Each variant evaluated against "
        "`backend/tests/fixtures/retrieval_eval_queries.json`",
        "(15 queries: 4 short_kw / 4 vocab_gap / 4 long_specific / 3 cross_lang).",
        "",
        "Hit Rate@5 = at least one expected (source, page) in top-5;  ",
        "MRR = 1 / rank of first hit (0 if no hit).",
        "",
        "## Comparison",
        "",
        "| Variant | top_k | Hit@5 | MRR |",
        "|---|---|---|---|",
    ]
    for r in all_results:
        lines.append(
            f"| {r['label']} | {r['top_k']} | "
            f"**{r['aggregate']['hit_rate@5']:.3f}** | "
            f"**{r['aggregate']['mrr']:.3f}** |"
        )
    lines.append("")

    for r in all_results:
        lines += [
            f"## {r['label']}",
            "",
            f"Top-K = {r['top_k']}.  Aggregate: "
            f"Hit@5 = **{r['aggregate']['hit_rate@5']:.3f}**, "
            f"MRR = **{r['aggregate']['mrr']:.3f}**.",
            "",
            "| # | Class | Query (truncated) | Hit Rank | Hit@5 | MRR |",
            "|---|---|---|---|---|---|",
        ]
        for i, q in enumerate(r["per_query"], 1):
            rank = q["hit_rank"] if q["hit_rank"] is not None else "—"
            truncated = q["query"][:55].replace("|", "\\|")
            lines.append(
                f"| {i} | {q['classification']} | {truncated} | {rank} | "
                f"{q['hit_rate@5']:.1f} | {q['mrr']:.3f} |"
            )
        lines.append("")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=list(VARIANT_LABELS.keys()),
        default="baseline",
    )
    parser.add_argument("--reindex", action="store_true")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    if args.variant == "baseline":
        retriever = build_baseline_retriever(args.reindex)
    elif args.variant == "hybrid_bm25_rrf":
        retriever = build_hybrid_retriever(args.reindex)
    else:
        retriever = build_reranked_retriever(args.reindex)

    queries = load_queries()
    print(f"\nRunning {len(queries)} queries with top_k={args.top_k}…")
    report = evaluate(retriever, queries, top_k=args.top_k)

    save_results(args.variant, report, args.top_k)
    regenerate_markdown(load_all_results())

    print(f"\nWrote {REPORT_PATH}\n")
    print(f"=== Variant: {args.variant} (top_k={args.top_k}) ===")
    print(f"  Hit Rate @ 5: {report.aggregate['hit_rate@5']:.3f}")
    print(f"  MRR        : {report.aggregate['mrr']:.3f}")
    print()
    for q in report.per_query:
        rank = q["hit_rank"] if q["hit_rank"] is not None else "—"
        truncated = q["query"][:50]
        print(f"  [{q['classification']:14}] rank={rank!s:>3} "
              f"hit@5={q['hit_rate@5']:.1f} mrr={q['mrr']:.3f}  {truncated}")


if __name__ == "__main__":
    main()
