from pathlib import Path

from app.rag.document_processor import DocumentProcessor


def test_chunk_text_produces_chunks_no_larger_than_chunk_size():
    dp = DocumentProcessor(chunk_size=100, chunk_overlap=20)
    text = ("Sentence number X. " * 60).strip()  # ~1140 chars

    chunks = dp.chunk_text(text, source="lec1.pdf", page=1)

    assert len(chunks) >= 2
    assert all(len(c["content"]) <= 100 for c in chunks)


def test_chunk_id_is_deterministic_and_unique_within_a_page():
    dp = DocumentProcessor(chunk_size=100, chunk_overlap=20)
    text = ("Sentence number X. " * 60).strip()

    a = dp.chunk_text(text, source="lec1.pdf", page=1)
    b = dp.chunk_text(text, source="lec1.pdf", page=1)

    ids_a = [c["chunk_id"] for c in a]
    assert ids_a == [c["chunk_id"] for c in b]      # deterministic
    assert len(set(ids_a)) == len(ids_a)             # unique within page


def test_drops_chunks_shorter_than_min_length():
    dp = DocumentProcessor(chunk_size=100, chunk_overlap=0, min_chunk_length=20)
    text = "tiny.\n\nThis is a normal-sized chunk of text content over twenty chars."

    chunks = dp.chunk_text(text, source="t.pdf", page=1)

    assert chunks
    assert all(len(c["content"]) >= 20 for c in chunks)


def test_process_pdf_chunks_across_pages_and_preserves_filename(monkeypatch):
    dp = DocumentProcessor(chunk_size=100, chunk_overlap=0, min_chunk_length=20)
    fake_pages = [
        (1, "Page one talks about prompt engineering and retrieval augmented generation."),
        (2, "Page two discusses HyDE which rewrites queries before embedding lookup."),
    ]
    monkeypatch.setattr(dp, "_load_pdf_pages", lambda p: fake_pages)

    chunks = dp.process_pdf(Path("/fake/lec_hyde.pdf"))

    assert {c["page"] for c in chunks} == {1, 2}
    assert all(c["source"] == "lec_hyde.pdf" for c in chunks)
    assert all(":" in c["chunk_id"] for c in chunks)


def test_process_pdf_concatenates_pages_so_cross_page_concept_survives(monkeypatch):
    """A concept that ends on page 1 and continues on page 2 should appear in
    a single chunk after splitting, instead of being torn apart at the page
    boundary by per-page splitting."""
    dp = DocumentProcessor(chunk_size=200, chunk_overlap=20, min_chunk_length=20)
    fake_pages = [
        (1, "The Cycle of Quality is a five-step framework"),
        (2, " starting with retrieval. The first step is data ingest."),
    ]
    monkeypatch.setattr(dp, "_load_pdf_pages", lambda p: fake_pages)

    chunks = dp.process_pdf(Path("/fake/topic4.pdf"))

    has_combined = any(
        "Cycle of Quality" in c["content"]
        and "starting with retrieval" in c["content"]
        for c in chunks
    )
    assert has_combined, (
        "cross-page concept must survive in a single chunk after concat-then-split"
    )
