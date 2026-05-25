from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentProcessor:
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_length: int = 20,
    ):
        self.min_chunk_length = min_chunk_length
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def chunk_text(self, text: str, source: str, page: int) -> list[dict]:
        chunks: list[dict] = []
        for piece in self._splitter.split_text(text):
            stripped = piece.strip()
            if len(stripped) < self.min_chunk_length:
                continue
            chunks.append({
                "content": stripped,
                "source": source,
                "page": page,
                "chunk_id": f"{source}:{page}:{len(chunks)}",
            })
        return chunks

    def _load_pdf_pages(self, path: Path) -> list[tuple[int, str]]:
        loader = PyPDFLoader(str(path))
        pages: list[tuple[int, str]] = []
        for doc in loader.load():
            raw_page = doc.metadata.get("page", -1)
            page = raw_page + 1 if isinstance(raw_page, int) and raw_page >= 0 else -1
            text = (doc.page_content or "").strip()
            if text:
                pages.append((page, text))
        return pages

    def process_pdf(self, path: str | Path) -> list[dict]:
        """Concat all PDF pages with a separator, split the full text, then
        map each chunk back to its origin page based on character offset.
        This prevents per-page splitting from tearing cross-page concepts."""
        path = Path(path)
        source = path.name
        pages = self._load_pdf_pages(path)
        if not pages:
            return []

        joiner = "\n\n"
        full_text = ""
        page_spans: list[tuple[int, int, int]] = []
        for page, text in pages:
            start = len(full_text)
            full_text += text + joiner
            page_spans.append((page, start, len(full_text)))

        chunks: list[dict] = []
        cursor = 0
        for piece in self._splitter.split_text(full_text):
            stripped = piece.strip()
            if len(stripped) < self.min_chunk_length:
                continue
            origin = full_text.find(piece, cursor)
            if origin < 0:
                origin = cursor
            cursor = origin + 1
            page = self._page_for_offset(origin, page_spans)
            chunks.append({
                "content": stripped,
                "source": source,
                "page": page,
                "chunk_id": f"{source}:{page}:{len(chunks)}",
            })
        return chunks

    @staticmethod
    def _page_for_offset(offset: int, spans: list[tuple[int, int, int]]) -> int:
        for page, start, end in spans:
            if start <= offset < end:
                return page
        return spans[-1][0] if spans else -1
