import os
import re
from typing import List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

import config


# ---------------------------------------------------------------------------
# Heading / section patterns (Vietnamese legal documents)
# Chi giu lai Chuong va Dieu
# ---------------------------------------------------------------------------
SECTION_PATTERNS = [
    # Chuong - without diacritics
    (r"(?i)^(CHUONG\s+[IVXLCDM\d]+[.:]*\s*.*)", "chuong"),
    (r"(?i)^(Chuong\s+[IVXLCDM\d]+[.:]*\s*.*)", "chuong"),
    # Chuong - with diacritics
    (r"(?i)^(CH\u01af\u01a0NG\s+[IVXLCDM\d]+[.:]*\s*.*)", "chuong"),
    (r"(?i)^(Ch\u01b0\u01a1ng\s+[IVXLCDM\d]+[.:]*\s*.*)", "chuong"),
    # Dieu - without diacritics
    (r"(?i)^(Dieu\s+\d+[.:]*\s*.*)", "dieu"),
    (r"(?i)^(DIEU\s+\d+[.:]*\s*.*)", "dieu"),
    # Dieu - with diacritics
    (r"(?i)^(\u0110i\u1ec1u\s+\d+[.:]*\s*.*)", "dieu"),
    (r"(?i)^(\u0110I\u1ec0U\s+\d+[.:]*\s*.*)", "dieu"),
]

# Hierarchy: Chuong > Dieu only
SECTION_HIERARCHY = ["chuong", "dieu"]


class DocumentProcessor:
    """Parse PDF files and split into chunks with Chuong/Dieu metadata.

    Uses PyPDFLoader with mode='single' to load the entire PDF as one
    document, preventing content loss when sections span across pages.
    """

    def __init__(
        self,
        chunk_size: int = config.CHUNK_SIZE,
        chunk_overlap: int = config.CHUNK_OVERLAP,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            keep_separator=True,
        )

    # ------------------------------------------------------------------
    # File loading  (PDF only, mode="single")
    # ------------------------------------------------------------------
    def load_file(self, file_path: str) -> List[Document]:
        """Load a PDF file as a single Document (all pages merged)."""
        file_name = os.path.basename(file_path)
        loader = PyPDFLoader(file_path, mode="single")
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = file_name
        return docs

    def load_files(self, file_paths: List[str]) -> List[Document]:
        """Load multiple PDF files."""
        all_docs: List[Document] = []
        for fp in file_paths:
            try:
                docs = self.load_file(fp)
                all_docs.extend(docs)
            except Exception as e:
                print(f"[WARNING] Cannot load {fp}: {e}")
        return all_docs

    # ------------------------------------------------------------------
    # Section / heading detection
    # ------------------------------------------------------------------
    @staticmethod
    def _detect_section(line: str) -> Optional[tuple]:
        """Return (heading_text, section_level) if line matches a heading."""
        stripped = line.strip()
        if not stripped:
            return None
        for pattern, level in SECTION_PATTERNS:
            m = re.match(pattern, stripped)
            if m:
                return (m.group(1).strip(), level)
        return None

    @staticmethod
    def _build_section_string(hierarchy: dict) -> str:
        """Build a readable section string from the heading hierarchy."""
        parts = []
        for level in SECTION_HIERARCHY:
            if level in hierarchy and hierarchy[level]:
                parts.append(hierarchy[level])
        return " > ".join(parts) if parts else ""

    # ------------------------------------------------------------------
    # Metadata-aware chunking
    # ------------------------------------------------------------------
    def _annotate_documents(self, docs: List[Document]) -> List[Document]:
        """Walk through the single-document text line-by-line, detect
        Chuong/Dieu headings, and create segments with section metadata."""
        annotated: List[Document] = []
        current_hierarchy: dict = {}

        for doc in docs:
            lines = doc.page_content.split("\n")
            current_segment_lines: List[str] = []

            for line in lines:
                detection = self._detect_section(line)
                if detection:
                    heading_text, level = detection
                    # Flush current segment
                    if current_segment_lines:
                        section_str = self._build_section_string(current_hierarchy)
                        meta = dict(doc.metadata)
                        meta["section"] = section_str
                        text = "\n".join(current_segment_lines).strip()
                        if text:
                            annotated.append(
                                Document(page_content=text, metadata=meta)
                            )
                        current_segment_lines = []

                    # Update hierarchy
                    current_hierarchy[level] = heading_text
                    lvl_idx = SECTION_HIERARCHY.index(level)
                    for lower in SECTION_HIERARCHY[lvl_idx + 1 :]:
                        current_hierarchy.pop(lower, None)

                current_segment_lines.append(line)

            # Flush remaining
            if current_segment_lines:
                section_str = self._build_section_string(current_hierarchy)
                meta = dict(doc.metadata)
                meta["section"] = section_str
                text = "\n".join(current_segment_lines).strip()
                if text:
                    annotated.append(Document(page_content=text, metadata=meta))

        return annotated

    def process_documents(self, docs: List[Document]) -> List[Document]:
        """Full pipeline: annotate with Chuong/Dieu metadata, then split."""
        annotated = self._annotate_documents(docs)

        all_chunks: List[Document] = []
        for doc in annotated:
            chunks = self.text_splitter.split_documents([doc])
            for chunk in chunks:
                chunk.metadata.setdefault("section", doc.metadata.get("section", ""))
                chunk.metadata.setdefault("source", doc.metadata.get("source", ""))
            all_chunks.extend(chunks)

        return all_chunks

    def load_and_process(self, file_paths: List[str]) -> List[Document]:
        """Convenience: load PDF files then process into chunks with metadata."""
        docs = self.load_files(file_paths)
        return self.process_documents(docs)
