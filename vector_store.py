from typing import List, Optional

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

import config


class VectorStoreManager:
    """Manage ChromaDB vector store and hybrid retrieval (BM25 + semantic)."""

    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL,
            openai_api_key=config.OPENAI_API_KEY,
            openai_api_base=config.BASE_URL,
        )
        self.vectorstore: Optional[Chroma] = None
        self.bm25_retriever: Optional[BM25Retriever] = None
        self.ensemble_retriever: Optional[EnsembleRetriever] = None
        self._documents: List[Document] = []

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to both ChromaDB and BM25 index."""
        if not documents:
            return

        self._documents.extend(documents)

        # --- ChromaDB (semantic) ---
        if self.vectorstore is None:
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=config.CHROMA_PERSIST_DIR,
                collection_name=config.CHROMA_COLLECTION_NAME,
            )
        else:
            self.vectorstore.add_documents(documents)

        # --- BM25 (keyword) ---
        self.bm25_retriever = BM25Retriever.from_documents(
            self._documents,
            k=config.RETRIEVAL_TOP_K,
        )

        # --- Ensemble (RRF-based fusion) ---
        semantic_retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": config.RETRIEVAL_TOP_K}
        )
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, semantic_retriever],
            weights=[config.BM25_WEIGHT, config.SEMANTIC_WEIGHT],
        )

    def get_retriever(self) -> Optional[EnsembleRetriever]:
        """Return the hybrid ensemble retriever."""
        return self.ensemble_retriever

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def clear(self) -> None:
        """Reset all indices."""
        if self.vectorstore is not None:
            try:
                self.vectorstore.delete_collection()
            except Exception:
                pass
        self.vectorstore = None
        self.bm25_retriever = None
        self.ensemble_retriever = None
        self._documents = []

    @property
    def document_count(self) -> int:
        return len(self._documents)

    @property
    def is_ready(self) -> bool:
        return self.ensemble_retriever is not None
