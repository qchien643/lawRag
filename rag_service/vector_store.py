from typing import List, Optional, Set
import uuid

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchAny,
)

import config


class VectorStoreManager:
    """Manage Qdrant vector store and hybrid retrieval (BM25 + semantic)."""

    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL,
            openai_api_key=config.OPENAI_API_KEY,
            openai_api_base=config.BASE_URL,
        )
        self.client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
        self.collection_name = config.QDRANT_COLLECTION_NAME
        self._documents: List[Document] = []
        self._source_files: Set[str] = set()
        self._vector_size: Optional[int] = None
        self.bm25_retriever: Optional[BM25Retriever] = None

        # Ensure collection exists
        self._ensure_collection()

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------
    def _ensure_collection(self) -> None:
        """Create collection if it doesn't exist, or load existing data."""
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name in collections:
            # Load existing source files from Qdrant
            self._load_existing_sources()
        # Collection will be created on first add_documents if it doesn't exist

    def _load_existing_sources(self) -> None:
        """Scan Qdrant collection to find all unique source file names."""
        try:
            # Scroll through all points to get unique sources
            offset = None
            while True:
                result = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                points, offset = result
                for point in points:
                    source = point.payload.get("source", "")
                    if source:
                        self._source_files.add(source)
                if offset is None:
                    break
            # Get collection info for vector size
            info = self.client.get_collection(self.collection_name)
            vec_config = info.config.params.vectors
            if hasattr(vec_config, "size"):
                self._vector_size = vec_config.size
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to Qdrant and BM25 index."""
        if not documents:
            return

        # Get embeddings
        texts = [doc.page_content for doc in documents]
        vectors = self.embeddings.embed_documents(texts)

        if self._vector_size is None:
            self._vector_size = len(vectors[0])

        # Ensure collection exists with correct vector size
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self._vector_size,
                    distance=Distance.COSINE,
                ),
            )

        # Build points
        points = []
        for doc, vector in zip(documents, vectors):
            point_id = str(uuid.uuid4())
            payload = {
                "page_content": doc.page_content,
                "source": doc.metadata.get("source", ""),
                "section": doc.metadata.get("section", ""),
                "page": doc.metadata.get("page", ""),
            }
            points.append(PointStruct(id=point_id, vector=vector, payload=payload))
            source = doc.metadata.get("source", "")
            if source:
                self._source_files.add(source)

        # Upsert in batches of 100
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
            )

        self._documents.extend(documents)

        # Rebuild BM25
        self._rebuild_bm25()

    def _rebuild_bm25(self) -> None:
        """Rebuild BM25 index from all documents in memory."""
        if self._documents:
            self.bm25_retriever = BM25Retriever.from_documents(
                self._documents,
                k=config.RETRIEVAL_TOP_K,
            )

    def get_retriever(
        self, source_filter: Optional[List[str]] = None
    ) -> Optional[EnsembleRetriever]:
        """Return a hybrid ensemble retriever, optionally filtered by source files."""
        if not self._source_files:
            return None

        # Build Qdrant filter
        qdrant_filter = None
        if source_filter:
            qdrant_filter = Filter(
                must=[
                    FieldCondition(
                        key="source",
                        match=MatchAny(any=source_filter),
                    )
                ]
            )

        # Create a custom retriever wrapper for Qdrant
        qdrant_retriever = QdrantRetriever(
            client=self.client,
            collection_name=self.collection_name,
            embeddings=self.embeddings,
            top_k=config.RETRIEVAL_TOP_K,
            qdrant_filter=qdrant_filter,
        )

        # Filter BM25 documents if needed
        if source_filter and self._documents:
            filtered_docs = [
                d for d in self._documents if d.metadata.get("source") in source_filter
            ]
            if filtered_docs:
                bm25 = BM25Retriever.from_documents(filtered_docs, k=config.RETRIEVAL_TOP_K)
            else:
                return qdrant_retriever  # Fallback to semantic only
        elif self._documents:
            bm25 = BM25Retriever.from_documents(self._documents, k=config.RETRIEVAL_TOP_K)
        else:
            return qdrant_retriever  # Fallback to semantic only

        return EnsembleRetriever(
            retrievers=[bm25, qdrant_retriever],
            weights=[config.BM25_WEIGHT, config.SEMANTIC_WEIGHT],
        )

    # ------------------------------------------------------------------
    # File management
    # ------------------------------------------------------------------
    def get_source_files(self) -> List[str]:
        """Return list of all unique source file names in the collection."""
        return sorted(self._source_files)

    def get_file_chunk_count(self, source: str) -> int:
        """Count chunks for a specific source file."""
        try:
            result = self.client.count(
                collection_name=self.collection_name,
                count_filter=Filter(
                    must=[FieldCondition(key="source", match=MatchAny(any=[source]))]
                ),
                exact=True,
            )
            return result.count
        except Exception:
            return 0

    def delete_file(self, source: str) -> bool:
        """Delete all chunks for a specific source file."""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[FieldCondition(key="source", match=MatchAny(any=[source]))]
                ),
            )
            self._source_files.discard(source)
            self._documents = [
                d for d in self._documents if d.metadata.get("source") != source
            ]
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def clear(self) -> None:
        """Reset all indices."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self._documents = []
        self._source_files = set()

    @property
    def document_count(self) -> int:
        try:
            info = self.client.get_collection(self.collection_name)
            return info.points_count
        except Exception:
            return 0

    @property
    def is_ready(self) -> bool:
        return len(self._source_files) > 0


# ---------------------------------------------------------------------------
# Custom Qdrant retriever compatible with LangChain
# ---------------------------------------------------------------------------
from langchain_core.retrievers import BaseRetriever
from pydantic import PrivateAttr


class QdrantRetriever(BaseRetriever):
    """A LangChain-compatible retriever that queries Qdrant directly."""

    # Public fields for Pydantic v2
    collection_name: str
    top_k: int = 6

    # Private attributes
    _client: QdrantClient = PrivateAttr()
    _embeddings: OpenAIEmbeddings = PrivateAttr()
    _qdrant_filter: Optional[Filter] = PrivateAttr(default=None)

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        embeddings: OpenAIEmbeddings,
        top_k: int = 6,
        qdrant_filter: Optional[Filter] = None,
        **kwargs,
    ):
        super().__init__(collection_name=collection_name, top_k=top_k, **kwargs)
        self._client = client
        self._embeddings = embeddings
        self._qdrant_filter = qdrant_filter

    def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        query_vector = self._embeddings.embed_query(query)
        results = self._client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=self.top_k,
            query_filter=self._qdrant_filter,
            with_payload=True,
        )
        docs = []
        for point in results.points:
            payload = point.payload or {}
            doc = Document(
                page_content=payload.get("page_content", ""),
                metadata={
                    "source": payload.get("source", ""),
                    "section": payload.get("section", ""),
                    "page": payload.get("page", ""),
                },
            )
            docs.append(doc)
        return docs
