"""
Multi-Layer Legal Query Router (binary RAG detection).

The router answers ONE question: "Does this query need document retrieval?"

Layers execute in waterfall order.  If ANY layer detects the query needs
RAG, the router immediately returns ``legal_rag``.  Only when ALL layers
find no signal does the router fall back to ``no_retrieval``.

Layer order:
  0. Too-short / vague check          →  needs_clarification
  1. Keyword matching                 →  legal_rag (if hit)
  2. Regex legal structure detection   →  legal_rag (if hit)
  3. Semantic similarity (embeddings)  →  legal_rag (if score ≥ threshold)
  4. LLM classifier (fallback)        →  legal_rag | no_retrieval
"""

import json
import re
from dataclasses import dataclass
from typing import List, Literal, Optional

import httpx
import numpy as np
from langchain_openai import OpenAIEmbeddings

import config

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
RouteType = Literal["legal_rag", "no_retrieval", "needs_clarification"]


@dataclass
class RouterDecision:
    """Result of the query router."""
    route: RouteType
    reason: str


# ---------------------------------------------------------------------------
# Regex patterns for Vietnamese legal references
# ---------------------------------------------------------------------------
LEGAL_STRUCTURE_RE = re.compile(
    r"(Điều\s?\d+|Khoản\s?\d+|Điều\s?\d+\s*Khoản\s?\d+"
    r"|Nghị định số|Thông tư số|Luật số"
    r"|Dieu\s?\d+|Khoan\s?\d+)",
    re.IGNORECASE,
)

LEGAL_NUMBER_RE = re.compile(
    r"(luật|nghị định|thông tư|pháp lệnh|bộ luật)\s*\d+",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# LLM classifier prompt
# ---------------------------------------------------------------------------
LLM_ROUTER_PROMPT = """Ban la chuyen gia phan loai cau hoi phap luat Viet Nam.
Nhiem vu: xac dinh cau hoi sau co CAN TRA CUU TAI LIEU hay khong.

Query: "{query}"

Quy tac:
- Neu cau hoi lien quan den luat, nghi dinh, thong tu, noi quy, quy dinh,
  hop dong, quyen loi lao dong, hoac bat ky van ban phap luat nao -> can tra cuu.
- Neu cau hoi la kien thuc chung, cau chao hoi, hoac khong lien quan den
  tai lieu cu the -> khong can tra cuu.

Tra ve dung 1 JSON (khong co gi khac):
{{"route": "legal_rag" hoac "no_retrieval", "reason": "..."}}"""


# ============================================================================
# LegalQueryRouter
# ============================================================================
class LegalQueryRouter:
    """Binary query router: decides if a query needs RAG retrieval or not.

    If ANY layer detects a RAG signal, the query goes to the RAG pipeline.
    Only when all layers find nothing does the router output ``no_retrieval``.
    """

    def __init__(self, llm_url: Optional[str] = None):
        self.llm_url = llm_url or f"{config.LLM_SERVICE_URL}/api/chat"
        self.keywords: List[str] = config.RAG_KEYWORDS
        self.threshold: float = config.SEMANTIC_SIMILARITY_THRESHOLD

        # --- Pre-compute embeddings for example RAG questions ---
        self.embeddings = OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL,
            openai_api_key=config.OPENAI_API_KEY,
            openai_api_base=config.BASE_URL,
        )
        self._example_vectors: Optional[np.ndarray] = None
        self._init_example_embeddings()

    def _init_example_embeddings(self) -> None:
        """Embed the RAG example questions from config at startup."""
        examples = config.RAG_EXAMPLE_QUESTIONS
        if not examples:
            return
        try:
            vectors = self.embeddings.embed_documents(examples)
            self._example_vectors = np.array(vectors)  # shape (N, dim)
        except Exception as e:
            print(f"[WARNING] Could not embed RAG examples: {e}")
            self._example_vectors = None

    # ---- Layer 0: Too-short check -----------------------------------------
    def _check_too_short(self, query: str) -> Optional[RouterDecision]:
        stripped = query.strip()
        if len(stripped) < 5 or len(stripped.split()) < 2:
            return RouterDecision(
                route="needs_clarification",
                reason="Query qua ngan / mo ho",
            )
        return None

    # ---- Layer 1: Keyword matching ----------------------------------------
    def _layer_keyword(self, query: str) -> Optional[RouterDecision]:
        q = query.lower()
        for kw in self.keywords:
            if kw in q:
                return RouterDecision(
                    route="legal_rag",
                    reason=f"Keyword match: '{kw}'",
                )
        return None

    # ---- Layer 2: Regex legal patterns ------------------------------------
    def _layer_regex(self, query: str) -> Optional[RouterDecision]:
        if LEGAL_STRUCTURE_RE.search(query):
            return RouterDecision(
                route="legal_rag",
                reason="Regex: cau truc phap ly (Dieu/Khoan/Nghi dinh so...)",
            )
        if LEGAL_NUMBER_RE.search(query):
            return RouterDecision(
                route="legal_rag",
                reason="Regex: tham chieu luat co so hieu",
            )
        return None

    # ---- Layer 3: Semantic similarity -------------------------------------
    def _layer_semantic(self, query: str) -> Optional[RouterDecision]:
        """Compare query embedding against RAG example embeddings."""
        if self._example_vectors is None:
            return None

        try:
            query_vec = np.array(self.embeddings.embed_query(query))  # (dim,)
            # Cosine similarity against all examples
            # dot / (norm_a * norm_b)
            dot = self._example_vectors @ query_vec
            norms = np.linalg.norm(self._example_vectors, axis=1) * np.linalg.norm(query_vec)
            similarities = dot / (norms + 1e-10)
            max_sim = float(similarities.max())

            if max_sim >= self.threshold:
                return RouterDecision(
                    route="legal_rag",
                    reason=f"Semantic similarity: {max_sim:.2f} >= {self.threshold}",
                )
        except Exception as e:
            print(f"[WARNING] Semantic layer error: {e}")
        return None

    # ---- Layer 4: LLM classifier (fallback) -------------------------------
    def _layer_llm(self, query: str) -> RouterDecision:
        prompt = LLM_ROUTER_PROMPT.format(query=query)
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
        }

        try:
            response = httpx.post(self.llm_url, json=payload, timeout=30.0)
            response.raise_for_status()
            content = response.json().get("content", "")

            # Clean potential markdown code block wrapper
            cleaned = content.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```\w*\n?", "", cleaned)
                cleaned = re.sub(r"\n?```$", "", cleaned)
                cleaned = cleaned.strip()

            data = json.loads(cleaned)
            route = data.get("route", "legal_rag")
            if route not in ("legal_rag", "no_retrieval"):
                route = "legal_rag"

            return RouterDecision(
                route=route,
                reason=f"LLM classifier: {data.get('reason', 'N/A')}",
            )
        except Exception as e:
            # If LLM fails, default to RAG (safest)
            return RouterDecision(
                route="legal_rag",
                reason=f"LLM fallback (error: {e})",
            )

    # ======== Main route method ============================================
    def route(self, query: str) -> RouterDecision:
        """Classify a query: does it need RAG retrieval?

        Layer order:
          0. Too-short check → needs_clarification
          1. Keyword match   → legal_rag
          2. Regex patterns  → legal_rag
          3. Semantic sim    → legal_rag
          4. LLM classifier  → legal_rag | no_retrieval
        """
        # Layer 0
        decision = self._check_too_short(query)
        if decision:
            return decision

        # Layer 1
        decision = self._layer_keyword(query)
        if decision:
            return decision

        # Layer 2
        decision = self._layer_regex(query)
        if decision:
            return decision

        # Layer 3
        decision = self._layer_semantic(query)
        if decision:
            return decision

        # Layer 4 (final — LLM decides)
        return self._layer_llm(query)
