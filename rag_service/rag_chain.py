"""
RAG Chain with integrated Query Router.

Routes queries through a multi-layer classifier:
  - legal_rag          → retrieve docs + LLM answer with context
  - no_retrieval       → LLM answers directly (general knowledge)
  - needs_clarification→ static message asking user for more detail
"""

from dataclasses import dataclass
from typing import Optional, List

import httpx
from langchain_core.documents import Document

import config
from query_router import LegalQueryRouter, RouterDecision
from vector_store import VectorStoreManager


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class RAGResult:
    """Full result from the RAG chain, including routing metadata."""

    answer: str
    route: str
    route_reason: str


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_RAG = """Ban la tro ly AI chuyen tra loi cau hoi dua tren tai lieu duoc cung cap.

Quy tac:
1. Chi tra loi dua tren noi dung trong phan "Noi dung tham khao" ben duoi.
2. Neu khong tim thay thong tin lien quan, hay tra loi: "Toi khong tim thay thong tin lien quan trong tai lieu duoc cung cap."
3. Tra loi chi tiet, chinh xac va day du.
4. LUON LUON trich dan nguon o cuoi cau tra loi theo dinh dang:
   ---
   Nguon tham khao:
   - [Ten file] | Muc: [ten muc/phan] | Trang: [so trang neu co]
5. Neu nhieu nguon, liet ke tat ca cac nguon da su dung.
"""

HUMAN_PROMPT_RAG = """Noi dung tham khao:
{context}

Cau hoi: {question}

Hay tra loi cau hoi tren dua vao noi dung tham khao. Nho trich dan nguon."""


SYSTEM_PROMPT_GENERAL = """Ban la tro ly AI co kien thuc rong ve phap luat Viet Nam.
Tra loi cau hoi mot cach chinh xac, ngan gon va de hieu.
Neu khong chac chan, hay noi ro rang ban khong chac."""

HUMAN_PROMPT_GENERAL = """Cau hoi: {question}

Hay tra loi cau hoi tren."""


# ---------------------------------------------------------------------------
# Static messages
# ---------------------------------------------------------------------------
MSG_NEEDS_CLARIFICATION = "Cau hoi ban dua ra chua day du. Vui long mo ta chi tiet hon de toi co the ho tro ban."
MSG_NO_DOCS = "Chua co tai lieu nao duoc tai len. Vui long upload file truoc."


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _format_docs(docs: list[Document]) -> str:
    """Format retrieved documents into a context string with source info."""
    formatted_parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "N/A")
        section = doc.metadata.get("section", "")
        page = doc.metadata.get("page", "")

        header_parts = [f"[Nguon: {source}]"]
        if section:
            header_parts.append(f"[Muc: {section}]")
        if page != "":
            header_parts.append(f"[Trang: {page}]")

        header = " ".join(header_parts)
        formatted_parts.append(f"--- Tai lieu {i} {header} ---\n{doc.page_content}")

    return "\n\n".join(formatted_parts)


# ============================================================================
# RAGChain
# ============================================================================
class RAGChain:
    """Build prompt, route query, retrieve context, and call LLM Service."""

    def __init__(self, vector_store_manager: VectorStoreManager):
        self.vs_manager = vector_store_manager
        self.llm_url = f"{config.LLM_SERVICE_URL}/api/chat"
        self.router = LegalQueryRouter(llm_url=self.llm_url)

    # ---- LLM call ---------------------------------------------------------
    def _call_llm(self, messages: list[dict]) -> str:
        """Call the LLM Service via HTTP POST."""
        payload = {
            "messages": messages,
            "temperature": 0.1,
        }
        response = httpx.post(self.llm_url, json=payload, timeout=120.0)
        response.raise_for_status()
        return response.json()["content"]

    # ---- Pipeline: legal_rag ----------------------------------------------
    def _handle_legal_rag(
        self, question: str, source_filter: Optional[List[str]] = None
    ) -> str:
        """Standard RAG: retrieve → format → LLM with context."""
        retriever = self.vs_manager.get_retriever(source_filter=source_filter)
        if retriever is None:
            return MSG_NO_DOCS

        docs = retriever.invoke(question)
        context = _format_docs(docs)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_RAG},
            {
                "role": "user",
                "content": HUMAN_PROMPT_RAG.format(context=context, question=question),
            },
        ]
        return self._call_llm(messages)

    # ---- Pipeline: no_retrieval -------------------------------------------
    def _handle_no_retrieval(self, question: str) -> str:
        """Answer directly without document retrieval."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_GENERAL},
            {
                "role": "user",
                "content": HUMAN_PROMPT_GENERAL.format(question=question),
            },
        ]
        print("debug : khong can rag")
        return self._call_llm(messages)

    # ======== Main entry point =============================================
    def query(
        self, question: str, source_filter: Optional[List[str]] = None
    ) -> RAGResult:
        """Route the question and execute the appropriate pipeline."""
        # Step 1: Route
        decision: RouterDecision = self.router.route(question)

        # Step 2: Execute
        if decision.route == "needs_clarification":
            answer = MSG_NEEDS_CLARIFICATION

        elif decision.route == "no_retrieval":
            answer = self._handle_no_retrieval(question)

        else:  # legal_rag
            answer = self._handle_legal_rag(question, source_filter=source_filter)

        return RAGResult(
            answer=answer,
            route=decision.route,
            route_reason=decision.reason,
        )
