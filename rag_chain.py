from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

import config
from vector_store import VectorStoreManager


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """Ban la tro ly AI chuyen tra loi cau hoi dua tren tai lieu duoc cung cap.

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

HUMAN_PROMPT = """Noi dung tham khao:
{context}

Cau hoi: {question}

Hay tra loi cau hoi tren dua vao noi dung tham khao. Nho trich dan nguon."""


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


class RAGChain:
    """Build and invoke a RAG chain with source citations."""

    def __init__(self, vector_store_manager: VectorStoreManager):
        self.vs_manager = vector_store_manager
        self.llm = ChatOpenAI(
            model=config.CHAT_MODEL,
            openai_api_key=config.OPENAI_API_KEY,
            openai_api_base=config.BASE_URL,
            temperature=0.1,
        )
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", HUMAN_PROMPT),
            ]
        )

    def query(self, question: str) -> str:
        """Run a question through the RAG chain and return answer with sources."""
        retriever = self.vs_manager.get_retriever()
        if retriever is None:
            return "Chua co tai lieu nao duoc tai len. Vui long upload file truoc."

        chain = (
            {"context": retriever | _format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        answer = chain.invoke(question)
        return answer
