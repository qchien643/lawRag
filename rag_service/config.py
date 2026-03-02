import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from parent directory (lawRag/.env)
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)

# Embedding model
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
BASE_URL = os.getenv("BASE_URL", "https://api.openai.com/v1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# Qdrant
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "rag_collection")

# Retrieval
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "6"))
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.4"))
SEMANTIC_WEIGHT = float(os.getenv("SEMANTIC_WEIGHT", "0.6"))

# Upload
SUPPORTED_EXTENSIONS = [".pdf"]
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")

# LLM Service
LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL", "http://localhost:8001")

# Server
RAG_SERVICE_PORT = int(os.getenv("RAG_SERVICE_PORT", "8000"))
# ---------------------------------------------------------------------------
# Query Router
# ---------------------------------------------------------------------------

# Keywords that signal the query likely needs RAG retrieval.
# If ANY of these appear in the query (case-insensitive), the router
# immediately decides "legal_rag".  Edit this list as needed.
RAG_KEYWORDS = [
    # Nội quy / quy chế
    "nội quy", "quy chế", "quy định nội bộ", "quy trình",
    "chính sách", "hợp đồng",
    # Luật nhà nước
    "luật", "nghị định", "thông tư", "bộ luật", "pháp lệnh",
    "luật lao động", "bộ luật dân sự", "thuế thu nhập",
    "bảo hiểm xã hội", "hình sự", "hành chính",
    # Quy phạm
    "điều khoản", "quy phạm", "văn bản pháp luật",
    "nghỉ phép", "thai sản", "thử việc", "sa thải",
    "bồi thường", "xử phạt", "kỷ luật",
    # Tham chiếu
    "điều", "khoản", "mục", "chương",
]

# Example questions that are TYPICAL of queries needing RAG.
# The router embeds these at startup and compares incoming queries
# against them using cosine similarity.  Edit / add examples as needed.
RAG_EXAMPLE_QUESTIONS = [
    "Quy định nghỉ phép năm của công ty là gì?",
    "Mức phạt đi muộn theo nội quy lao động?",
    "Hợp đồng thử việc có thời hạn bao lâu?",
    "Theo Luật Lao động thì nghỉ thai sản bao nhiêu tháng?",
    "Nghị định 145 quy định gì về bảo hiểm xã hội?",
    "Bộ luật Dân sự Điều 428 nói về gì?",
    "Quy trình xử lý kỷ luật lao động như thế nào?",
    "Điều kiện sa thải nhân viên theo luật?",
    "Chính sách bồi thường thiệt hại trong hợp đồng?",
    "Thông tư hướng dẫn về thuế thu nhập cá nhân?",
]

# Cosine similarity threshold for semantic layer.
# Query is considered "needs RAG" if similarity >= this value.
SEMANTIC_SIMILARITY_THRESHOLD = float(os.getenv("SEMANTIC_SIMILARITY_THRESHOLD", "0.75"))