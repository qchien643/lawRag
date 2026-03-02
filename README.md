# 1. Qdrant (cần Docker)
docker run -p 6333:6333 qdrant/qdrant
# 2. LLM Service
cd e:\code\project\updatalaw\lawRag\llm_service
& "..\\.venv\\Scripts\\python.exe" llm_api.py
# 3. RAG Service + Frontend
cd e:\code\project\updatalaw\lawRag\rag_service
& "..\\.venv\\Scripts\\python.exe" rag_api.py
# → Mở http://localhost:8000