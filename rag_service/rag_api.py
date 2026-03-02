"""
RAG Service — Document upload, retrieval, and question answering.

Calls LLM Service for text generation. Uses Qdrant for vector storage.

Run:
    python rag_api.py
    # or: uvicorn rag_api:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import shutil
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

import config
from document_processor import DocumentProcessor
from vector_store import VectorStoreManager
from rag_chain import RAGChain
from schemas import (
    QuestionRequest,
    QuestionResponse,
    UploadResponse,
    FileListResponse,
    FileInfo,
    StatusResponse,
    HealthResponse,
)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="RAG Service",
    description="Document upload, retrieval, and question answering API (Qdrant)",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
processor = DocumentProcessor()
vs_manager = VectorStoreManager()
rag_chain = RAGChain(vs_manager)

os.makedirs(config.UPLOAD_DIR, exist_ok=True)

# Serve static files (frontend)
static_dir = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir), html=True), name="static")


@app.get("/")
def root():
    """Redirect root to frontend."""
    return RedirectResponse(url="/static/index.html")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/health", response_model=HealthResponse)
def health():
    """Health check — shows document count and readiness."""
    return HealthResponse(
        status="ok",
        documents_loaded=vs_manager.document_count,
        is_ready=vs_manager.is_ready,
    )


@app.post("/api/upload", response_model=UploadResponse)
async def upload_files(files: list[UploadFile] = File(...)):
    """Upload and process PDF files into the vector store."""
    if not files:
        raise HTTPException(status_code=400, detail="Vui long chon it nhat mot file.")

    saved_paths = []
    for f in files:
        ext = os.path.splitext(f.filename or "")[1].lower()
        if ext not in config.SUPPORTED_EXTENSIONS:
            continue
        dest = os.path.join(config.UPLOAD_DIR, f.filename or "upload.pdf")
        with open(dest, "wb") as buf:
            content = await f.read()
            buf.write(content)
        saved_paths.append(dest)

    if not saved_paths:
        raise HTTPException(
            status_code=400,
            detail=f"Khong co file hop le. Dinh dang ho tro: {', '.join(config.SUPPORTED_EXTENSIONS)}",
        )

    try:
        chunks = processor.load_and_process(saved_paths)
        vs_manager.add_documents(chunks)
        file_names = [os.path.basename(p) for p in saved_paths]
        return UploadResponse(
            message=f"Da xu ly thanh cong {len(saved_paths)} file ({', '.join(file_names)}).",
            file_count=len(saved_paths),
            chunk_count=len(chunks),
            total_documents=vs_manager.document_count,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Loi khi xu ly file: {str(e)}")


@app.get("/api/files", response_model=FileListResponse)
def list_files():
    """List all source files currently in the vector database."""
    source_files = vs_manager.get_source_files()
    files = []
    for name in source_files:
        chunk_count = vs_manager.get_file_chunk_count(name)
        files.append(FileInfo(name=name, chunk_count=chunk_count))
    return FileListResponse(files=files, total_files=len(files))


@app.delete("/api/files/{file_name}", response_model=StatusResponse)
def delete_file(file_name: str):
    """Delete a specific file and its chunks from the vector database."""
    if file_name not in vs_manager.get_source_files():
        raise HTTPException(status_code=404, detail=f"File '{file_name}' khong ton tai.")

    success = vs_manager.delete_file(file_name)
    if success:
        # Also remove from upload directory if exists
        upload_path = os.path.join(config.UPLOAD_DIR, file_name)
        if os.path.exists(upload_path):
            os.remove(upload_path)
        return StatusResponse(message=f"Da xoa file '{file_name}' va {file_name} chunks.")
    else:
        raise HTTPException(status_code=500, detail=f"Loi khi xoa file '{file_name}'.")


@app.post("/api/query", response_model=QuestionResponse)
def query(req: QuestionRequest):
    """Ask a question — routes through the query router, then executes the appropriate pipeline.

    Optionally filter retrieval to specific source files via source_filter.
    """
    try:
        result = rag_chain.query(
            req.question.strip(),
            source_filter=req.source_filter,
        )
        return QuestionResponse(
            answer=result.answer,
            route=result.route,
            route_reason=result.route_reason,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Loi khi xu ly cau hoi: {str(e)}")


@app.delete("/api/clear", response_model=StatusResponse)
def clear_all():
    """Clear vector store and all indexed data."""
    vs_manager.clear()
    # Clean upload directory
    if os.path.exists(config.UPLOAD_DIR):
        shutil.rmtree(config.UPLOAD_DIR)
        os.makedirs(config.UPLOAD_DIR, exist_ok=True)
    return StatusResponse(message="Da xoa toan bo du lieu.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "rag_api:app",
        host="0.0.0.0",
        port=config.RAG_SERVICE_PORT,
        reload=True,
    )
