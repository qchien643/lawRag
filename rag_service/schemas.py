from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    """Request body for the /api/query endpoint."""
    question: str = Field(..., min_length=1, description="User question")
    source_filter: list[str] | None = Field(
        None, description="Optional list of source file names to filter retrieval"
    )


class QuestionResponse(BaseModel):
    """Response body for the /api/query endpoint."""
    answer: str = Field(..., description="RAG-generated answer with source citations")
    route: str = Field("", description="Router decision: legal_rag, no_retrieval, needs_clarification")
    route_reason: str = Field("", description="Explanation of why this route was chosen")


class UploadResponse(BaseModel):
    """Response body for the /api/upload endpoint."""
    message: str
    file_count: int
    chunk_count: int
    total_documents: int


class FileInfo(BaseModel):
    """Info about a single file in the database."""
    name: str
    chunk_count: int


class FileListResponse(BaseModel):
    """Response body for the /api/files endpoint."""
    files: list[FileInfo]
    total_files: int


class StatusResponse(BaseModel):
    """Generic status response."""
    message: str


class HealthResponse(BaseModel):
    """Response body for the /api/health endpoint."""
    status: str
    documents_loaded: int
    is_ready: bool
