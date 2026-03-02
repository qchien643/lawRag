from pydantic import BaseModel, Field


class MessageItem(BaseModel):
    """A single chat message."""
    role: str = Field(..., description="Role: 'system', 'user', or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Request body for the /api/chat endpoint."""
    messages: list[MessageItem] = Field(..., description="List of chat messages")
    temperature: float = Field(0.1, ge=0, le=2, description="Sampling temperature")
    model: str | None = Field(None, description="Override model name (optional)")


class ChatResponse(BaseModel):
    """Response body for the /api/chat endpoint."""
    content: str = Field(..., description="LLM generated text")


class HealthResponse(BaseModel):
    """Response body for the /api/health endpoint."""
    status: str
    model: str
