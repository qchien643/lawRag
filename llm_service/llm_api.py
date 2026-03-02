"""
LLM Service — Lightweight API wrapper around OpenAI-compatible chat models.

Run:
    python llm_api.py
    # or: uvicorn llm_api:app --host 0.0.0.0 --port 8001 --reload
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

import config
from schemas import ChatRequest, ChatResponse, HealthResponse

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="LLM Service",
    description="API wrapper for OpenAI-compatible chat models",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# OpenAI client (created once)
# ---------------------------------------------------------------------------
client = OpenAI(
    api_key=config.OPENAI_API_KEY,
    base_url=config.BASE_URL,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/health", response_model=HealthResponse)
def health():
    """Health check."""
    return HealthResponse(status="ok", model=config.CHAT_MODEL)


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Send messages to the LLM and return the generated text."""
    model = req.model or config.CHAT_MODEL

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[m.model_dump() for m in req.messages],
            temperature=req.temperature,
        )
        content = response.choices[0].message.content or ""
        return ChatResponse(content=content)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM error: {str(e)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "llm_api:app",
        host="0.0.0.0",
        port=config.LLM_SERVICE_PORT,
        reload=True,
    )
