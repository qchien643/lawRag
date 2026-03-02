import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from parent directory (lawRag/.env)
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)

# OpenAI / compatible API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
BASE_URL = os.getenv("BASE_URL", "https://api.openai.com/v1")

# Model
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o")

# Server
LLM_SERVICE_PORT = int(os.getenv("LLM_SERVICE_PORT", "8001"))
