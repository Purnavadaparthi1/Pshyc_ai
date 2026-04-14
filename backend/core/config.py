from pydantic_settings import BaseSettings
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    GEMINI_API_KEY: str
    GEMINI_MODEL: str = "gemini-2.5-flash"
    CHROMA_PERSIST_DIR: str = str(BASE_DIR / "chroma_db")
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    RAG_SIMILARITY_THRESHOLD: float = 0.6
    RAG_TOP_K: int = 5
    FRONTEND_DIST: str = str(BASE_DIR.parent / "frontend" / "dist")

    class Config:
        env_file = str(BASE_DIR / ".env")


settings = Settings()
