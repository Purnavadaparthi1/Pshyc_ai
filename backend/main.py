"""
PSYCH.AI — FastAPI backend
Serves the React frontend from ../frontend/dist and exposes /api/* endpoints.
"""

import logging
import os
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Ensure backend/ directory is on path so scripts/ can be imported
sys.path.insert(0, str(Path(__file__).parent))

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.chat import router as chat_router
from core.config import settings
from core.rag import RAGPipeline
from core.agent import GeminiAgent

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("psych_ai")

FRONTEND_DIST = Path(settings.FRONTEND_DIST)
FRONTEND_SRC  = FRONTEND_DIST.parent.parent  # psych-ai/frontend/


def build_frontend():
    """Build the React frontend if dist/ doesn't exist or is stale."""
    if FRONTEND_DIST.exists() and any(FRONTEND_DIST.iterdir()):
        logger.info("Frontend dist already built — skipping build step.")
        return

    logger.info("Building React frontend (npm run build)…")
    result = subprocess.run(
        ["npm", "run", "build"],
        cwd=str(FRONTEND_SRC),
        capture_output=False,
    )
    if result.returncode != 0:
        logger.error("Frontend build failed! Run manually: cd frontend && npm run build")
        sys.exit(1)
    logger.info("Frontend build complete.")



# No-op: PDF generation and auto-ingestion removed. Use scripts/ingest.py for MAPC docs.
def bootstrap_knowledge_base():
    logger.info("Knowledge base bootstrap skipped. Use MAPC docs and scripts/ingest.py for ingestion.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ────────────────────────────────────────────────────────────
    logger.info("═" * 55)
    logger.info("  PSYCH.AI  — Starting up")
    logger.info("═" * 55)

    # Build frontend (no-op if already built)
    build_frontend()

    # Pre-warm RAG and Gemini singletons
    RAGPipeline.get_instance()
    GeminiAgent.get_instance()

    # Auto-generate + ingest knowledge base on first run
    bootstrap_knowledge_base()

    logger.info("PSYCH.AI is ready  →  http://localhost:8000")
    yield
    # ── Shutdown ───────────────────────────────────────────────────────────
    logger.info("PSYCH.AI shutting down.")


# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="PSYCH.AI",
    description="AI-powered psychology tutor for IGNOU MA Psychology students",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # tighten in prod: ["https://yourdomain.com"]
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── API routes ─────────────────────────────────────────────────────────────
app.include_router(chat_router)


# ── Static frontend ────────────────────────────────────────────────────────
if FRONTEND_DIST.exists():
    # Serve JS/CSS/assets
    app.mount(
        "/assets",
        StaticFiles(directory=str(FRONTEND_DIST / "assets")),
        name="assets",
    )

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        """Catch-all — serve index.html for React Router / SPA."""
        index = FRONTEND_DIST / "index.html"
        return FileResponse(str(index))
else:
    @app.get("/", include_in_schema=False)
    async def not_built():
        return {
            "message": "Frontend not built. Run: cd frontend && npm install && npm run build"
        }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,           # set True for dev: PSYCH_DEV=1
        workers=1,
        log_level="info",
    )
