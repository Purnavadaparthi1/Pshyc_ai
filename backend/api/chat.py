import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from fastapi.responses import JSONResponse

try:
    from core.rag import RAGPipeline
except ImportError:
    RAGPipeline = None
from core.agent import GeminiAgent

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["chat"])


# ── Request / Response models ──────────────────────────────────────────────

class Message(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    history: list[Message] = Field(default_factory=list)
    session_id: str | None = None


class ChatResponse(BaseModel):
    reply: str
    source: str          # "rag" | "gemini"
    rag_sources: list[str] = []
    rag_similarity: float = 0.0


class HealthResponse(BaseModel):
    status: str
    rag_chunks: int
    model: str


# ── Endpoints ──────────────────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    history = [m.model_dump() for m in req.history]

    try:
        rag = RAGPipeline.get_instance()
        rag_result = rag.query(req.message)

        agent = GeminiAgent.get_instance()

        if rag_result["found"]:
            reply = await agent.generate_rag_response(
                message=req.message,
                context=rag_result["context"],
                sources=rag_result["sources"],
                history=history,
            )
            return ChatResponse(
                reply=reply,
                source="rag",
                rag_sources=rag_result["sources"],
                rag_similarity=round(rag_result["similarity"], 3),
            )
        else:
            reply = await agent.generate_fallback_response(
                message=req.message,
                history=history,
            )
            return ChatResponse(
                reply=reply,
                source="gemini",
                rag_similarity=round(rag_result["similarity"], 3),
            )

    # ✅ ✅ FIXED ERROR HANDLING
    except Exception as exc:
        logger.exception("Error in /api/chat")
        
        error_str = str(exc)

    # ✅ Handle quota error
    if "429" in error_str or "quota" in error_str:
        return JSONResponse(
            status_code=429,
            content={
                "detail": "⏳ Too many requests right now. Please wait a few seconds and try again."
            },
        )

    # ✅ Generic error
    return JSONResponse(
        status_code=500,
        content={
            "detail": "⚠️ Something went wrong. Please try again."
        },
    )


@router.delete("/chat/session/{session_id}")
async def clear_session(session_id: str):
    """Placeholder — extend with Redis-backed session store if needed."""
    return {"cleared": True, "session_id": session_id}
