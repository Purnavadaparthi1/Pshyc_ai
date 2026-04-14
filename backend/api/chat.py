import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from fastapi.responses import JSONResponse
from core.rag import RAGPipeline
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

@router.get("/health", response_model=HealthResponse)
async def health():
    rag = RAGPipeline.get_instance()
    return HealthResponse(
        status="ok",
        rag_chunks=rag.collection.count(),
        model="gemini-2.5-flash",
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    history = [m.model_dump() for m in req.history]

    try:
        # ── Step 1: RAG retrieval ──────────────────────────────────────────
        rag = RAGPipeline.get_instance()
        rag_result = rag.query(req.message)

        agent = GeminiAgent.get_instance()

        # Serve RAG context directly if similarity is high (e.g., >= 0.5)
        if rag_result["found"] and rag_result["similarity"] >= 0.5:
            logger.info(
                f"RAG direct answer (similarity={rag_result['similarity']:.3f}) — "
                f"sources: {rag_result['sources']}"
            )
            # Remove PDF/source names from the response
            reply = rag_result["context"]
            return ChatResponse(
                reply=reply,
                source="rag",
                rag_sources=[],  # Do not return PDF/source names
                rag_similarity=round(rag_result["similarity"], 3),
            )
        elif rag_result["found"]:
            logger.info(
                f"RAG hit (similarity={rag_result['similarity']:.3f}) — "
                f"sources: {rag_result['sources']}"
            )
            # ── Step 2a: RAG-grounded response via Gemini ────────────────
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
            logger.info(
                f"RAG miss (similarity={rag_result['similarity']:.3f}) — "
                "using Gemini agent fallback"
            )
            # ── Step 2b: Pure Gemini agent fallback ───────────────────────
            reply = await agent.generate_fallback_response(
                message=req.message,
                history=history,
            )
            return ChatResponse(
                reply=reply,
                source="gemini",
                rag_similarity=round(rag_result["similarity"], 3),
            )

    except Exception as exc:
        logger.exception("Error in /api/chat")

        error_str = str(exc)

    # ✅ Quota / rate limit error
    if "429" in error_str or "quota" in error_str:
        return JSONResponse(
            status_code=429,
            content={
                "detail": "⏳ Too many requests right now. Please wait a few seconds and try again."
            },
        )

    # ✅ Network / timeout
    if "timeout" in error_str.lower():
        return JSONResponse(
            status_code=500,
            content={
                "detail": "🌐 Network is slow. Please try again."
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
