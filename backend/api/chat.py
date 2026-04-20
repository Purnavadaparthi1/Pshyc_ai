import logging
import re as _re
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field


from core.rag import RAGPipeline
from core.agent import GeminiAgent
from core.config import settings

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


# ── Greeting detector ──────────────────────────────────────────────────────
_GREETING_RE = _re.compile(
    r"^\s*(hi|hello|hey|howdy|greetings|good\s*(morning|afternoon|evening|night)|"
    r"what'?s up|sup|hiya|yo|namaste|helo|hii+|heyy+)\W*\s*$",
    _re.IGNORECASE,
)


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

    import time
    t0 = time.perf_counter()
    history = [m.model_dump() for m in req.history]


    try:
        # ── Greeting bypass: skip RAG for casual greetings ─────────────────
        if _GREETING_RE.match(req.message):
            logger.info("Greeting detected — skipping RAG, using Gemini directly.")
            agent = GeminiAgent.get_instance()
            try:
                reply = await agent.generate_fallback_response(
                    message=req.message,
                    history=history,
                )
                return ChatResponse(
                    reply=reply,
                    source="gemini",
                    rag_sources=[],
                    rag_similarity=0.0,
                )
            except Exception as exc:
                logger.exception("Error in Gemini greeting response")
                raise HTTPException(status_code=500, detail="Error in Gemini greeting: " + str(exc))
        # ── end greeting bypass ────────────────────────────────────────────

        rag = RAGPipeline.get_instance()
        t1 = time.perf_counter()
        rag_result = rag.query(req.message)
        t2 = time.perf_counter()

        agent = GeminiAgent.get_instance()
        similarity_threshold = getattr(rag, 'similarity_threshold', None) or getattr(settings, 'RAG_SIMILARITY_THRESHOLD', 0.7)
        top_score = rag_result.get("similarity", 0.0)

        # 3-stage decision: high, medium, low confidence
        # Unified fallback logic: always use Gemini for insufficient or low RAG context
        rag_confident = rag_result["found"] and top_score >= similarity_threshold
        rag_medium = rag_result["found"] and top_score >= (similarity_threshold * 0.7)
        use_rag = False
        reply = None
        if rag_confident or rag_medium:
            # Try RAG answer first
            logger.info(
                f"RAG {'high' if rag_confident else 'medium'} confidence (similarity={rag_result['similarity']:.3f}) — using RAG+LLM reasoning"
            )
            t3 = time.perf_counter()
            reply = await agent.generate_rag_response(
                message=req.message,
                context=rag_result["context"],
                sources=rag_result["sources"],
                history=history,
            )
            t4 = time.perf_counter()
            logger.info(f"Gemini RAG response: {reply}")
            # If RAG answer is insufficient, fallback to Gemini direct answer
            if not GeminiAgent.is_insufficient_context(reply, req.message):
                use_rag = True
        if not use_rag:
            # Always use Gemini fallback for insufficient or low RAG context
            logger.info("Using Gemini fallback for insufficient or low RAG context.")
            try:
                # Always request a concise, strong answer
                fallback_reply = await agent.generate_fallback_response(
                    message=f"(medium, strong, concise) {req.message}",
                    history=history,
                )
                logger.info(f"Gemini fallback response: {fallback_reply}")
                return ChatResponse(
                    reply=fallback_reply,
                    source="gemini",
                    rag_sources=rag_result.get("sources", []),
                    rag_similarity=round(rag_result.get("similarity", 0.0), 3),
                )
            except Exception as exc:
                logger.exception("Error in Gemini fallback response")
                raise HTTPException(status_code=500, detail="Error in Gemini fallback: " + str(exc))
        logger.info(f"TIMING: RAG setup: {t1-t0:.2f}s, RAG query: {t2-t1:.2f}s, Gemini RAG: {t4-t3:.2f}s, Total: {t4-t0:.2f}s")
        return ChatResponse(
            reply=reply,
            source="rag",
            rag_sources=rag_result["sources"],
            rag_similarity=round(rag_result["similarity"], 3),
        )

    except Exception as exc:
        logger.exception("Error in /api/chat")
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete("/chat/session/{session_id}")
async def clear_session(session_id: str):
    """Placeholder — extend with Redis-backed session store if needed."""
    return {"cleared": True, "session_id": session_id}