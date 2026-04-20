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


# ── Friendly error helper ──────────────────────────────────────────────────
def _friendly_error(exc: Exception) -> str:
    """Convert raw API/network exceptions into user-friendly messages."""
    raw = str(exc)
    if "429" in raw or "quota" in raw.lower() or "rate" in raw.lower():
        return "We're getting a lot of questions right now! Please wait a few seconds and try again."
    if "503" in raw or "unavailable" in raw.lower():
        return "PSYCH.AI is temporarily unavailable. Please try again in a moment."
    if "timeout" in raw.lower():
        return "The request timed out — PSYCH.AI is thinking hard! Please try again."
    if "network" in raw.lower() or "connection" in raw.lower():
        return "Can't reach the server — please check your internet connection."
    return "PSYCH.AI hit a server hiccup — please try again in a moment."


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
                raise HTTPException(status_code=500, detail=_friendly_error(exc))
        # ── end greeting bypass ────────────────────────────────────────────

        rag = RAGPipeline.get_instance()
        t1 = time.perf_counter()
        rag_result = rag.query(req.message)
        t2 = time.perf_counter()

        agent = GeminiAgent.get_instance()
        similarity_threshold = getattr(rag, 'similarity_threshold', None) or getattr(settings, 'RAG_SIMILARITY_THRESHOLD', 0.7)
        top_score = rag_result.get("similarity", 0.0)

        # 3-stage decision: high, medium, low confidence
        if rag_result["found"] and top_score >= similarity_threshold:
            # High confidence: use RAG answer
            logger.info(
                f"RAG hit (similarity={rag_result['similarity']:.3f}) — "
                f"sources: {rag_result['sources']}"
            )
            t3 = time.perf_counter()
            logger.info(f"Calling Gemini RAG response with message: {req.message}\nContext: {rag_result['context']}\nSources: {rag_result['sources']}")
            reply = await agent.generate_rag_response(
                message=req.message,
                context=rag_result["context"],
                sources=rag_result["sources"],
                history=history,
            )
            t4 = time.perf_counter()
            logger.info(f"Gemini RAG response: {reply}")
            if GeminiAgent.is_insufficient_context(reply, req.message):
                logger.info("LLM indicated insufficient context, triggering Gemini fallback.")
                try:
                    logger.info(f"Calling Gemini fallback with message: {req.message}")
                    fallback_reply = await agent.generate_fallback_response(
                        message=req.message,
                        history=history,
                    )
                    logger.info(f"Gemini fallback response: {fallback_reply}")
                    fallback_reply = (
                        f"**Note:** This answer is based on general knowledge, not from IGNOU material.\n\n"
                        f"{fallback_reply}"
                    )
                    return ChatResponse(
                        reply=fallback_reply,
                        source="gemini",
                        rag_sources=rag_result["sources"],
                        rag_similarity=round(rag_result["similarity"], 3),
                    )
                except Exception as exc:
                    logger.exception("Error in Gemini fallback response")
                    raise HTTPException(status_code=500, detail=_friendly_error(exc))
            logger.info(f"TIMING: RAG setup: {t1-t0:.2f}s, RAG query: {t2-t1:.2f}s, Gemini RAG: {t4-t3:.2f}s, Total: {t4-t0:.2f}s")
            return ChatResponse(
                reply=reply,
                source="rag",
                rag_sources=rag_result["sources"],
                rag_similarity=round(rag_result["similarity"], 3),
            )
        elif rag_result["found"] and top_score >= (similarity_threshold * 0.7):
            # Medium confidence: use RAG+LLM reasoning
            logger.info(
                f"RAG medium confidence (similarity={rag_result['similarity']:.3f}) — using RAG+LLM reasoning"
            )
            t3 = time.perf_counter()
            try:
                reply = await agent.generate_rag_response(
                    message=req.message,
                    context=rag_result["context"],
                    sources=rag_result["sources"],
                    history=history,
                )
            except Exception as exc:
                logger.exception("Error in Gemini RAG+LLM response")
                raise HTTPException(status_code=500, detail=_friendly_error(exc))
            t4 = time.perf_counter()
            logger.info(f"Gemini RAG+LLM response: {reply}")
            reply = (
                f"**Note:** This answer is based on partial matches from IGNOU material and general knowledge.\n\n"
                f"{reply}"
            )
            return ChatResponse(
                reply=reply,
                source="rag",
                rag_sources=rag_result["sources"],
                rag_similarity=round(rag_result["similarity"], 3),
            )
        else:
            # Low confidence: fallback to Gemini with disclaimer
            logger.info(
                f"RAG miss or low similarity (similarity={rag_result['similarity']:.3f}) — using Gemini agent fallback"
            )
            try:
                logger.info(f"Calling Gemini fallback with message: {req.message}")
                fallback_reply = await agent.generate_fallback_response(
                    message=req.message,
                    history=history,
                )
                logger.info(f"Gemini fallback response: {fallback_reply}")
                fallback_reply = (
                    f"**Note:** This answer is based on general knowledge, not from IGNOU material.\n\n"
                    f"{fallback_reply}"
                )
                return ChatResponse(
                    reply=fallback_reply,
                    source="gemini",
                    rag_similarity=round(rag_result["similarity"], 3),
                )
            except Exception as exc:
                logger.exception("Error in Gemini fallback response")
                raise HTTPException(status_code=500, detail=_friendly_error(exc))

    except HTTPException:
        raise  # re-raise already-friendly HTTP exceptions as-is
    except Exception as exc:
        logger.exception("Error in /api/chat")
        raise HTTPException(status_code=500, detail=_friendly_error(exc))


@router.delete("/chat/session/{session_id}")
async def clear_session(session_id: str):
    """Placeholder — extend with Redis-backed session store if needed."""
    return {"cleared": True, "session_id": session_id}