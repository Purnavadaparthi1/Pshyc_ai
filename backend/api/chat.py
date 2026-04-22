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
        # Unified fallback logic: always use Gemini for insufficient or low RAG context
        # Strict fallback: Only use RAG if answer is sufficient and related, otherwise always use Gemini
        rag_confident = rag_result["found"] and top_score >= similarity_threshold
        reply = None
        if rag_confident:
            logger.info(
                f"RAG high confidence (similarity={rag_result['similarity']:.3f}) — using RAG+LLM reasoning"
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
            if not GeminiAgent.is_insufficient_context(reply, req.message):
                logger.info("[RESPONSE TYPE] RAG answer (structured by Gemini, but content from RAG). Returning as 'rag'.")
                return ChatResponse(
                    reply=reply,
                    source="rag",
                    rag_sources=rag_result["sources"],
                    rag_similarity=round(rag_result["similarity"], 3),
                )
            else:
                logger.info("[RESPONSE TYPE] RAG insufficient, falling back to Gemini. Will return as 'gemini'.")
        # Always fallback to Gemini if not high confidence or insufficient RAG answer
        try:
            logger.info(f"Calling Gemini fallback with message: {req.message}")
            fallback_reply = await agent.generate_fallback_response(
                message=req.message,
                history=history,
            )
            logger.info(f"Gemini fallback response: {fallback_reply}")
            logger.info("[RESPONSE TYPE] Gemini fallback (no sufficient RAG context, Gemini answered directly). Returning as 'gemini'.")
            # Only show RAG sources if similarity is above threshold (e.g., 0.6)
            rag_sources = rag_result.get("sources", [])
            rag_similarity = round(rag_result.get("similarity", 0.0), 3)
            SIMILARITY_THRESHOLD = 0.6
            if rag_sources and rag_similarity >= SIMILARITY_THRESHOLD:
                formatted_sources = []
                for src in rag_sources:
                    if src.startswith("http://") or src.startswith("https://"):
                        formatted_sources.append(f"[{src}]({src})")
                    else:
                        formatted_sources.append(src)
                sources_md = "\n".join(f"- {s}" for s in formatted_sources)
                platform_md = "\n\n**Source Platform:** IGNOU Material (retrieved via PSYCH.AI RAG)"
                fallback_reply = f"{fallback_reply}\n\n---\n\n## Source of Information\n\n{sources_md}{platform_md}"
            else:
                fallback_reply = f"{fallback_reply}\n\n---\n\n## Source of Information\n\nNo relevant IGNOU document matched your query, so this answer is based on general psychology knowledge."
            return ChatResponse(
                reply=fallback_reply,
                source="gemini",
                rag_sources=rag_sources,
                rag_similarity=rag_similarity,
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