import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from core.rag import RAGPipeline
from core.agent import GeminiAgent

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["chat"])

def is_greeting(msg: str) -> bool:
    msg = msg.lower().strip()
    return any(word in msg for word in [
        "hi", "hello", "hey", "hii",
        "good morning", "good afternoon",  "good evening", "how are you"
    ])

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

    import time
    t0 = time.perf_counter()
    history = [m.model_dump() for m in req.history]

    try:
        agent = GeminiAgent.get_instance()

        # NEW: Greeting handling
        if is_greeting(req.message):
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

        # ── Step 1: RAG retrieval ──────────────────────────
        rag = RAGPipeline.get_instance()
        t1 = time.perf_counter()
        rag_result = rag.query(req.message)
        t2 = time.perf_counter()

        if rag_result["found"]:
            logger.info(
                f"RAG hit (similarity={rag_result['similarity']:.3f}) — "
                f"sources: {rag_result['sources']}"
            )

            t3 = time.perf_counter()
            reply = await agent.generate_rag_response(
                message=req.message,
                context=rag_result["context"],
                sources=rag_result["sources"],
                history=history,
            )
            t4 = time.perf_counter()

            print("LLM RESPONSE:", reply)

            if GeminiAgent.is_insufficient_context(reply):
                logger.info("LLM indicated insufficient context, using Gemini fallback.")
                try:
                    reply = await agent.generate_fallback_response(
                        message=req.message,
                        history=history,
                    )
                    return ChatResponse(
                        reply=reply,
                        source="gemini",
                        rag_sources=rag_result["sources"],
                        rag_similarity=round(rag_result["similarity"], 3),
                    )
                except Exception:
                    raise HTTPException(
                        status_code=500,
                        detail="⚠️ I'm having trouble responding right now. Please try again."
                    )

            return ChatResponse(
                reply=reply,
                source="rag",
                rag_sources=rag_result["sources"],
                rag_similarity=round(rag_result["similarity"], 3),
            )

        else:
            try:
                reply = await agent.generate_fallback_response(
                    message=req.message,
                    history=history,
                )
                return ChatResponse(
                    reply=reply,
                    source="gemini",
                    rag_similarity=round(rag_result["similarity"], 3),
                )
            except Exception:
                raise HTTPException(
                    status_code=500,
                    detail="I'm having trouble responding right now. Please try again."
                )

    except Exception as exc:
        logger.exception("Error in /api/chat")

        error_msg = str(exc).lower()

        # Friendly quota error
        if "quota" in error_msg or "429" in error_msg or "rate" in error_msg:
            raise HTTPException(
                status_code=429,
                detail="I'm a bit busy right now. Please try again in a few seconds"
            )

        # Generic friendly error
        raise HTTPException(
            status_code=500,
            detail="Something went wrong. Please try again later."
        )


@router.delete("/chat/session/{session_id}")
async def clear_session(session_id: str):
    """Placeholder — extend with Redis-backed session store if needed."""
    return {"cleared": True, "session_id": session_id}
