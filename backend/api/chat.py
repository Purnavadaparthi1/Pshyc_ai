import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

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

    import time
    t0 = time.perf_counter()
    history = [m.model_dump() for m in req.history]

    try:
        rag = RAGPipeline.get_instance()
        t1 = time.perf_counter()
        rag_result = rag.query(req.message)
        t2 = time.perf_counter()

        agent = GeminiAgent.get_instance()

        if rag_result["found"]:
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
                    reply = await agent.generate_fallback_response(
                        message=req.message,
                        history=history,
                    )
                    logger.info(f"Gemini fallback response: {reply}")
                    return ChatResponse(
                        reply=reply,
                        source="gemini",
                        rag_sources=rag_result["sources"],
                        rag_similarity=round(rag_result["similarity"], 3),
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
        else:
            logger.info(
                f"RAG miss (similarity={rag_result['similarity']:.3f}) — using Gemini agent fallback"
            )
            try:
                logger.info(f"Calling Gemini fallback with message: {req.message}")
                reply = await agent.generate_fallback_response(
                    message=req.message,
                    history=history,
                )
                logger.info(f"Gemini fallback response: {reply}")
                return ChatResponse(
                    reply=reply,
                    source="gemini",
                    rag_similarity=round(rag_result["similarity"], 3),
                )
            except Exception as exc:
                logger.exception("Error in Gemini fallback response")
                raise HTTPException(status_code=500, detail="Error in Gemini fallback: " + str(exc))

    except Exception as exc:
        logger.exception("Error in /api/chat")
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete("/chat/session/{session_id}")
async def clear_session(session_id: str):
    """Placeholder — extend with Redis-backed session store if needed."""
    return {"cleared": True, "session_id": session_id}
