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
    history = [m.model_dump() for m in req.history]

    try:
        # ── Step 1: RAG retrieval ──────────────────────────────────────────
        rag = RAGPipeline.get_instance()
        rag_result = rag.query(req.message)

        agent = GeminiAgent.get_instance()

        # Always use Gemini LLM to generate the final answer, never return raw context
        if rag_result["found"]:
            logger.info(
                f"RAG hit (similarity={rag_result['similarity']:.3f}) — "
                f"sources: {rag_result['sources']}"
            )
            # Step 2: RAG-grounded response via Gemini
            reply = await agent.generate_rag_response(
                message=req.message,
                context=rag_result["context"],
                sources=rag_result["sources"],
                history=history,
            )
            print("LLM RESPONSE:", reply)
            # Fallback to Gemini if LLM says context is insufficient or not accurate
            fallback_phrases = [
                "insufficient context", "not enough context", "no relevant content", "context does not provide", "sorry, no relevant content"
            ]
            if any(phrase in reply.lower() for phrase in fallback_phrases):
                logger.info("LLM indicated insufficient context, using Gemini fallback.")
                reply = await agent.generate_fallback_response(
                    message=req.message,
                    history=history,
                )
                print("GEMINI FALLBACK RESPONSE:", reply)
                return ChatResponse(
                    reply=reply,
                    source="gemini",
                    rag_sources=rag_result["sources"],
                    rag_similarity=round(rag_result["similarity"], 3),
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
            # Step 2b: Pure Gemini agent fallback
            reply = await agent.generate_fallback_response(
                message=req.message,
                history=history,
            )
            print("LLM RESPONSE:", reply)
            return ChatResponse(
                reply=reply,
                source="gemini",
                rag_similarity=round(rag_result["similarity"], 3),
            )

    except Exception as exc:
        logger.exception("Error in /api/chat")
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete("/chat/session/{session_id}")
async def clear_session(session_id: str):
    """Placeholder — extend with Redis-backed session store if needed."""
    return {"cleared": True, "session_id": session_id}
