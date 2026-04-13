import logging
from typing import Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
from .config import settings

logger = logging.getLogger(__name__)


class RAGPipeline:
    _instance: Optional["RAGPipeline"] = None

    def __init__(self):
        logger.info("Initialising RAG pipeline...")
        self.embedder = SentenceTransformer(settings.EMBEDDING_MODEL)
        self.client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIR,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name="psych_ai_docs",
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"RAG ready — {self.collection.count()} chunks indexed in ChromaDB"
        )

    @classmethod
    def get_instance(cls) -> "RAGPipeline":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def query(self, question: str, top_k: int | None = None) -> dict:
        k = top_k or settings.RAG_TOP_K
        count = self.collection.count()
        if count == 0:
            return {"found": False, "context": "", "sources": [], "similarity": 0.0}

        query_embedding = self.embedder.encode(question).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, count),
            include=["documents", "distances", "metadatas"],
        )

        if not results["documents"] or not results["documents"][0]:
            return {"found": False, "context": "", "sources": [], "similarity": 0.0}

        distances = results["distances"][0]
        # In ChromaDB cosine space: distance = 1 - cosine_similarity
        best_similarity = 1.0 - distances[0]

        if best_similarity < settings.RAG_SIMILARITY_THRESHOLD:
            logger.info(
                f"RAG below threshold ({best_similarity:.3f} < "
                f"{settings.RAG_SIMILARITY_THRESHOLD}), falling back to Gemini"
            )
            return {"found": False, "context": "", "sources": [], "similarity": best_similarity}

        docs = results["documents"][0]
        metadatas = results["metadatas"][0]

        # Build ranked context — closest chunk first
        context_parts = []
        for doc, meta in zip(docs, metadatas):
            source_label = meta.get("source", "IGNOU Material")
            context_parts.append(f"[{source_label}]\n{doc}")

        context = "\n\n---\n\n".join(context_parts)
        sources = list({m.get("source", "IGNOU Material") for m in metadatas})

        return {
            "found": True,
            "context": context,
            "sources": sources,
            "similarity": best_similarity,
        }
