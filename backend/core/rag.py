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
        import time
        t0 = time.perf_counter()
        keyword_boost = "id ego superego structural model personality explanation"
        hybrid_query = f"{keyword_boost} {question}"
        k = top_k or settings.RAG_TOP_K
        count = self.collection.count()
        if count == 0:
            return {"found": False, "context": "", "sources": [], "similarity": 0.0}

        # Embedding timing
        t1 = time.perf_counter()
        query_embedding = self.embedder.encode(hybrid_query).tolist()
        t2 = time.perf_counter()
        logger.info(f"TIMING: Embedding: {t2-t1:.2f}s")

        # Retrieval timing
        t3 = time.perf_counter()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, count),
            include=["documents", "distances", "metadatas"],
        )
        t4 = time.perf_counter()
        logger.info(f"TIMING: Retrieval: {t4-t3:.2f}s")

        if not results["documents"] or not results["documents"][0]:
            return {"found": False, "context": "", "sources": [], "similarity": 0.0}

        # Re-rank: sort by similarity (lowest distance)
        docs = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        doc_tuples = list(zip(docs, metadatas, distances))
        doc_tuples.sort(key=lambda x: x[2])  # sort by distance ascending
        # Take best 3 for context (configurable)
        top_docs = doc_tuples[:k]

        import re
        def clean_text(text):
            text = re.sub(r"^\s*\[.*?\]\s*", "", text, flags=re.MULTILINE)
            text = re.sub(r"(\b\d+(?:[\.-]\d+)+\b)", "", text)
            text = re.sub(r"(?<!\d)(\b\d{1,2}\b)(?!\d)", "", text)
            text = re.sub(r"(?m)^[-_]+", "", text)
            text = text.replace("\n", " ")
            text = re.sub(r"\s+", " ", text)
            return text.strip()

        context_parts = []
        sources = set()
        for doc, meta, _ in top_docs:
            cleaned = clean_text(doc)
            context_parts.append(cleaned)
            sources.add(meta.get("source", "IGNOU Material"))

        # Minimize context sent to Gemini (truncate to 1000 chars)
        context = " ".join(context_parts)[:1000]

        best_similarity = 1.0 - top_docs[0][2] if top_docs else 0.0

        return {
            "found": True,
            "context": context,
            "sources": list(sources),
            "similarity": best_similarity,
        }
