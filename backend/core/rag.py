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
        # Hybrid query: boost keywords for Freud's model
        keyword_boost = "id ego superego structural model personality explanation"
        hybrid_query = f"{keyword_boost} {question}"
        k = top_k or settings.RAG_TOP_K
        count = self.collection.count()
        if count == 0:
            return {"found": False, "context": "", "sources": [], "similarity": 0.0}

        query_embedding = self.embedder.encode(hybrid_query).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, count),
            include=["documents", "distances", "metadatas"],
        )

        if not results["documents"] or not results["documents"][0]:
            return {"found": False, "context": "", "sources": [], "similarity": 0.0}

        # Re-rank: sort by similarity (lowest distance)
        docs = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        doc_tuples = list(zip(docs, metadatas, distances))
        doc_tuples.sort(key=lambda x: x[2])  # sort by distance ascending
        # Take best 3 for context
        top_docs = doc_tuples[:3]

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
        # Collect topic/subtopic from the best match
        if top_docs:
            best_meta = top_docs[0][1]
            best_topic = best_meta.get("topic")
            best_unit = best_meta.get("unit")
            best_subtopic = best_meta.get("subtopic")
        else:
            best_topic = best_unit = best_subtopic = None

        # Aggregate all chunks from the same topic/subtopic if context is too short
        context_min_lines = 15
        for doc, meta, _ in top_docs:
            cleaned = clean_text(doc)
            context_parts.append(cleaned)
            sources.add(meta.get("source", "IGNOU Material"))

        context = " ".join(context_parts)
        # If context is too short, fetch more from same topic/subtopic
        if len(context.splitlines()) < context_min_lines and best_topic:
            # Query all docs with same topic/subtopic
            filter_query = {"topic": best_topic}
            if best_unit:
                filter_query["unit"] = best_unit
            if best_subtopic:
                filter_query["subtopic"] = best_subtopic
            extra_results = self.collection.get(where=filter_query, include=["documents", "metadatas"])
            extra_docs = extra_results.get("documents", [])
            extra_metas = extra_results.get("metadatas", [])
            # Flatten and clean
            extra_contexts = []
            for doclist in extra_docs:
                for doc in doclist:
                    cleaned = clean_text(doc)
                    if cleaned not in context_parts:
                        extra_contexts.append(cleaned)
            if extra_contexts:
                context += "\n" + "\n".join(extra_contexts)

        best_similarity = 1.0 - top_docs[0][2] if top_docs else 0.0

        return {
            "found": True,
            "context": context,
            "sources": list(sources),
            "similarity": best_similarity,
        }
