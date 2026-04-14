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
                f"{settings.RAG_SIMILARITY_THRESHOLD}), returning no answer"
            )
            return {
                "found": False,
                "context": "Sorry, no relevant content was found in the MAPC documents.",
                "sources": [],
                "similarity": best_similarity,
            }

        docs = results["documents"][0]
        metadatas = results["metadatas"][0]

        import re
        def clean_text(text):
            # Remove [Unit-1.pdf] or similar at the start of any line
            text = re.sub(r"^\s*\[.*?\]\s*", "", text, flags=re.MULTILINE)
            # Remove section numbers like 1.2.2, 2.1, 3. etc. anywhere in the text
            text = re.sub(r"(\b\d+(?:[\.-]\d+)+\b)", "", text)
            # Remove single numbers surrounded by spaces (not years)
            text = re.sub(r"(?<!\d)(\b\d{1,2}\b)(?!\d)", "", text)
            # Remove excessive dashes/underscores at the start of any line
            text = re.sub(r"(?m)^[-_]+", "", text)

            # Format side headings: lines that are all uppercase or short (<=7 words, not ending with punctuation)
            lines = text.splitlines()
            formatted = []
            for i, line in enumerate(lines):
                stripped = line.strip()
                # Heading: all uppercase and not too long, or short and not ending with .!?
                if (stripped.isupper() and 2 <= len(stripped) <= 60) or (len(stripped.split()) <= 7 and not stripped.endswith(('.', '!', '?')) and stripped):
                    # Add a blank line before headings except at the top
                    if formatted and formatted[-1] != '':
                        formatted.append("")
                    formatted.append(f"**{stripped}**")
                else:
                    formatted.append(stripped)
            return "\n".join(formatted).strip()

        # Build ranked context — closest chunk first, clean up each chunk
        context_parts = []
        for doc in docs:
            cleaned = clean_text(doc)
            context_parts.append(cleaned)

        context = "\n\n---\n\n".join(context_parts)
        sources = list({m.get("source", "IGNOU Material") for m in metadatas})

        return {
            "found": True,
            "context": context,
            "sources": sources,
            "similarity": best_similarity,
        }
