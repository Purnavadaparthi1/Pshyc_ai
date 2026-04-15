"""
PSYCH.AI — Document Ingestion Script
-------------------------------------
Ingests IGNOU course PDFs and text files into the ChromaDB vector store.

Usage:
    python scripts/ingest.py                        # ingest all docs in data/documents/
    python scripts/ingest.py --file path/to/doc.pdf # ingest single file
    python scripts/ingest.py --reset                # wipe collection and re-ingest

Supported formats: .pdf, .txt, .md
"""

import argparse
import hashlib
import logging
import sys
import os
import re
from pathlib import Path

# Allow importing from backend/
sys.path.insert(0, str(Path(__file__).parent.parent))

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from core.config import settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger("ingest")

CHUNK_SIZE     = 850    # characters  # ~700–900 tokens (approx. 1.2 chars/token)
CHUNK_OVERLAP  = 125    # characters  # ~100–150 tokens
DOCS_DIR       = Path(__file__).parent.parent / "data" / "documents"
MAPC_DIR       = Path(__file__).parent.parent / "data" / "MAPC"


# ── Text extraction ────────────────────────────────────────────────────────

def extract_text_pdf(path: Path) -> str:
    try:
        import pypdf
        reader = pypdf.PdfReader(str(path))
        pages = [p.extract_text() or "" for p in reader.pages]
        return "\n\n".join(pages)
    except Exception as e:
        logger.error(f"PDF extraction failed for {path.name}: {e}")
        return ""


def extract_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return extract_text_pdf(path)
    elif ext in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    else:
        logger.warning(f"Unsupported format: {path.name} — skipping")
        return ""


import re
from typing import List, Dict, Tuple

def extract_headings_and_chunks(text: str) -> List[Tuple[str, str, str, str]]:
    """
    Splits text into topic-aware chunks and extracts Unit, Topic, Subtopic as metadata.
    Returns a list of (unit, topic, subtopic, chunk_text) tuples.
    """
    # Regex patterns for headings (customize as needed)
    unit_pat = re.compile(r'^(UNIT[\s\-]*\d+|UNIT[\s\-]*[A-Z]+)', re.MULTILINE)
    topic_pat = re.compile(r'^(TOPIC[\s\-]*\d*:?|[A-Z][A-Z\s]{5,})', re.MULTILINE)
    subtopic_pat = re.compile(r'^(SUBTOPIC[\s\-]*\d*:?|[A-Z][A-Z\s]{5,})', re.MULTILINE)

    # Find all unit headings
    units = [(m.start(), m.group().strip()) for m in unit_pat.finditer(text)]
    if not units:
        units = [(0, "Unit-Unknown")]
    units.append((len(text), None))

    chunks = []
    for i in range(len(units) - 1):
        unit_start, unit_heading = units[i]
        unit_end = units[i+1][0]
        unit_text = text[unit_start:unit_end]

        # Find topics within this unit
        topics = [(m.start(), m.group().strip()) for m in topic_pat.finditer(unit_text)]
        if not topics:
            topics = [(0, "Topic-Unknown")]
        topics.append((len(unit_text), None))

        for j in range(len(topics) - 1):
            topic_start, topic_heading = topics[j]
            topic_end = topics[j+1][0]
            topic_text = unit_text[topic_start:topic_end]

            # Find subtopics within this topic
            subtopics = [(m.start(), m.group().strip()) for m in subtopic_pat.finditer(topic_text)]
            if not subtopics:
                subtopics = [(0, "Subtopic-Unknown")]
            subtopics.append((len(topic_text), None))

            for k in range(len(subtopics) - 1):
                sub_start, sub_heading = subtopics[k]
                sub_end = subtopics[k+1][0]
                chunk = topic_text[sub_start:sub_end].strip()
                # Only keep non-empty chunks
                if chunk:
                    chunks.append((unit_heading, topic_heading, sub_heading, chunk))
    return chunks

import re
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into chunks that always end at sentence boundaries and never cut off words or sentences.
    Overlap is applied at the sentence level, never splitting sentences or words.
    """
    # Split text into sentences using regex (handles ., !, ?)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_len = 0
    for i, sentence in enumerate(sentences):
        sentence = sentence.strip()
        if not sentence:
            continue
        # If adding the next sentence would exceed the chunk size, start a new chunk
        if current_len + len(sentence) + (1 if current_chunk else 0) > chunk_size:
            if current_chunk:
                chunk = ' '.join(current_chunk).strip()
                # Clean up whitespace and broken words
                chunk = re.sub(r'\s+', ' ', chunk)
                chunks.append(chunk)
            # Start new chunk, possibly with overlap
            if overlap > 0 and chunks:
                # Add overlap sentences from previous chunk
                overlap_sents = []
                overlap_len = 0
                for prev_sentence in reversed(current_chunk):
                    overlap_len += len(prev_sentence) + 1
                    overlap_sents.insert(0, prev_sentence)
                    if overlap_len >= overlap:
                        break
                current_chunk = overlap_sents[:]
                current_len = sum(len(s) + 1 for s in current_chunk)
            else:
                current_chunk = []
                current_len = 0
        current_chunk.append(sentence)
        current_len += len(sentence) + (1 if current_chunk else 0)
    if current_chunk:
        chunk = ' '.join(current_chunk).strip()
        chunk = re.sub(r'\s+', ' ', chunk)
        chunks.append(chunk)
    return chunks


# ── Main ingestion ─────────────────────────────────────────────────────────

def ingest_files(files: list[Path], reset: bool = False):
    logger.info(f"Connecting to ChromaDB at: {settings.CHROMA_PERSIST_DIR}")
    client = chromadb.PersistentClient(
        path=settings.CHROMA_PERSIST_DIR,
        settings=ChromaSettings(anonymized_telemetry=False),
    )

    if reset:
        logger.warning("Resetting collection 'psych_ai_docs'…")
        try:
            client.delete_collection("psych_ai_docs")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name="psych_ai_docs",
        metadata={"hnsw:space": "cosine"},
    )

    logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
    embedder = SentenceTransformer(settings.EMBEDDING_MODEL)

    total_chunks = 0
    for path in files:
        logger.info(f"Processing: {path.name}")
        text = extract_text(path)
        if not text.strip():
            logger.warning(f"  → Empty content, skipping.")
            continue

        chunks = chunk_text(text)
        logger.info(f"  → {len(chunks)} chunks")

        ids, embeddings, documents, metadatas = [], [], [], []
        for i, chunk in enumerate(chunks):
            # Stable ID based on file name + chunk index
            chunk_id = hashlib.md5(f"{path.name}::{i}::{chunk[:50]}".encode()).hexdigest()
            emb = embedder.encode(chunk).tolist()

            ids.append(chunk_id)
            embeddings.append(emb)
            documents.append(chunk)
            metadatas.append({
                "source": path.name,
                "chunk_index": i,
                "file_path": str(path),
            })

        # Upsert in batches of 100
        batch_size = 100
        for b_start in range(0, len(ids), batch_size):
            collection.upsert(
                ids=ids[b_start : b_start + batch_size],
                embeddings=embeddings[b_start : b_start + batch_size],
                documents=documents[b_start : b_start + batch_size],
                metadatas=metadatas[b_start : b_start + batch_size],
            )

        total_chunks += len(chunks)
        logger.info(f"  ✓ {path.name} ingested ({len(chunks)} chunks)")

    logger.info("═" * 50)
    logger.info(f"Ingestion complete: {total_chunks} chunks added")
    logger.info(f"Total in collection: {collection.count()}")


def main():
    parser = argparse.ArgumentParser(description="PSYCH.AI document ingestion")
    parser.add_argument("--file", type=str, help="Path to a single file to ingest")
    parser.add_argument("--reset", action="store_true", help="Wipe and re-ingest")
    args = parser.parse_args()
    logger.info("[DEBUG] Starting ingest.py main()")
    if args.file:
        files = [Path(args.file)]
    else:
        files = []
        # Only ingest from MAPC dir
        files += list(MAPC_DIR.glob("**/*.pdf"))
        files += list(MAPC_DIR.glob("**/*.txt"))
        files += list(MAPC_DIR.glob("**/*.md"))
        logger.info(f"[DEBUG] Found {len(files)} files for ingestion in {MAPC_DIR}")
        if not files:
            logger.warning(
                f"No documents found in {MAPC_DIR}.\n"
                "Place your MAPC PDF/TXT files there and re-run."
            )
            return

    ingest_files(files, reset=args.reset)


if __name__ == "__main__":
    main()
