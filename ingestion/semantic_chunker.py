"""
Semantic Chunking for unstructured/messy data.

Algorithm:
  1. Split text into sentences (Vietnamese-aware)
  2. Compute embeddings per sentence (lightweight, cached)
  3. Calculate cosine similarity between consecutive sentence embeddings
  4. When similarity drops below threshold → create chunk boundary
  5. Enforce min/max chunk sizes with sentence overlap
"""
import re
from typing import Optional

import numpy as np

from src.core.config import get_settings
from src.core.logger import get_logger
from src.ingestion.preprocessor import preprocess_text, split_sentences_vietnamese
from src.models.chunk import ChunkMetadata, ProcessedChunk

logger = get_logger(__name__)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class SemanticChunker:
    """
    Chunks text based on semantic similarity between sentences.
    Ideal for unstructured data: FAQ, brochures, blog posts, raw text.
    """

    def __init__(
        self,
        embedding_fn=None,
        similarity_threshold: float = 0.5,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1500,
        overlap_sentences: int = 2,
    ):
        self.embedding_fn = embedding_fn
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap_sentences = overlap_sentences

        # Load config overrides
        settings = get_settings()
        sem_cfg = settings.chunking_config.get("semantic", {})
        self.similarity_threshold = sem_cfg.get("similarity_threshold", similarity_threshold)
        self.min_chunk_size = sem_cfg.get("min_chunk_size", min_chunk_size)
        self.max_chunk_size = sem_cfg.get("max_chunk_size", max_chunk_size)
        self.overlap_sentences = sem_cfg.get("overlap_sentences", overlap_sentences)

    def chunk(
        self,
        text: str,
        source: str,
        metadata_extra: Optional[dict] = None,
    ) -> list[ProcessedChunk]:
        """
        Split text into semantic chunks.

        Args:
            text: Raw text to chunk
            source: Source file name
            metadata_extra: Additional metadata fields

        Returns:
            List of ProcessedChunk objects
        """
        # Preprocess
        text = preprocess_text(text)
        sentences = split_sentences_vietnamese(text)

        if not sentences:
            return []

        logger.info("semantic_chunking_start", source=source, num_sentences=len(sentences))

        # If no embedding function, fallback to fixed-size chunking
        if self.embedding_fn is None:
            return self._fallback_fixed_chunk(sentences, source, metadata_extra)

        # Compute sentence embeddings
        embeddings = self.embedding_fn(sentences)

        # Find chunk boundaries based on similarity drops
        boundaries = self._find_boundaries(embeddings)

        # Create chunks from boundaries
        chunks = self._create_chunks(sentences, boundaries, source, metadata_extra)

        logger.info("semantic_chunking_done", source=source, num_chunks=len(chunks))
        return chunks

    def _find_boundaries(self, embeddings: list[np.ndarray]) -> list[int]:
        """Find indices where semantic similarity drops below threshold."""
        boundaries = [0]
        for i in range(1, len(embeddings)):
            sim = _cosine_similarity(embeddings[i - 1], embeddings[i])
            if sim < self.similarity_threshold:
                boundaries.append(i)
        return boundaries

    def _create_chunks(
        self,
        sentences: list[str],
        boundaries: list[int],
        source: str,
        metadata_extra: Optional[dict],
    ) -> list[ProcessedChunk]:
        """Create ProcessedChunk objects from sentence groups."""
        chunks = []
        for idx, start in enumerate(boundaries):
            end = boundaries[idx + 1] if idx + 1 < len(boundaries) else len(sentences)

            # Add overlap from previous chunk
            overlap_start = max(0, start - self.overlap_sentences)
            chunk_sentences = sentences[overlap_start:end]
            content = " ".join(chunk_sentences)

            # Enforce size limits
            if len(content) < self.min_chunk_size and chunks:
                # Merge with previous chunk
                prev = chunks[-1]
                merged = prev.content + " " + content
                if len(merged) <= self.max_chunk_size:
                    chunks[-1] = ProcessedChunk(
                        content=merged,
                        metadata=prev.metadata,
                    )
                    continue

            # Trim if exceeding max
            if len(content) > self.max_chunk_size:
                content = content[: self.max_chunk_size]

            meta = ChunkMetadata(
                source=source,
                **(metadata_extra or {}),
            )
            chunks.append(ProcessedChunk(content=content, metadata=meta))

        return chunks

    def _fallback_fixed_chunk(
        self,
        sentences: list[str],
        source: str,
        metadata_extra: Optional[dict],
    ) -> list[ProcessedChunk]:
        """Fallback: fixed-size chunking when no embedding function available."""
        chunks = []
        current = []
        current_len = 0

        for sent in sentences:
            if current_len + len(sent) > self.max_chunk_size and current:
                content = " ".join(current)
                meta = ChunkMetadata(source=source, **(metadata_extra or {}))
                chunks.append(ProcessedChunk(content=content, metadata=meta))

                # Overlap
                current = current[-self.overlap_sentences:]
                current_len = sum(len(s) for s in current)

            current.append(sent)
            current_len += len(sent)

        # Final chunk
        if current:
            content = " ".join(current)
            meta = ChunkMetadata(source=source, **(metadata_extra or {}))
            chunks.append(ProcessedChunk(content=content, metadata=meta))

        return chunks
