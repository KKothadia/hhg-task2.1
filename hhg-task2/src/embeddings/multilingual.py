"""
Multilingual embedding service.

Wraps sentence-transformers model with:
- Singleton loading (model loaded once at startup, reused everywhere)
- Batch encoding for ingestion
- Single-query encoding for retrieval (fast path)
- GPU auto-detection
"""

import time
import numpy as np
import structlog
from sentence_transformers import SentenceTransformer

from src.config import settings

logger = structlog.get_logger(__name__)


class EmbeddingService:
    """Multilingual embedding model wrapper with singleton pattern."""

    _instance: "EmbeddingService | None" = None
    _model: SentenceTransformer | None = None

    def __new__(cls):
        """Singleton — only one model instance loaded in memory."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load_model(self):
        """Load the embedding model into memory. Call once at startup."""
        if self._model is not None:
            logger.info("embedding_model_already_loaded")
            return

        start = time.perf_counter()
        import torch
        import os

        # Optimize PyTorch CPU thread allocation for low-memory container execution
        if not torch.cuda.is_available():
            torch.set_num_threads(1)
            try:
                torch.set_num_interop_threads(1)
            except Exception:
                pass

        device = "cuda" if torch.cuda.is_available() else "cpu"

        self._model = SentenceTransformer(
            settings.embedding_model,
            device=device,
        )

        # Warm up JIT and PyTorch execution graph
        self._model.encode("warmup query", convert_to_numpy=True, show_progress_bar=False)

        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "embedding_model_loaded",
            model=settings.embedding_model,
            device=device,
            dimension=settings.embedding_dimension,
            load_time_ms=round(duration_ms, 2),
        )

    @property
    def model(self) -> SentenceTransformer:
        """Get the loaded model, raising if not yet loaded."""
        if self._model is None:
            raise RuntimeError("Embedding model not loaded. Call load_model() at startup.")
        return self._model

    def encode_query(self, query: str) -> list[float]:
        """
        Encode a single query string. Optimized for low latency (single inference).

        Args:
            query: The search query text.

        Returns:
            Embedding vector as list of floats.
        """
        start = time.perf_counter()
        embedding = self.model.encode(query, convert_to_numpy=True, show_progress_bar=False)
        duration_ms = (time.perf_counter() - start) * 1000

        logger.debug("query_embedded", duration_ms=round(duration_ms, 2), text_length=len(query))

        return embedding.tolist()

    def encode_batch(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        """
        Encode a batch of texts. Optimized for throughput during ingestion.

        Args:
            texts: List of text strings to encode.
            batch_size: Number of texts to encode per batch.

        Returns:
            List of embedding vectors.
        """
        start = time.perf_counter()
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        duration_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "batch_embedded",
            count=len(texts),
            batch_size=batch_size,
            duration_ms=round(duration_ms, 2),
            throughput=round(len(texts) / (duration_ms / 1000), 1) if duration_ms > 0 else 0,
        )

        return embeddings.tolist()

    def compute_similarity(self, vec_a: list[float], vec_b: list[float]) -> float:
        """Compute cosine similarity between two embedding vectors."""
        a = np.array(vec_a)
        b = np.array(vec_b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
