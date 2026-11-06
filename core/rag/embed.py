"""Embedding providers and factory for RAG documents and search queries.

Provides the Embedder interface, FastEmbedEmbedder with strict offline/cached-only
enforcement (no implicit downloads during runs), and FakeDeterministicEmbedder for
lightning-fast offline unit tests.
"""

import os
import hashlib
import threading
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Tuple
import numpy as np

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_CACHE_DIR = "data/models"


class Embedder(ABC):
    """Abstract interface for text embedding models."""

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of strings into normalized float32 vectors."""
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Return dimensionality of the embedding vectors."""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the canonical model identifier string."""
        pass


class FakeDeterministicEmbedder(Embedder):
    """Offline, deterministic embedding provider for fast automated tests.

    Produces normalized 384-dimensional vectors derived deterministically from text hashes.
    Guarantees that identical text yields identical vectors and similar text yields higher cosine similarity.
    """

    def __init__(self, model_name: str = "fake-deterministic-384", dim: int = 384):
        self.model_name = model_name
        self.dim = dim

    def get_model_name(self) -> str:
        return self.model_name

    def get_dimension(self) -> int:
        return self.dim

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        results: List[List[float]] = []
        for text in texts:
            words = text.lower().split()
            vec = np.zeros(self.dim, dtype=np.float32)

            # Word-level deterministic hashing
            for word in words:
                h = hashlib.sha256(word.encode("utf-8")).digest()
                for i in range(min(len(h), self.dim)):
                    idx = (i * 7 + int(h[i])) % self.dim
                    val = (int(h[i]) - 128) / 128.0
                    vec[idx] += val

            # Fallback for empty text
            if len(words) == 0:
                h = hashlib.sha256(b"empty").digest()
                for i in range(min(len(h), self.dim)):
                    vec[i % self.dim] = 1.0

            # L2 normalize
            norm = np.linalg.norm(vec)
            if norm > 1e-9:
                vec = vec / norm
            else:
                vec[0] = 1.0

            results.append(vec.tolist())
        return results


class FastEmbedEmbedder(Embedder):
    """ONNX-based local embedding using fastembed.

    Strictly forbids implicit downloads: requires model to have been downloaded
    explicitly beforehand via `python -m core.rag.download_model`.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        self.model_name = model_name or os.environ.get("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        self.cache_dir = cache_dir or os.environ.get("EMBEDDING_CACHE_DIR", DEFAULT_CACHE_DIR)
        self._dim: Optional[int] = None
        self._model = None
        self._init_model()

    def _init_model(self) -> None:
        if not os.path.exists(self.cache_dir) or not os.listdir(self.cache_dir):
            raise RuntimeError(
                f"Embedding model '{self.model_name}' was not found in cache directory '{self.cache_dir}'.\n"
                f"Research runs must NEVER trigger implicit model downloads.\n"
                f"Please download the model explicitly first by executing:\n"
                f"    python -m core.rag.download_model\n"
                f"Error detail: Cache directory is missing or empty."
            )

        from fastembed import TextEmbedding
        os.makedirs(self.cache_dir, exist_ok=True)
        try:
            # Enforce local_files_only: never download implicitly in a run
            self._model = TextEmbedding(
                model_name=self.model_name,
                cache_dir=self.cache_dir,
                local_files_only=True
            )
        except Exception as exc:
            raise RuntimeError(
                f"Embedding model '{self.model_name}' was not found in cache directory '{self.cache_dir}'.\n"
                f"Research runs must NEVER trigger implicit model downloads.\n"
                f"Please download the model explicitly first by executing:\n"
                f"    python -m core.rag.download_model\n"
                f"Error detail: {exc}"
            ) from exc

    def get_model_name(self) -> str:
        return self.model_name

    def get_dimension(self) -> int:
        if self._dim is None:
            # Probe dimension by embedding a test token
            vecs = self.embed_texts(["dim_probe"])
            self._dim = len(vecs[0]) if vecs else 384
        return self._dim

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        embeddings_gen = self._model.embed(texts)
        results: List[List[float]] = []
        for arr in embeddings_gen:
            vec = np.asarray(arr, dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 1e-9:
                vec = vec / norm
            results.append(vec.tolist())
        return results


_EMBEDDER_SINGLETONS: Dict[Tuple[bool, str, str], Embedder] = {}
_EMBEDDER_LOCK = threading.Lock()


def reset_embedder_cache() -> None:
    """Clear all cached embedder singleton instances in a thread-safe manner."""
    with _EMBEDDER_LOCK:
        _EMBEDDER_SINGLETONS.clear()


def get_embedder(model_name: Optional[str] = None) -> Embedder:
    """Factory creating and caching configured Embedder singleton based on environment.

    Thread-safe double-checked locking around creation.
    When USE_FAKE_EMBEDDER is active or configured_model is a fake model,
    strictly returns FakeDeterministicEmbedder without loading FastEmbed or touching ONNX.
    """
    configured_model = model_name or os.environ.get("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    use_fake = (
        os.environ.get("USE_FAKE_EMBEDDER", "0").lower() in ("1", "true", "yes")
        or configured_model in ("fake", "fake-deterministic-384")
    )
    cache_dir = os.environ.get("EMBEDDING_CACHE_DIR", DEFAULT_CACHE_DIR)

    key = (use_fake, configured_model, cache_dir)
    # Fast path without lock
    if key in _EMBEDDER_SINGLETONS:
        return _EMBEDDER_SINGLETONS[key]

    with _EMBEDDER_LOCK:
        if key in _EMBEDDER_SINGLETONS:
            return _EMBEDDER_SINGLETONS[key]

        if use_fake:
            # Pure hashing/in-memory embedder; strictly never touches ONNX or fastembed
            embedder: Embedder = FakeDeterministicEmbedder(model_name=configured_model)
        else:
            embedder = FastEmbedEmbedder(model_name=configured_model, cache_dir=cache_dir)

        _EMBEDDER_SINGLETONS[key] = embedder
        return embedder
