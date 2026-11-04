"""CLI utility to explicitly download and cache RAG embedding models.

Usage:
    python -m core.rag.download_model

Explicitly fetches the ONNX weights and tokenizer files into EMBEDDING_CACHE_DIR,
measures and logs the exact on-disk size in MB, and ensures research runs never
trigger implicit or unexpected network downloads.
"""

import os
import sys
import logging
from pathlib import Path
from fastembed import TextEmbedding

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_CACHE_DIR = "data/models"


def get_dir_size_bytes(dir_path: Path) -> int:
    """Calculate total size in bytes of all files in a directory recursively."""
    if not dir_path.is_dir():
        return 0
    total = 0
    for root, _, files in os.walk(dir_path):
        for f in files:
            fp = os.path.join(root, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total


def download_model(model_name: str, cache_dir: str) -> float:
    """Download model to cache_dir explicitly and return total size in MB."""
    target_path = Path(cache_dir).resolve()
    target_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting explicit download for model '{model_name}' into '{target_path}'...")
    # Initializing without local_files_only triggers the download
    embedding_model = TextEmbedding(
        model_name=model_name,
        cache_dir=str(target_path)
    )
    # Warm up with a test token to ensure ONNX session and tokenizer are fully loaded
    _ = list(embedding_model.embed(["model download verification token"]))

    total_bytes = get_dir_size_bytes(target_path)
    size_mb = total_bytes / (1024 * 1024)
    logger.info(
        f"Model '{model_name}' downloaded successfully.\n"
        f"Cache directory: {target_path}\n"
        f"Total cached size: {size_mb:.2f} MB"
    )
    return size_mb


def main():
    model_name = os.environ.get("EMBEDDING_MODEL", DEFAULT_MODEL)
    cache_dir = os.environ.get("EMBEDDING_CACHE_DIR", DEFAULT_CACHE_DIR)
    try:
        size_mb = download_model(model_name, cache_dir)
        print(f"\n[SUCCESS] Embedding model '{model_name}' is ready ({size_mb:.2f} MB).")
        sys.exit(0)
    except Exception as exc:
        logger.error(f"Failed to download embedding model: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
