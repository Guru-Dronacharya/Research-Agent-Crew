"""Recursive text chunking and content hashing for RAG documents.

Splits text hierarchically across paragraph boundaries, sentences, and words
with configurable chunk size and overlap, computing deterministic SHA-256 hashes.
"""

import os
import re
import hashlib
from typing import List, Dict, Any

CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "100"))


def compute_content_hash(text: str) -> str:
    """Compute deterministic SHA-256 hash of text content."""
    normalized = text.strip().encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
) -> List[Dict[str, Any]]:
    """Recursively split text into overlapping chunks respecting structural boundaries.

    Args:
        text: Input string to partition
        chunk_size: Maximum character length per chunk
        chunk_overlap: Target character overlap between consecutive chunks

    Returns:
        List of chunk dictionaries with idx, text, token_count, and char_count.
    """
    clean_text = text.strip()
    if not clean_text:
        return []

    if len(clean_text) <= chunk_size:
        return [{
            "idx": 0,
            "text": clean_text,
            "token_count": max(1, len(clean_text.split())),
            "char_count": len(clean_text)
        }]

    # Paragraph-level split first
    paragraphs = re.split(r"\n\s*\n", clean_text)
    raw_pieces: List[str] = []

    for para in paragraphs:
        p_clean = para.strip()
        if not p_clean:
            continue

        if len(p_clean) <= chunk_size:
            raw_pieces.append(p_clean)
        else:
            # Split long paragraph by sentence boundaries
            sentences = re.split(r"(?<=[.!?])\s+", p_clean)
            for sent in sentences:
                s_clean = sent.strip()
                if not s_clean:
                    continue
                if len(s_clean) <= chunk_size:
                    raw_pieces.append(s_clean)
                else:
                    # Split very long sentences by spaces
                    words = s_clean.split(" ")
                    buf: List[str] = []
                    buf_len = 0
                    for w in words:
                        if buf_len + len(w) + 1 > chunk_size and buf:
                            raw_pieces.append(" ".join(buf))
                            buf = [w]
                            buf_len = len(w)
                        else:
                            buf.append(w)
                            buf_len += len(w) + 1
                    if buf:
                        raw_pieces.append(" ".join(buf))

    # Assemble raw pieces into chunks with overlap
    chunks: List[Dict[str, Any]] = []
    current_chunk: List[str] = []
    current_len = 0

    for piece in raw_pieces:
        added_len = len(piece) + (2 if current_chunk else 0)
        if current_len + added_len > chunk_size and current_chunk:
            chunk_str = "\n\n".join(current_chunk).strip()
            chunks.append({
                "idx": len(chunks),
                "text": chunk_str,
                "token_count": max(1, len(chunk_str.split())),
                "char_count": len(chunk_str)
            })

            # Form overlap from end of current chunk
            overlap_pieces: List[str] = []
            overlap_len = 0
            for prev in reversed(current_chunk):
                if overlap_len + len(prev) <= chunk_overlap:
                    overlap_pieces.insert(0, prev)
                    overlap_len += len(prev)
                else:
                    break

            current_chunk = overlap_pieces
            current_len = sum(len(p) for p in current_chunk) + max(0, (len(current_chunk) - 1) * 2)

        current_chunk.append(piece)
        current_len += len(piece) + (2 if len(current_chunk) > 1 else 0)

    if current_chunk:
        chunk_str = "\n\n".join(current_chunk).strip()
        chunks.append({
            "idx": len(chunks),
            "text": chunk_str,
            "token_count": max(1, len(chunk_str.split())),
            "char_count": len(chunk_str)
        })

    return chunks
