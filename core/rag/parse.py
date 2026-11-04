"""Document parsing, format detection, and validation for RAG ingestion.

Detects document types via magic bytes (%PDF- vs UTF-8 text/md),
enforces file size limits, page counts, parsing timeouts, and filename sanitization.
"""

import os
import io
import re
import asyncio
from typing import Tuple, Optional, Any
import pypdf

MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "25"))
MAX_PDF_PAGES = int(os.environ.get("MAX_PDF_PAGES", "200"))
PARSING_TIMEOUT_SECONDS = float(os.environ.get("PARSING_TIMEOUT_SECONDS", "30.0"))


async def stream_and_validate_upload_bytes(
    file_or_stream: Any,
    max_mb: int = MAX_UPLOAD_MB,
    chunk_size: int = 64 * 1024
) -> bytes:
    """Stream upload body enforcing max_mb, aborting immediately if exceeded without buffering remainder.

    Reads in chunk_size increments and halts immediately if total bytes read exceeds max_mb.
    """
    max_bytes = max_mb * 1024 * 1024
    total_read = 0
    buffer = bytearray()

    while True:
        chunk = await file_or_stream.read(chunk_size)
        if not chunk:
            break
        total_read += len(chunk)
        if total_read > max_bytes:
            raise ValueError(f"File size exceeds maximum allowed upload limit of {max_mb} MB")
        buffer.extend(chunk)

    if total_read == 0:
        raise ValueError("Cannot parse empty file")
    return bytes(buffer)


def sanitize_filename(filename: str) -> str:
    """Sanitize client-provided filename to prevent directory traversal and invalid chars.

    Preserves safe extensions (.pdf, .md, .txt) and limits length.
    """
    if not filename:
        return "uploaded_document"

    # Extract base name without path separators
    base = os.path.basename(filename.replace("\\", "/"))
    # Remove null bytes and control chars
    base = re.sub(r"[\x00-\x1f\x7f]", "", base)
    # Replace unsafe characters with underscore
    clean = re.sub(r"[^\w\s\.-]", "_", base).strip(" .")
    # Collapse consecutive spaces or underscores
    clean = re.sub(r"[\s_]+", "_", clean)
    if not clean:
        return "uploaded_document"
    return clean[:100]


def detect_file_type(data: bytes) -> str:
    """Detect whether bytes are PDF or printable UTF-8 text/markdown.

    Uses magic bytes detection, not client-supplied extensions.
    Raises ValueError for unsupported formats.
    """
    if not data:
        raise ValueError("Cannot parse empty file")

    if data.startswith(b"%PDF-"):
        return "pdf"

    # Check for UTF-8 text / markdown
    try:
        decoded_sample = data[:8192].decode("utf-8")
        if "\x00" in decoded_sample:
            raise ValueError("Binary null bytes detected in text content")

        printable_count = sum(1 for c in decoded_sample if c.isprintable() or c in "\r\n\t")
        if len(decoded_sample) > 0 and (printable_count / len(decoded_sample)) > 0.90:
            return "text"
    except UnicodeDecodeError:
        pass

    raise ValueError("Unsupported file format: only PDF and UTF-8 plain text / markdown files are accepted")


def _parse_pdf_sync(data: bytes, max_pages: int = MAX_PDF_PAGES) -> Tuple[str, int]:
    """Synchronous PDF parser enforcing page cap and encryption check."""
    stream = io.BytesIO(data)
    try:
        reader = pypdf.PdfReader(stream)
    except Exception as exc:
        raise ValueError(f"Invalid or corrupted PDF file: {exc}") from exc

    if reader.is_encrypted:
        raise ValueError("Encrypted PDFs are not supported")

    page_count = len(reader.pages)
    if page_count > max_pages:
        raise ValueError(
            f"PDF exceeds maximum page limit of {max_pages} pages (document contains {page_count} pages)"
        )

    text_parts = []
    for idx, page in enumerate(reader.pages):
        try:
            page_text = page.extract_text() or ""
            text_parts.append(page_text.strip())
        except Exception as exc:
            text_parts.append(f"[Error extracting page {idx + 1}: {exc}]")

    full_text = "\n\n".join(part for part in text_parts if part)
    # Strip null characters
    full_text = full_text.replace("\x00", "")
    return full_text.strip(), page_count


async def parse_document_bytes(
    data: bytes,
    filename: Optional[str] = None,
    max_mb: int = MAX_UPLOAD_MB,
    max_pages: int = MAX_PDF_PAGES,
    timeout_sec: float = PARSING_TIMEOUT_SECONDS
) -> Tuple[str, str, int]:
    """Parse raw bytes into extracted text.

    Args:
        data: Raw file bytes
        filename: Optional client filename for metadata
        max_mb: Maximum allowed file size in megabytes
        max_pages: Maximum pages allowed for PDFs
        timeout_sec: Timeout for PDF parsing thread

    Returns:
        (extracted_text, detected_type, page_count)
    """
    max_bytes = max_mb * 1024 * 1024
    if len(data) > max_bytes:
        raise ValueError(f"File size ({len(data) / (1024 * 1024):.1f} MB) exceeds maximum allowed {max_mb} MB")

    kind = detect_file_type(data)

    if kind == "text":
        try:
            text = data.decode("utf-8")
            text = text.replace("\x00", "").strip()
            return text, "text", 1
        except UnicodeDecodeError as exc:
            raise ValueError(f"Failed to decode text as UTF-8: {exc}") from exc

    # PDF parsing in thread with timeout
    try:
        text, page_count = await asyncio.wait_for(
            asyncio.to_thread(_parse_pdf_sync, data, max_pages),
            timeout=timeout_sec
        )
        return text, "pdf", page_count
    except asyncio.TimeoutError:
        raise TimeoutError(f"PDF parsing timed out after {timeout_sec:.1f} seconds")
