"""Streaming citation buffer and filter.

Buffers citation markers like `[chunk-1]` that may be split across multiple
streaming token deltas, validates cited IDs against the retrieved chunk set,
emits only valid citation markers, drops hallucinated/invalid markers, and
compiles the final citation mapping.
"""

from typing import Set, Dict, List, Any


class CitationStreamFilter:
    """Stateful stream filter that validates citations across streaming chunks."""

    def __init__(self, valid_chunk_ids: Set[str]):
        self.valid_chunk_ids: Set[str] = set(valid_chunk_ids)
        self.cited_ids: Set[str] = set()
        self.buffer: str = ""
        self.in_bracket: bool = False

    def process_delta(self, delta: str) -> List[str]:
        """Process an incoming delta string and return sanitized text deltas to emit."""
        emitted: List[str] = []
        i = 0
        n = len(delta)

        while i < n:
            ch = delta[i]

            if not self.in_bracket:
                if ch == "[":
                    self.in_bracket = True
                    self.buffer = "["
                    i += 1
                else:
                    # Collect contiguous non-bracket text
                    start = i
                    while i < n and delta[i] != "[":
                        i += 1
                    emitted.append(delta[start:i])
            else:
                # We are buffering inside a bracket
                if ch == "]":
                    self.buffer += "]"
                    self.in_bracket = False
                    candidate_id = self.buffer[1:-1].strip()

                    # Validate if candidate_id is one of our retrieved chunk IDs
                    if candidate_id in self.valid_chunk_ids:
                        self.cited_ids.add(candidate_id)
                        emitted.append(f"[{candidate_id}]")
                    else:
                        # Drop invalid / hallucinated citation ID
                        pass

                    self.buffer = ""
                    i += 1
                elif ch == "\n" or len(self.buffer) > 80:
                    # Newline or oversized buffer means this is not a citation marker
                    # (e.g. markdown link or regular bracketed text); flush buffer
                    self.buffer += ch
                    emitted.append(self.buffer)
                    self.buffer = ""
                    self.in_bracket = False
                    i += 1
                else:
                    self.buffer += ch
                    i += 1

        return emitted

    def finalize(self) -> List[str]:
        """Flush any trailing unclosed buffer at the end of the stream."""
        if self.buffer:
            flushed = [self.buffer]
            self.buffer = ""
            self.in_bracket = False
            return flushed
        return []

    def build_citations(self, chunk_map: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Map cited IDs to their title, URL, and page metadata for the final citation event."""
        citations: List[Dict[str, Any]] = []
        for cid in sorted(self.cited_ids):
            info = chunk_map.get(cid, {})
            citations.append({
                "id": cid,
                "title": info.get("title", cid),
                "url": info.get("url", ""),
                "source_urls": info.get("source_urls", [info.get("url", "")] if info.get("url") else []),
                "page": info.get("page")
            })
        return citations
