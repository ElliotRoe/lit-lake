from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from litlake.providers.extraction import ExtractionResult


TARGET_CHUNK_TOKENS = 512
CHARS_PER_TOKEN = 4.0


def split_sentence_spans(text: str, base_offset: int = 0) -> list[tuple[str, int, int]]:
    spans: list[tuple[str, int, int]] = []
    sentence_start = 0

    for idx, char in enumerate(text):
        if char not in {".", "!", "?"}:
            continue
        raw = text[sentence_start : idx + 1]
        stripped = raw.strip()
        if stripped:
            lead_ws = len(raw) - len(raw.lstrip())
            trail_ws = len(raw) - len(raw.rstrip())
            start = base_offset + sentence_start + lead_ws
            end = base_offset + idx + 1 - trail_ws
            spans.append((stripped, start, end))
        sentence_start = idx + 1

    tail = text[sentence_start:]
    stripped_tail = tail.strip()
    if stripped_tail:
        lead_ws = len(tail) - len(tail.lstrip())
        trail_ws = len(tail) - len(tail.rstrip())
        start = base_offset + sentence_start + lead_ws
        end = base_offset + len(text) - trail_ws
        spans.append((stripped_tail, start, end))

    return spans


def _paragraph_spans(text: str) -> list[tuple[str, int, int]]:
    parts: list[tuple[str, int, int]] = []
    cursor = 0

    while True:
        split_at = text.find("\n\n", cursor)
        if split_at == -1:
            raw_start, raw_end = cursor, len(text)
        else:
            raw_start, raw_end = cursor, split_at

        raw = text[raw_start:raw_end]
        stripped = raw.strip()
        if stripped:
            lead_ws = len(raw) - len(raw.lstrip())
            trail_ws = len(raw) - len(raw.rstrip())
            start = raw_start + lead_ws
            end = raw_end - trail_ws
            parts.append((stripped, start, end))

        if split_at == -1:
            break
        cursor = split_at + 2

    return parts


def chunk_text_with_spans(
    text: str, target_tokens: int = TARGET_CHUNK_TOKENS
) -> list[tuple[str, int, int]]:
    target_chars = int(target_tokens * CHARS_PER_TOKEN)
    if len(text) <= target_chars:
        return [(text, 0, len(text))]

    chunks: list[tuple[str, int, int]] = []
    current_text = ""
    current_start: int | None = None
    current_end: int | None = None

    def flush_current() -> None:
        nonlocal current_text, current_start, current_end
        if (
            current_text.strip()
            and current_start is not None
            and current_end is not None
            and current_end > current_start
        ):
            chunks.append((current_text.strip(), current_start, current_end))
        current_text = ""
        current_start = None
        current_end = None

    def append_piece(piece: str, piece_start: int, piece_end: int, sep: str = "") -> None:
        nonlocal current_text, current_start, current_end
        if not piece:
            return
        if current_text:
            current_text += sep + piece
            current_end = piece_end
        else:
            current_text = piece
            current_start = piece_start
            current_end = piece_end

    for paragraph, para_start, _para_end in _paragraph_spans(text):
        if current_text and len(current_text) + len(paragraph) + 2 > target_chars:
            flush_current()

        if len(paragraph) > target_chars:
            if current_text:
                flush_current()
            for sentence, sent_start, sent_end in split_sentence_spans(paragraph, para_start):
                if current_text and len(current_text) + len(sentence) + 1 > target_chars:
                    flush_current()
                append_piece(
                    sentence,
                    sent_start,
                    sent_end,
                    sep=" " if current_text else "",
                )
        else:
            append_piece(
                paragraph,
                para_start,
                para_start + len(paragraph),
                sep="\n\n" if current_text else "",
            )

    flush_current()
    return chunks


def _page_number_for_offset(
    offset: int,
    page_ranges: list[tuple[int, int, int]],
) -> int | None:
    if not page_ranges:
        return None

    for idx, (page_num, start, end) in enumerate(page_ranges):
        if start <= offset < end:
            return page_num

        if idx < len(page_ranges) - 1:
            next_start = page_ranges[idx + 1][1]
            if end <= offset < next_start:
                return page_ranges[idx + 1][0]

    if offset >= page_ranges[-1][2]:
        return page_ranges[-1][0]

    return None


def map_chunk_spans_to_page_ranges(
    chunk_spans: list[tuple[str, int, int]],
    page_texts: list[str],
    separator: str = "\n\n",
) -> list[tuple[int | None, int | None]]:
    if not page_texts:
        return [(None, None) for _ in chunk_spans]

    full_text = separator.join(page_texts)
    if not full_text:
        return [(None, None) for _ in chunk_spans]

    page_ranges: list[tuple[int, int, int]] = []
    cursor = 0
    for idx, page_text in enumerate(page_texts, start=1):
        start = cursor
        end = start + len(page_text)
        page_ranges.append((idx, start, end))
        cursor = end
        if idx < len(page_texts):
            cursor += len(separator)

    spans: list[tuple[int | None, int | None]] = []

    for _chunk_text, start_offset, end_offset_exclusive in chunk_spans:
        if end_offset_exclusive <= start_offset:
            spans.append((None, None))
            continue

        end_offset = end_offset_exclusive - 1
        page_start = _page_number_for_offset(start_offset, page_ranges)
        page_end = _page_number_for_offset(end_offset, page_ranges)
        spans.append((page_start, page_end))

    return spans


class ChunkingValidationError(ValueError):
    """Raised when chunking output violates invariants."""


@dataclass(frozen=True)
class ChunkArtifact:
    content: str
    page_start: int | None = None
    page_end: int | None = None


class ChunkingProvider(Protocol):
    mime_type: str

    def chunk(self, extraction: ExtractionResult) -> list[ChunkArtifact]:
        ...


@dataclass
class PdfChunkingProvider:
    mime_type: str = "application/pdf"

    def chunk(self, extraction: ExtractionResult) -> list[ChunkArtifact]:
        text = extraction.text or ""
        if not text.strip():
            raise ChunkingValidationError("Normalized extracted PDF text is empty")

        chunk_spans = chunk_text_with_spans(text, TARGET_CHUNK_TOKENS)
        if not chunk_spans:
            raise ChunkingValidationError("PDF chunker produced no chunks")

        page_spans = (
            map_chunk_spans_to_page_ranges(chunk_spans, extraction.page_texts)
            if extraction.page_texts is not None
            else [(None, None) for _ in chunk_spans]
        )

        artifacts: list[ChunkArtifact] = []
        for idx, (chunk_text, _, _) in enumerate(chunk_spans):
            if not chunk_text or not chunk_text.strip():
                raise ChunkingValidationError(
                    f"PDF chunker produced empty chunk at index={idx}"
                )
            page_start, page_end = page_spans[idx]
            artifacts.append(
                ChunkArtifact(
                    content=chunk_text,
                    page_start=page_start,
                    page_end=page_end,
                )
            )

        return artifacts


@dataclass
class HtmlChunkingProvider:
    mime_type: str = "text/html"

    def chunk(self, extraction: ExtractionResult) -> list[ChunkArtifact]:
        text = extraction.text or ""
        if not text.strip():
            raise ChunkingValidationError("Normalized extracted HTML text is empty")

        chunk_spans = chunk_text_with_spans(text, TARGET_CHUNK_TOKENS)
        if not chunk_spans:
            raise ChunkingValidationError("HTML chunker produced no chunks")

        artifacts: list[ChunkArtifact] = []
        for idx, (chunk_text, _, _) in enumerate(chunk_spans):
            if not chunk_text or not chunk_text.strip():
                raise ChunkingValidationError(
                    f"HTML chunker produced empty chunk at index={idx}"
                )
            artifacts.append(ChunkArtifact(content=chunk_text))

        return artifacts


__all__ = [
    "chunk_text_with_spans",
    "ChunkArtifact",
    "ChunkingProvider",
    "ChunkingValidationError",
    "HtmlChunkingProvider",
    "map_chunk_spans_to_page_ranges",
    "PdfChunkingProvider",
    "split_sentence_spans",
    "TARGET_CHUNK_TOKENS",
]
