from __future__ import annotations

TARGET_CHUNK_TOKENS = 512
CHARS_PER_TOKEN = 4.0


def split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    current: list[str] = []
    for char in text:
        current.append(char)
        if char in {".", "!", "?"}:
            s = "".join(current).strip()
            if s:
                sentences.append(s)
            current = []
    tail = "".join(current).strip()
    if tail:
        sentences.append(tail)
    return sentences


def chunk_text(text: str, target_tokens: int = TARGET_CHUNK_TOKENS) -> list[str]:
    target_chars = int(target_tokens * CHARS_PER_TOKEN)
    if len(text) <= target_chars:
        return [text]

    chunks: list[str] = []
    current = ""

    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        if current and len(current) + len(paragraph) + 2 > target_chars:
            chunks.append(current.strip())
            current = ""

        if len(paragraph) > target_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            for sentence in split_sentences(paragraph):
                if len(current) + len(sentence) + 1 > target_chars:
                    if current:
                        chunks.append(current.strip())
                    current = ""
                if current:
                    current += " "
                current += sentence
        else:
            if current:
                current += "\n\n"
            current += paragraph

    if current.strip():
        chunks.append(current.strip())

    return [c for c in chunks if c]


def normalize_extracted_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    out_parts: list[str] = []

    for block in normalized.split("\n\n"):
        block = block.strip()
        if not block:
            continue

        paragraph = ""
        pending_hyphen = False
        for line in block.splitlines():
            line = line.strip()
            if not line:
                continue

            if pending_hyphen:
                paragraph += line
                pending_hyphen = False
            elif not paragraph:
                paragraph = line
            else:
                paragraph += f" {line}"

            if paragraph.endswith("-"):
                paragraph = paragraph[:-1]
                pending_hyphen = True

        collapsed = " ".join(paragraph.split())
        if collapsed:
            out_parts.append(collapsed)

    return "\n\n".join(out_parts)


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


def map_chunks_to_page_ranges(
    chunks: list[str],
    page_texts: list[str],
    separator: str = "\n\n",
) -> list[tuple[int | None, int | None]]:
    """Best-effort mapping of chunk text to 1-based physical PDF page ranges."""

    if not page_texts:
        return [(None, None) for _ in chunks]

    full_text = separator.join(page_texts)
    if not full_text:
        return [(None, None) for _ in chunks]

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
    search_from = 0

    for chunk in chunks:
        if not chunk:
            spans.append((None, None))
            continue

        start_offset = full_text.find(chunk, search_from)
        if start_offset == -1:
            start_offset = full_text.find(chunk)
        if start_offset == -1:
            spans.append((None, None))
            continue

        end_offset = start_offset + len(chunk) - 1
        page_start = _page_number_for_offset(start_offset, page_ranges)
        page_end = _page_number_for_offset(end_offset, page_ranges)
        spans.append((page_start, page_end))
        search_from = start_offset + len(chunk)

    return spans


__all__ = [
    "TARGET_CHUNK_TOKENS",
    "chunk_text",
    "normalize_extracted_text",
    "map_chunks_to_page_ranges",
]
