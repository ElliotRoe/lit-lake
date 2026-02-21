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


__all__ = ["TARGET_CHUNK_TOKENS", "chunk_text", "normalize_extracted_text"]
