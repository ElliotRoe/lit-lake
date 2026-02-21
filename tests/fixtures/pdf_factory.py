from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject


MB = 1024 * 1024


@dataclass(frozen=True)
class PdfFixtureSpec:
    name: str
    path: Path
    expected_outcome: Literal["success", "failure"]
    min_size_bytes: int


def _escape_pdf_text(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace("(", "\\(")
        .replace(")", "\\)")
        .replace("\n", " ")
        .strip()
    )


def create_valid_pdf(
    path: Path,
    *,
    pages: int,
    text_seed: str,
    min_size_bytes: int = 0,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)

    writer = PdfWriter()
    font_obj = writer._add_object(  # type: ignore[attr-defined]
        DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
    )

    for idx in range(max(1, pages)):
        page = writer.add_blank_page(width=612, height=792)
        resources = DictionaryObject(
            {
                NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_obj}),
            }
        )
        page[NameObject("/Resources")] = resources

        escaped = _escape_pdf_text(
            f"{text_seed} page={idx + 1} " + ("lorem ipsum " * 64)
        )
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 11 Tf 72 720 Td ({escaped}) Tj ET".encode("latin-1", "replace"))
        page[NameObject("/Contents")] = writer._add_object(stream)  # type: ignore[attr-defined]

    with path.open("wb") as handle:
        writer.write(handle)

    if min_size_bytes > 0:
        pad_pdf_to_size(path, min_size_bytes)
    return path


def pad_pdf_to_size(path: Path, target_bytes: int) -> Path:
    current = path.stat().st_size
    if current >= target_bytes:
        return path

    remaining = target_bytes - current
    chunk = b"0" * MB
    with path.open("ab") as handle:
        handle.write(b"\n% lit-lake fixture padding\n")
        while remaining > 0:
            size = min(remaining, len(chunk))
            handle.write(chunk[:size])
            remaining -= size
    return path


def create_malformed_pdf(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not-a-pdf")
    return path


def build_stall_regression_fixture_set(
    root: Path,
    *,
    large_size_bytes: int = 55 * MB,
) -> list[PdfFixtureSpec]:
    root.mkdir(parents=True, exist_ok=True)

    specs: list[PdfFixtureSpec] = []

    for name, pages in [("small_1", 2), ("small_2", 3)]:
        path = create_valid_pdf(root / f"{name}.pdf", pages=pages, text_seed=name)
        specs.append(PdfFixtureSpec(name=name, path=path, expected_outcome="success", min_size_bytes=0))

    for name, pages in [("medium_1", 20), ("medium_2", 28)]:
        path = create_valid_pdf(root / f"{name}.pdf", pages=pages, text_seed=name, min_size_bytes=2 * MB)
        specs.append(
            PdfFixtureSpec(
                name=name,
                path=path,
                expected_outcome="success",
                min_size_bytes=2 * MB,
            )
        )

    for name in ("large_1", "large_2"):
        path = create_valid_pdf(
            root / f"{name}.pdf",
            pages=6,
            text_seed=name,
            min_size_bytes=large_size_bytes,
        )
        specs.append(
            PdfFixtureSpec(
                name=name,
                path=path,
                expected_outcome="success",
                min_size_bytes=large_size_bytes,
            )
        )

    malformed_path = create_malformed_pdf(root / "malformed_1.pdf")
    specs.append(
        PdfFixtureSpec(
            name="malformed_1",
            path=malformed_path,
            expected_outcome="failure",
            min_size_bytes=0,
        )
    )
    return specs


def should_run_true_large_fixture() -> bool:
    return os.getenv("LIT_LAKE_RUN_LARGE_FIXTURES", "").lower() in {"1", "true", "yes", "on"}

