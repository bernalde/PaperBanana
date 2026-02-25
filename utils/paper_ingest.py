# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Paper ingestion helpers for converting PDF/DOCX to markdown-like text.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple


def supported_extensions() -> tuple[str, str]:
    """Return supported upload extensions."""
    return (".pdf", ".docx")


def _normalize_markdown(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_with_docling(file_path: Path) -> Tuple[str, List[str]]:
    """
    Try Docling conversion first. Returns (markdown, warnings).
    Raises RuntimeError when not available or conversion fails.
    """
    warnings: List[str] = []
    try:
        from docling.document_converter import DocumentConverter
    except Exception as exc:
        raise RuntimeError(f"Docling is unavailable: {exc}") from exc

    try:
        converter = DocumentConverter()
        result = converter.convert(str(file_path))
    except Exception as exc:
        raise RuntimeError(f"Docling conversion failed: {exc}") from exc

    doc_obj = getattr(result, "document", result)
    markdown = ""
    for method_name in ("export_to_markdown", "to_markdown", "as_markdown"):
        method = getattr(doc_obj, method_name, None)
        if callable(method):
            try:
                markdown = method()
                break
            except Exception:
                continue

    if not markdown:
        markdown = str(doc_obj)
        warnings.append(
            "Docling conversion did not expose a markdown export method; "
            "falling back to string representation."
        )

    return _normalize_markdown(markdown), warnings


def _extract_pdf_fallback(file_path: Path) -> str:
    try:
        from pypdf import PdfReader
    except Exception as exc:
        raise RuntimeError(
            "pypdf is required for PDF fallback extraction. "
            "Install with `uv pip install pypdf`."
        ) from exc

    reader = PdfReader(str(file_path))
    page_chunks: List[str] = []
    for idx, page in enumerate(reader.pages):
        page_text = (page.extract_text() or "").strip()
        if not page_text:
            continue
        page_chunks.append(f"## Page {idx + 1}\n\n{page_text}")

    if not page_chunks:
        raise RuntimeError(
            "No text extracted from PDF. The file may be scanned/image-only and OCR is not enabled."
        )

    return _normalize_markdown("\n\n".join(page_chunks))


def _extract_docx_fallback(file_path: Path) -> str:
    try:
        import docx
    except Exception as exc:
        raise RuntimeError(
            "python-docx is required for DOCX fallback extraction. "
            "Install with `uv pip install python-docx`."
        ) from exc

    document = docx.Document(str(file_path))
    chunks: List[str] = []

    for para in document.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        style_name = (para.style.name or "").lower() if para.style else ""
        if "heading" in style_name:
            level_match = re.search(r"(\d+)", style_name)
            level = int(level_match.group(1)) if level_match else 2
            level = max(1, min(6, level))
            chunks.append(f"{'#' * level} {text}")
        else:
            chunks.append(text)

    if not chunks:
        raise RuntimeError("No text extracted from DOCX.")

    return _normalize_markdown("\n\n".join(chunks))


def convert_document_to_markdown(file_path: Path) -> Dict[str, object]:
    """
    Convert a paper file to markdown-like text with docling-first strategy.
    """
    suffix = file_path.suffix.lower()
    if suffix not in supported_extensions():
        raise ValueError(f"Unsupported file type: {suffix}. Supported: {supported_extensions()}")

    warnings: List[str] = []

    try:
        markdown, docling_warnings = _extract_with_docling(file_path)
        warnings.extend(docling_warnings)
        engine = "docling"
        return {
            "markdown": markdown,
            "engine": engine,
            "warnings": warnings,
        }
    except Exception as exc:
        warnings.append(str(exc))

    if suffix == ".pdf":
        markdown = _extract_pdf_fallback(file_path)
        engine = "pypdf-fallback"
    else:
        markdown = _extract_docx_fallback(file_path)
        engine = "python-docx-fallback"

    return {
        "markdown": markdown,
        "engine": engine,
        "warnings": warnings,
    }

