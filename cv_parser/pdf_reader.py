from __future__ import annotations
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional
if TYPE_CHECKING:
    # pyrefly: ignore [missing-import]
    from docling.document_converter import DocumentConverter
from .models import simple_clean

def build_converter(*, ocr: bool = False, table_structure: bool = True) -> "DocumentConverter":
    # pyrefly: ignore [missing-import]
    from docling.datamodel.base_models import InputFormat
    # pyrefly: ignore [missing-import]
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    # pyrefly: ignore [missing-import]
    from docling.document_converter import DocumentConverter, PdfFormatOption

    opts = PdfPipelineOptions(do_ocr=ocr, do_table_structure=table_structure)
    return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})


def _load_document(pdf_source: str | Path | bytes, converter: "DocumentConverter") -> Any:
    if isinstance(pdf_source, bytes):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf_source)
            tmp_path = tmp.name
        try:
            return converter.convert(tmp_path).document
        finally:
            os.unlink(tmp_path)
    return converter.convert(str(pdf_source)).document


def _pypdf_to_markdown(pdf_source: str | Path | bytes) -> str:
    from io import BytesIO
    # pyrefly: ignore [missing-import]
    from PyPDF2 import PdfReader

    reader = PdfReader(BytesIO(pdf_source) if isinstance(pdf_source, bytes) else str(pdf_source))
    pages = [(p.extract_text() or "").strip() for p in reader.pages]
    markdown = simple_clean("\n\n---\n\n".join(p for p in pages if p))
    if not markdown:
        raise ValueError("PDF text fallback produced no text")
    return markdown


def _ocr_to_markdown(pdf_source: str | Path | bytes, *, zoom: float = 2.0) -> str:
    # pyrefly: ignore [missing-import]
    import fitz
    import numpy as np
    # pyrefly: ignore [missing-import]
    from rapidocr import RapidOCR

    doc = (
        fitz.open(stream=pdf_source, filetype="pdf")
        if isinstance(pdf_source, bytes)
        else fitz.open(str(pdf_source))
    )
    engine = RapidOCR()
    pages: List[str] = []
    try:
        for page in doc:
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            lines = []
            for text in getattr(engine(image), "txts", None) or []:
                line = text.strip()
                if line:
                    lines.append(line)
            if lines:
                pages.append("\n".join(lines))
    finally:
        doc.close()

    markdown = simple_clean("\n\n---\n\n".join(pages))
    if not markdown:
        raise ValueError("OCR fallback produced no text")
    return markdown


def convert_cv_pdf(
    pdf_source: str | Path | bytes,
    *,
    ocr: bool = False,
    converter: Optional["DocumentConverter"] = None,
) -> tuple[Any | None, str]:
    """Convert a CV PDF to (docling_doc | None, markdown_text).

    Conversion pipeline: Docling (full) → Docling (no table) → PyPDF2 → OCR.
    """
    supplied = converter is not None
    converter = converter or build_converter(ocr=ocr)
    errors: List[str] = []

    try:
        doc = _load_document(pdf_source, converter)
        md = simple_clean(doc.export_to_markdown())
        if md:
            return doc, md
        errors.append("Docling: empty output")
    except Exception as e:
        errors.append(f"Docling: {e}")

    if not supplied:
        try:
            doc = _load_document(pdf_source, build_converter(ocr=ocr, table_structure=False))
            md = simple_clean(doc.export_to_markdown())
            if md:
                return doc, md
            errors.append("Docling light: empty output")
        except Exception as e:
            errors.append(f"Docling light: {e}")

    try:
        return None, _pypdf_to_markdown(pdf_source)
    except Exception as e:
        errors.append(f"PyPDF2: {e}")

    try:
        return None, _ocr_to_markdown(pdf_source)
    except Exception as e:
        errors.append(f"OCR: {e}")
        raise RuntimeError(
            "CV PDF conversion produced no readable text. " + " | ".join(errors)
        ) from e
