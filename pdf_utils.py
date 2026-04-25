import re
import html
import tempfile
import os
from pathlib import Path
from typing import List, Dict, Optional

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling_core.transforms.chunker.hierarchical_chunker import HierarchicalChunker

from semantic_chunking import (
    normalize_headings_semantic,
    merge_semantically_similar_chunks,
    debug_heading_scores,       # dùng trong CLI --test-heading
    save_embedding_cache,       # dùng trong CLI --save-cache
)

try:
    from bedrock_llm import extract_skills_from_experience
    SKILL_EXTRACTION_AVAILABLE = True
except ImportError:
    SKILL_EXTRACTION_AVAILABLE = False


# =============================================================================
# 1. CONVERTER FACTORY
# =============================================================================

def build_converter(*, ocr: bool = False, table_structure: bool = True) -> DocumentConverter:
    opts = PdfPipelineOptions(do_ocr=ocr, do_table_structure=table_structure)
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )


# =============================================================================
# 2. TEXT CLEANING
# =============================================================================

def fix_spaced_letters(text: str) -> str:
    """S T R E N G T H -> STRENGTH (lỗi encoding PDF phổ biến)."""
    return re.sub(
        r'\b(?:[A-Za-z]\s){3,}[A-Za-z]\b',
        lambda m: m.group(0).replace(" ", ""),
        text,
    )


def clean_text(text: str) -> str:
    """Unescape HTML, fix spaced letters, collapse whitespace."""
    text = html.unescape(text)
    text = fix_spaced_letters(text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# =============================================================================
# 3. PDF -> DOCLING DOCUMENT
# =============================================================================

def _load_document(pdf_source: str | Path | bytes, converter: DocumentConverter):
    """Convert PDF -> Docling Document. Nhận path hoặc bytes."""
    if isinstance(pdf_source, bytes):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf_source)
            tmp_path = tmp.name
        try:
            return converter.convert(tmp_path).document
        finally:
            os.unlink(tmp_path)
    return converter.convert(str(pdf_source)).document


# =============================================================================
# 4. CANDIDATE NAME EXTRACTION
# =============================================================================

def _extract_candidate_name(doc) -> Optional[str]:
    """
    Trích xuất tên ứng viên từ phần đầu document.
    Tìm text element ngắn (1-6 từ), bắt đầu chữ hoa,
    không chứa keyword section hoặc thông tin liên lạc.
    """
    skip = {
        "experience", "education", "skills", "contact", "summary",
        "objective", "profile", "about", "resume", "cv", "certif",
        "language", "project", "activities", "awards", "strength",
        "information", "personal", "career", "work", "employment",
    }
    try:
        for item in doc.iterate_items():
            text = ""
            if hasattr(item, 'text'):
                text = item.text.strip()
            elif hasattr(item, 'export_to_text'):
                text = item.export_to_text().strip()

            if not text:
                continue

            text = fix_spaced_letters(text)
            lower = text.lower()

            if any(kw in lower for kw in skip):
                continue
            if re.search(r'[@+\d/\\:]', text):
                continue
            words = text.split()
            if not (1 <= len(words) <= 6):
                continue
            if not text[0].isupper():
                continue
            if sum(c.isalpha() for c in text) / len(text) < 0.7:
                continue

            return text
    except Exception:
        pass

    return None


# =============================================================================
# 5. SECTION LABEL TỪ HEADING PATH
# =============================================================================

def _heading_to_section(headings: List[str]) -> str:
    """
    Chuyển heading path từ Docling thành section label chuẩn.
    Dùng semantic matching (embedding-based) — không dùng keyword dict.
    """
    return normalize_headings_semantic(headings)


# =============================================================================
# 6. MAIN PIPELINE
# =============================================================================

def cv_pdf_to_semantic_chunks(
    pdf_source: str | Path | bytes,
    *,
    merge_peers: bool = True,
    converter: Optional[DocumentConverter] = None,
    ocr: bool = False,
    semantic_merge_threshold: float = 0.7,
) -> List[Dict]:
    """
    Pipeline chính: PDF -> List[Dict] semantic chunks.

    Luồng:
      1. Docling convert PDF -> Document
      2. HierarchicalChunker tạo chunks theo cấu trúc heading
      3. _heading_to_section gán section label qua semantic matching
      4. (Tuỳ chọn) extract implicit skills từ experience chunks
      5. merge_semantically_similar_chunks gộp chunk nhỏ theo ngữ nghĩa

    Returns:
        [{"section", "text", "chunk_index", "headings", "candidate_name"?}, ...]
    """
    if converter is None:
        converter = build_converter(ocr=ocr)

    # Bước 1: Convert PDF
    doc = _load_document(pdf_source, converter)

    # Bước 2: Extract tên ứng viên
    candidate_name = _extract_candidate_name(doc)

    # Bước 3: Chunk theo cấu trúc document
    chunker = HierarchicalChunker(merge_peers=merge_peers)
    raw_chunks = list(chunker.chunk(doc))

    # Bước 4: Build kết quả
    results: List[Dict] = []
    is_first = True

    for raw in raw_chunks:
        # Lấy text
        chunk_text = ""
        if hasattr(raw, 'text'):
            chunk_text = raw.text.strip()
        if not chunk_text and hasattr(raw, 'export_to_text'):
            chunk_text = raw.export_to_text().strip()
        if not chunk_text:
            continue

        chunk_text = clean_text(chunk_text)
        if len(chunk_text) < 20:
            continue

        # Lấy và normalize heading path
        headings: List[str] = []
        if hasattr(raw, 'meta') and raw.meta:
            raw_headings = getattr(raw.meta, 'headings', None) or []
            headings = [fix_spaced_letters(h).strip() for h in raw_headings]

        section = _heading_to_section(headings)

        entry: Dict = {
            "section":     section,
            "text":        chunk_text,
            "chunk_index": len(results),
            "headings":    headings,
        }
        if is_first and candidate_name:
            entry["candidate_name"] = candidate_name
            is_first = False

        results.append(entry)

        # Extract implicit skills từ experience chunk (gọi Bedrock)
        if section == "experience" and SKILL_EXTRACTION_AVAILABLE:
            try:
                skills = extract_skills_from_experience(chunk_text)
                if skills:
                    source_index = len(results) - 1  # index của experience chunk vừa append
                    results.append({
                        "section":      "skills",        # nhất quán với các section label khác
                        "text":         "\n".join(skills),
                        "chunk_index":  len(results),
                        "headings":     headings,
                        "source_chunk": source_index,    # tính đúng sau khi đã append
                    })
            except Exception as e:
                print(f"[WARNING] Failed to extract implicit skills: {e}")

    # Bước 5: Merge chunk nhỏ theo semantic similarity
    results = merge_semantically_similar_chunks(
        results,
        min_chars=80,
        similarity_threshold=semantic_merge_threshold,
    )

    # Re-index
    for i, r in enumerate(results):
        r["chunk_index"] = i

    return results


# =============================================================================
# 7. PDF -> MARKDOWN
# =============================================================================

def pdf_to_markdown(
    pdf_source: str | Path | bytes,
    *,
    include_toc: bool = False,
    ocr: bool = False,
    converter: Optional[DocumentConverter] = None,
) -> str:
    if converter is None:
        converter = build_converter(ocr=ocr)
    doc = _load_document(pdf_source, converter)
    md = clean_text(doc.export_to_markdown())

    if include_toc:
        headings = re.findall(r'^(#{1,3}) (.+)', md, re.MULTILINE)
        if headings:
            toc = ["## Table of Contents\n"]
            for hashes, title in headings:
                indent = "  " * (len(hashes) - 1)
                anchor = re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-')
                toc.append(f"{indent}- [{title}](#{anchor})")
            md = "\n".join(toc) + "\n\n---\n\n" + md

    return md


# =============================================================================
# 8. CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="pdf_utils - Docling CV processor")
    parser.add_argument("pdf",              help="Duong dan file PDF")
    parser.add_argument("-o", "--output",   help="Luu Markdown ra file")
    parser.add_argument("--chunk",          action="store_true", help="In semantic chunks")
    parser.add_argument("--ocr",            action="store_true", help="Bat OCR")
    parser.add_argument("--toc",            action="store_true", help="Them TOC vao markdown")
    parser.add_argument("--sim-threshold",  type=float, default=0.7,
                        help="Similarity threshold cho semantic merging (0-1)")
    parser.add_argument("--test-heading",   type=str,
                        help="Test semantic section detection cho mot heading cu the")
    parser.add_argument("--save-cache",     action="store_true",
                        help="Luu embedding cache sau khi process")
    args = parser.parse_args()

    conv = build_converter(ocr=args.ocr)

    if args.test_heading:
        debug_heading_scores(args.test_heading)
        exit(0)

    if args.chunk:
        chunks = cv_pdf_to_semantic_chunks(
            args.pdf,
            converter=conv,
            semantic_merge_threshold=args.sim_threshold,
        )
        print(f"\nTong chunks: {len(chunks)}\n")
        for c in chunks:
            name_info = f"  [Candidate: {c['candidate_name']}]" if "candidate_name" in c else ""
            hdg_info  = f"  [{' > '.join(c['headings'])}]" if c.get("headings") else ""
            print(f"{'─'*60}")
            print(f"Section: {c['section'].upper()}  |  #{c['chunk_index']}{name_info}{hdg_info}")
            print(c["text"][:400])
            if len(c["text"]) > 400:
                print("...")

        if args.save_cache:
            save_embedding_cache()
    else:
        md = pdf_to_markdown(args.pdf, include_toc=args.toc, converter=conv)
        if args.output:
            Path(args.output).write_text(md, encoding="utf-8")
            print(f"Da luu: {args.output}")
        else:
            print(md)