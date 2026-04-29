import re # Chức năng regex cho text cleaning và experience parsing
import html # Unescape HTML entities trong text (nếu có)
import tempfile # Tạo file tạm cho converter khi nhận bytes
import os # Xử lý file tạm và đường dẫn
from pathlib import Path # Xử lý đường dẫn file một cách linh hoạt
from typing import List, Dict, Optional #Chức năng làm type hinting cho code rõ ràng hơn

from docling.document_converter import DocumentConverter, PdfFormatOption # Docling converter để chuyển PDF -> Document
from docling.datamodel.pipeline_options import PdfPipelineOptions # Tùy chọn cho PDF converter (e.g. bật OCR)
from docling.datamodel.base_models import InputFormat # Định nghĩa định dạng đầu vào cho converter factory
from docling.chunking import HybridChunker # Chunker chính để chia nhỏ document thành semantic chunks

#Functions để xử lý PDF -> semantic chunks, bao gồm text cleaning, heading extraction, và merge small chunks.

def build_converter(*, ocr: bool = False, table_structure: bool = True) -> DocumentConverter:
    opts = PdfPipelineOptions(do_ocr=ocr, do_table_structure=table_structure)
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )

#Functions để clean text, fix lỗi spaced letters (S T R E N G T H -> STRENGTH), và merge các chunk nhỏ vào chunk trước nếu cùng heading path.
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


def _load_document(pdf_source: str | Path | bytes, converter: DocumentConverter): #Chức năng nội bộ để load PDF vào Document của Docling, hỗ trợ cả bytes và file path.
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



def _heading_to_section(headings: List[str]) -> str:
    if not headings:
        return "unknown"
    return headings[-1]


def _merge_small_chunks(chunks: List[Dict], min_chars: int = 80) -> List[Dict]:
    if not chunks:
        return chunks

    merged = [chunks[0].copy()]
    for curr in chunks[1:]:
        prev = merged[-1]
        same_heading = curr.get("headings") == prev.get("headings")
        too_small = len(curr.get("text", "")) < min_chars

        if too_small and same_heading:
            prev["text"] = prev["text"] + "\n" + curr["text"]
        else:
            merged.append(curr.copy())

    return merged



def cv_pdf_to_chunks(
    pdf_source: str | Path | bytes,
    *,
    max_tokens: int = 512,
    min_chars: int = 80,
    converter: Optional[DocumentConverter] = None,
    ocr: bool = False,
) -> List[Dict]:
    if converter is None:
        converter = build_converter(ocr=ocr)

    doc = _load_document(pdf_source, converter)

    chunker = HybridChunker(max_tokens=max_tokens, merge_peers=True)
    raw_chunks = list(chunker.chunk(doc))

    results: List[Dict] = []

    for raw in raw_chunks:
        # Lấy text từ chunk
        chunk_text = ""
        if hasattr(raw, "text"):
            chunk_text = raw.text.strip()
        if not chunk_text and hasattr(raw, "export_to_text"):
            chunk_text = raw.export_to_text().strip()
        if not chunk_text:
            continue

        chunk_text = clean_text(chunk_text)
        if len(chunk_text) < 20:
            continue

        # Lấy heading path (metadata thuần, không classify)
        headings: List[str] = []
        if hasattr(raw, "meta") and raw.meta:
            raw_headings = getattr(raw.meta, "headings", None) or []
            headings = [fix_spaced_letters(h).strip() for h in raw_headings]

        results.append({
            "chunk_index": len(results),
            "section":     _heading_to_section(headings),
            "text":        chunk_text,
            "headings":    headings,
        })

    results = _merge_small_chunks(results, min_chars=min_chars)

    # Re-index sau merge
    for i, r in enumerate(results):
        r["chunk_index"] = i

    return results


def cv_pdf_to_semantic_chunks(
    pdf_source: str | Path | bytes,
    *,
    max_tokens: int = 512,
    min_chars: int = 80,
    converter: Optional[DocumentConverter] = None,
    ocr: bool = False,
) -> List[Dict]:
    return cv_pdf_to_chunks(
        pdf_source,
        max_tokens=max_tokens,
        min_chars=min_chars,
        converter=converter,
        ocr=ocr,
    )


# =============================================================================
# 7. PDF -> MARKDOWN (tiện ích phụ)
# =============================================================================

def pdf_to_markdown(
    pdf_source: str | Path | bytes,
    *,
    include_toc: bool = False,
    ocr: bool = False,
    converter: Optional[DocumentConverter] = None,
) -> str:
    """PDF -> Markdown sử dụng Docling export. Giữ đúng heading, bảng, list."""
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
    import json

    parser = argparse.ArgumentParser(description="pdf_utils - Docling CV semantic chunker")
    parser.add_argument("pdf",                        help="Đường dẫn file PDF")
    parser.add_argument("-o", "--output",             help="Lưu output ra file")
    parser.add_argument("--chunk",   action="store_true", help="In semantic chunks (default: markdown)")
    parser.add_argument("--json",    action="store_true", help="Output chunks dạng JSON")
    parser.add_argument("--ocr",     action="store_true", help="Bật OCR")
    parser.add_argument("--toc",     action="store_true", help="Thêm TOC vào markdown")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens mỗi chunk (default=512)")
    parser.add_argument("--min-chars",  type=int, default=80,  help="Min chars để merge chunk (default=80)")
    args = parser.parse_args()

    conv = build_converter(ocr=args.ocr)

    if args.chunk:
        chunks = cv_pdf_to_chunks(
            args.pdf,
            max_tokens=args.max_tokens,
            min_chars=args.min_chars,
            converter=conv,
        )

        if args.json:
            output = json.dumps(chunks, ensure_ascii=False, indent=2)
            if args.output:
                Path(args.output).write_text(output, encoding="utf-8")
                print(f"Đã lưu {len(chunks)} chunks -> {args.output}")
            else:
                print(output)
        else:
            print(f"\nTổng chunks: {len(chunks)}\n")
            for c in chunks:
                hdg = " > ".join(c["headings"]) if c["headings"] else "(no heading)"
                print(f"{'─' * 60}")
                print(f"#{c['chunk_index']}  section={c['section']}  path=[{hdg}]")
                print(c["text"][:400])
                if len(c["text"]) > 400:
                    print("...")
    else:
        md = pdf_to_markdown(args.pdf, include_toc=args.toc, converter=conv)
        if args.output:
            Path(args.output).write_text(md, encoding="utf-8")
            print(f"Đã lưu: {args.output}")
        else:
            print(md)