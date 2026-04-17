"""
cv_semantic_chunker.py
======================
Pipeline chunking CV/PDF hoàn chỉnh:

  1. Extract text từ PDF   (pdf_utils → PyMuPDF)
  2. Detect section headers (pdf_utils → group_by_section)
  3. Semantic chunking     (LangChain SemanticChunker + sentence-transformers)
  4. Đóng gói kết quả      (List[Dict] tương thích RAG / vector store)

Cài đặt:
    pip install pymupdf langchain-experimental sentence-transformers

Dùng nhanh:
    python cv_semantic_chunker.py <cv.pdf>
"""

# ── stdlib ────────────────────────────────────────────────────────────────────
import re
import html
import sys
from collections import Counter
from typing import List, Dict, Literal

# ── third-party ───────────────────────────────────────────────────────────────
import fitz  # PyMuPDF

from langchain_experimental.text_splitter import SemanticChunker
try:
    from langchain_huggingface import HuggingFaceEmbeddings          # langchain-huggingface >= 0.1
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings  # fallback cũ


# =============================================================================
# 1. TEXT CLEANING  (giữ nguyên từ pdf_utils.py)
# =============================================================================

def fix_spaced_letters(text: str) -> str:
    """S T R E N G T H  →  STRENGTH"""
    return re.sub(
        r'\b(?:[A-Za-z]\s){3,}[A-Za-z]\b',
        lambda m: m.group(0).replace(" ", ""),
        text,
    )


def clean_text(text: str) -> str:
    text = html.unescape(text)
    text = fix_spaced_letters(text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# =============================================================================
# 2. PDF EXTRACTION  (layout-aware, từ pdf_utils.py)
# =============================================================================

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []

    for page in doc:
        blocks = page.get_text("blocks")
        # sắp xếp top-down, left-right
        blocks = sorted(blocks, key=lambda b: (round(b[1] / 5) * 5, b[0]))

        page_lines = []
        for block in blocks:
            text = block[4].strip()
            if text:
                page_lines.append(text)

        pages.append("\n".join(page_lines))

    full_text = "\n\n".join(pages)

    if len(full_text.strip()) < 100:
        raise ValueError("PDF có thể là scan image – cần OCR fallback")

    return clean_text(full_text)


# =============================================================================
# 3. SECTION DETECTION  (từ pdf_utils.py)
# =============================================================================

SECTION_HEADERS = {
    "experience":     ["experience", "work experience", "employment history",
                       "professional experience", "work history", "kinh nghiệm"],
    "education":      ["education", "academic background", "academic history",
                       "học vấn", "trình độ học vấn"],
    "skills":         ["skills", "technical skills", "core competencies",
                       "key skills", "kỹ năng", "kỹ năng kỹ thuật"],
    "summary":        ["summary", "objective", "profile", "about me",
                       "career objective", "tóm tắt", "mục tiêu"],
    "certifications": ["certifications", "certificates", "chứng chỉ",
                       "licenses & certifications"],
    "activities":     ["activities", "extracurricular", "volunteer",
                       "hoạt động", "hoạt động ngoại khóa"],
    "projects":       ["projects", "personal projects", "dự án",
                       "key projects", "project experience"],
    "awards":         ["awards", "honors", "achievements",
                       "giải thưởng", "thành tích"],
}

SECTION_KEYWORDS = {
    "summary":        ["objective", "summary", "seeking", "motivated", "goal"],
    "skills":         ["python", "java", "react", "sql", "linux", "docker",
                       "git", "javascript", "html", "css", "mongodb", "fastapi"],
    "experience":     ["configured", "implemented", "managed", "responsible",
                       "developed", "deployed", "worked", "project", "built"],
    "education":      ["university", "college", "degree", "student", "gpa", "bachelor"],
    "certifications": ["certificate", "certification", "toeic", "ielts"],
    "activities":     ["activities", "volunteer", "club", "participate"],
}

SECTION_PRIORITY = ["experience", "education", "skills",
                    "summary", "certifications", "activities"]


def is_section_header(text: str) -> str | None:
    t = text.strip().lower()
    if len(t) > 50:
        return None
    for section, keywords in SECTION_HEADERS.items():
        if any(t == kw or t.startswith(kw) for kw in keywords):
            return section
    return None


def guess_section_by_pattern(text: str) -> str | None:
    t = text.lower()
    if re.search(r'gpa|university|college|student|\(20\d{2}', t):
        return "education"
    if re.search(r'configured|implemented|managed|deployed|set up|developed|built', t):
        return "experience"
    if ":" in text and len(text.split(",")) >= 3:
        tech_hints = ["python", "java", "sql", "linux", "react", "docker",
                      "git", "javascript", "html", "css", "mongodb", "fastapi",
                      "node", "typescript", "aws", "azure", "gcp"]
        if any(h in t for h in tech_hints):
            return "skills"
    return None


def guess_section_by_keywords(text: str) -> str | None:
    t = text.lower()
    scores: Dict[str, int] = {}
    for section, keywords in SECTION_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in t)
        if score > 0:
            scores[section] = score
    if not scores:
        return None
    max_score = max(scores.values())
    top = [s for s, sc in scores.items() if sc == max_score]
    if len(top) == 1:
        return top[0]
    for section in SECTION_PRIORITY:
        if section in top:
            return section
    return top[0]


def infer_section(text: str) -> str:
    return (guess_section_by_pattern(text)
            or guess_section_by_keywords(text)
            or "other")


def split_paragraphs(text: str) -> List[str]:
    return [p.strip() for p in text.split("\n") if len(p.strip()) > 2]


def group_by_section(paragraphs: List[str]) -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = {}
    current_section = "other"
    for p in paragraphs:
        header = is_section_header(p)
        if header:
            current_section = header
            continue
        grouped.setdefault(current_section, []).append(p)
    return grouped


# =============================================================================
# 4. SEMANTIC CHUNKER WRAPPER
# =============================================================================

def build_semantic_chunker(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    breakpoint_threshold_type: Literal[
        "percentile", "standard_deviation", "interquartile", "gradient"
    ] = "percentile",
    breakpoint_threshold_amount: float = 80.0,
    buffer_size: int = 1,
    min_chunk_size: int | None = 30,
) -> SemanticChunker:
    """
    Tạo SemanticChunker dùng HuggingFace embeddings (không cần API key).

    Tham số chính:
        model_name               - model HuggingFace để embed câu
        breakpoint_threshold_type- cách tính điểm ngắt:
                                   'percentile'         → ngắt khi similarity < X-th percentile (mặc định)
                                   'standard_deviation' → ngắt khi < mean - X*std
                                   'interquartile'      → dùng IQR
                                   'gradient'           → dựa trên gradient thay đổi
        breakpoint_threshold_amount - ngưỡng (ý nghĩa tuỳ loại trên)
        buffer_size              - số câu lân cận để tính similarity trung bình
        min_chunk_size           - số ký tự tối thiểu mỗi chunk (lọc chunk rác)
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return SemanticChunker(
        embeddings=embeddings,
        buffer_size=buffer_size,
        breakpoint_threshold_type=breakpoint_threshold_type,
        breakpoint_threshold_amount=breakpoint_threshold_amount,
        min_chunk_size=min_chunk_size,
    )


# =============================================================================
# 5. MAIN PIPELINE
# =============================================================================

def cv_pdf_to_semantic_chunks(
    pdf_bytes: bytes,
    chunker: SemanticChunker | None = None,
    *,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    breakpoint_threshold_type: Literal[
        "percentile", "standard_deviation", "interquartile", "gradient"
    ] = "percentile",
    breakpoint_threshold_amount: float = 80.0,
    buffer_size: int = 1,
    min_chunk_size: int = 30,
) -> List[Dict]:
    """
    Pipeline hoàn chỉnh: PDF bytes → List[{section, text, chunk_index}]

    Truyền `chunker` nếu đã khởi tạo sẵn (tái sử dụng model, tránh load lại).
    Nếu không truyền, hàm tự tạo chunker với các tham số còn lại.

    Trả về:
        [
          {
            "section":     "experience",   # section từ CV
            "text":        "...",          # nội dung chunk
            "chunk_index": 0,             # thứ tự chunk trong section
          },
          ...
        ]
    """
    # ── bước 1-2: trích xuất & phân section ──────────────────────────────────
    raw_text   = extract_text_from_pdf(pdf_bytes)
    paragraphs = split_paragraphs(raw_text)
    sections   = group_by_section(paragraphs)

    # ── bước 3: khởi tạo chunker nếu chưa có ────────────────────────────────
    if chunker is None:
        chunker = build_semantic_chunker(
            model_name=model_name,
            breakpoint_threshold_type=breakpoint_threshold_type,
            breakpoint_threshold_amount=breakpoint_threshold_amount,
            buffer_size=buffer_size,
            min_chunk_size=min_chunk_size,
        )

    # ── bước 4: semantic chunking theo từng section ──────────────────────────
    results: List[Dict] = []

    for section, paras in sections.items():
        # Ghép các đoạn trong section thành một khối văn bản
        section_text = "\n".join(paras).strip()
        if not section_text:
            continue

        # SemanticChunker.create_documents nhận list[str]
        docs = chunker.create_documents([section_text])

        for idx, doc in enumerate(docs):
            chunk_text = doc.page_content.strip()
            if len(chunk_text) < min_chunk_size:
                continue  # bỏ chunk quá ngắn / rác
            results.append(
                {
                    "section":     section,
                    "text":        chunk_text,
                    "chunk_index": idx,
                }
            )

    return results


# =============================================================================
# 6. CLI / TEST
# =============================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python cv_semantic_chunker.py <cv.pdf> [threshold_percentile]")
        sys.exit(1)

    pdf_path   = sys.argv[1]
    threshold  = float(sys.argv[2]) if len(sys.argv) > 2 else 80.0

    print(f"\n[1/3] Đọc file: {pdf_path}")
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    print("[2/3] Khởi tạo SemanticChunker (HuggingFace MiniLM)…")
    chunker = build_semantic_chunker(
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=threshold,
    )

    print("[3/3] Chunking…")
    chunks = cv_pdf_to_semantic_chunks(pdf_bytes, chunker=chunker)

    # ── thống kê ─────────────────────────────────────────────────────────────
    print(f"\n✅ Tổng số chunks: {len(chunks)}")

    section_counts = Counter(ch["section"] for ch in chunks)
    print("\nPhân bố theo section:")
    for section, count in sorted(section_counts.items()):
        print(f"  {section:<20} {count} chunks")

    print("\n--- Chi tiết 10 chunks đầu ---")
    for i, ch in enumerate(chunks[:10]):
        print(f"\n{'─'*60}")
        print(f"Chunk {i:02d} | section={ch['section']} | index_in_section={ch['chunk_index']}")
        print(ch["text"][:400])