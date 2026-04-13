import fitz  # PyMuPDF
import re
import html
from typing import List, Dict


# =====================================================
# TEXT CLEANING
# =====================================================

def fix_spaced_letters(text: str) -> str:
    """
    Fix chữ bị tách từng ký tự:
    S T R E N G T H -> STRENGTH
    m a c h i n e -> machine
    """
    return re.sub(
        r'\b(?:[A-Za-z]\s){3,}[A-Za-z]\b',
        lambda m: m.group(0).replace(" ", ""),
        text
    )


def clean_text(text: str) -> str:
    text = html.unescape(text)
    text = fix_spaced_letters(text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# =====================================================
# PDF EXTRACTION (LAYOUT-AWARE)
# =====================================================

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []

    for page in doc:
        blocks = page.get_text("blocks")
        blocks = sorted(blocks, key=lambda b: (round(b[1] / 5) * 5, b[0]))  # top-down, left-right

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


# =====================================================
# PARAGRAPH SPLIT
# FIX 1: Hạ threshold từ > 15 xuống > 2
# Tránh mất: tên, SĐT, GPA, skill ngắn
# =====================================================

def split_paragraphs(text: str) -> List[str]:
    paragraphs = [
        p.strip()
        for p in text.split("\n")
        if len(p.strip()) > 2   # chỉ bỏ dòng trắng thực sự
    ]
    return paragraphs


# =====================================================
# SECTION HEADERS (dùng cho FIX 2)
# Chỉ đổi section khi gặp HEADER thực sự, không phải keyword ngẫu nhiên
# =====================================================

SECTION_HEADERS = {
    "experience": [
        "experience", "work experience", "employment history",
        "professional experience", "work history", "kinh nghiệm"
    ],
    "education": [
        "education", "academic background", "academic history",
        "học vấn", "trình độ học vấn"
    ],
    "skills": [
        "skills", "technical skills", "core competencies",
        "key skills", "kỹ năng", "kỹ năng kỹ thuật"
    ],
    "summary": [
        "summary", "objective", "profile", "about me",
        "career objective", "tóm tắt", "mục tiêu"
    ],
    "certifications": [
        "certifications", "certificates", "chứng chỉ",
        "licenses & certifications"
    ],
    "activities": [
        "activities", "extracurricular", "volunteer",
        "hoạt động", "hoạt động ngoại khóa"
    ],
    "projects": [
        "projects", "personal projects", "dự án",
        "key projects", "project experience"
    ],
    "awards": [
        "awards", "honors", "achievements",
        "giải thưởng", "thành tích"
    ]
}


def is_section_header(text: str) -> str | None:
    """
    Chỉ nhận diện header nếu dòng đó NGẮN (< 50 ký tự)
    và khớp chính xác với từ khóa section.
    Tránh nhầm dòng nội dung thành header.
    """
    t = text.strip().lower()

    # Header thường rất ngắn
    if len(t) > 50:
        return None

    for section, keywords in SECTION_HEADERS.items():
        if any(t == kw or t.startswith(kw) for kw in keywords):
            return section

    return None


# =====================================================
# SECTION INFERENCE (fallback khi không có header rõ ràng)
# =====================================================

SECTION_KEYWORDS = {
    "summary": [
        "objective", "summary", "seeking", "motivated", "goal"
    ],
    "skills": [
        "python", "java", "react", "sql", "linux", "docker",
        "git", "javascript", "html", "css", "mongodb", "fastapi"
    ],
    "experience": [
        "configured", "implemented", "managed", "responsible",
        "developed", "deployed", "worked", "project", "built"
    ],
    "education": [
        "university", "college", "degree", "student", "gpa", "bachelor"
    ],
    "certifications": [
        "certificate", "certification", "toeic", "ielts"
    ],
    "activities": [
        "activities", "volunteer", "club", "participate"
    ]
}


def guess_section_by_pattern(text: str) -> str | None:
    t = text.lower()

    # Education patterns
    if re.search(r'gpa|university|college|student|\(20\d{2}', t):
        return "education"

    # Experience patterns
    if re.search(r'configured|implemented|managed|deployed|set up|developed|built', t):
        return "experience"

    # FIX 4: Điều kiện skills chặt hơn — phải có từ kỹ thuật cụ thể
    # Tránh nhầm "Hanoi, Vietnam" hay "Jan 2022, FPT Corp" thành skills
    if ":" in text and len(text.split(",")) >= 3:
        tech_hints = [
            "python", "java", "sql", "linux", "react", "docker",
            "git", "javascript", "html", "css", "mongodb", "fastapi",
            "node", "typescript", "aws", "azure", "gcp"
        ]
        if any(h in t for h in tech_hints):
            return "skills"

    return None


def guess_section_by_keywords(text: str) -> str | None:
    t = text.lower()
    scores = {}

    for section, keywords in SECTION_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in t)
        if score > 0:
            scores[section] = score

    if not scores:
        return None

    return max(scores, key=scores.get)
SECTION_PRIORITY = [
    "experience", "education", "skills",
    "summary", "certifications", "activities"
]

def guess_section_by_keywords(text: str) -> str | None:
    t = text.lower()
    scores = {}
    for section, keywords in SECTION_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in t)
        if score > 0:
            scores[section] = score

    if not scores:
        return None

    max_score = max(scores.values())
    # Lấy tất cả section có cùng điểm cao nhất
    top_sections = [s for s, sc in scores.items() if sc == max_score]

    if len(top_sections) == 1:
        return top_sections[0]

    # Tie-breaking theo priority
    for section in SECTION_PRIORITY:
        if section in top_sections:
            return section

    return top_sections[0]


def infer_section(text: str) -> str:
    return (
        guess_section_by_pattern(text)
        or guess_section_by_keywords(text)
        or "other"
    )


# =====================================================
# GROUP PARAGRAPHS BY SECTION
# FIX 2: Chỉ đổi section khi gặp header thực sự
# Tránh "drift" section do keyword ngẫu nhiên trong nội dung
# =====================================================

def group_by_section(paragraphs: List[str]) -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = {}
    current_section = "other"

    for p in paragraphs:
        # Ưu tiên kiểm tra header trước
        header = is_section_header(p)

        if header:
            # Đây là dòng header → đổi section, KHÔNG thêm vào content
            current_section = header
            continue

        # Không phải header → thêm vào section hiện tại
        grouped.setdefault(current_section, []).append(p)

    return grouped


# =====================================================
# SMART CHUNKING
# FIX 3: Xử lý đoạn văn dài hơn chunk_size, không bị mất
# =====================================================

def smart_overlap(prev: str, max_chars: int) -> str:
    words = prev.split()
    result = ""

    for w in reversed(words):
        if len(result) + len(w) + 1 <= max_chars:
            result = w + " " + result
        else:
            break

    return result.strip()


def chunk_section(
    paragraphs: List[str],
    chunk_size: int = 500,
    overlap: int = 80
) -> List[str]:
    chunks = []
    current = ""

    for p in paragraphs:
        # FIX 3: Đoạn đơn lẻ dài hơn chunk_size → tách thành nhiều chunk nhỏ
        if len(p) > chunk_size:
            if current.strip():
                chunks.append(current.strip())
                current = ""

            # Tách theo từ, không cắt giữa chừng
            words = p.split()
            buf = ""
            for w in words:
                if len(buf) + len(w) + 1 <= chunk_size:
                    buf += (" " if buf else "") + w
                else:
                    if buf:
                        chunks.append(buf.strip())
                    buf = w
            if buf.strip():
                chunks.append(buf.strip())
            continue

        if len(current) + len(p) + 1 <= chunk_size:
            current += ("\n" if current else "") + p
        else:
            if current.strip():
                chunks.append(current.strip())
            current = p

    if current.strip():
        chunks.append(current.strip())

    # Thêm overlap giữa các chunk
    final_chunks = []
    for i, c in enumerate(chunks):
        if i == 0:
            final_chunks.append(c)
        else:
            ov = smart_overlap(chunks[i - 1], overlap)
            final_chunks.append((ov + "\n" + c) if ov else c)

    return final_chunks


# =====================================================
# MAIN PIPELINE
# =====================================================

def cv_pdf_to_section_chunks(pdf_bytes: bytes) -> List[Dict]:
    text = extract_text_from_pdf(pdf_bytes)
    paragraphs = split_paragraphs(text)
    sections = group_by_section(paragraphs)

    results = []

    for section, paras in sections.items():
        chunks = chunk_section(paras)

        for c in chunks:
            if c.strip():
                results.append({
                    "section": section,
                    "text": c
                })

    return results


# =====================================================
# TEST
# =====================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python cv_parser.py <cv.pdf>")
        exit(1)

    with open(sys.argv[1], "rb") as f:
        pdf_bytes = f.read()

    chunks = cv_pdf_to_section_chunks(pdf_bytes)

    print(f"\nTong so chunks: {len(chunks)}")

    # Thống kê theo section
    from collections import Counter
    section_counts = Counter(ch["section"] for ch in chunks)
    print("\nPhan bo theo section:")
    for section, count in sorted(section_counts.items()):
        print(f"  {section:<20} {count} chunks")

    print("\n--- Chi tiet 10 chunks dau ---")
    for i, ch in enumerate(chunks[:10]):
        print(f"\n--- Chunk {i} [{ch['section']}] ---")
        print(ch["text"][:400])