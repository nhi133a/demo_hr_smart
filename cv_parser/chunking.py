from __future__ import annotations
import re
from typing import Any, Dict, List, Optional
from .models import simple_clean, _clean_value, _as_list, _dedupe_strings, NormalizedText

SECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("summary",        re.compile(r"\b(summary|objective|career objective|profile|about me)\b", re.I)),
    ("skills",         re.compile(r"\b(skills?|technical skills?|tools?|technologies|tech stack|competenc(?:e|ies)|expertise|k[yỹ]\s*n[aă]ng|c[oô]ng\s*c[uụ])\b", re.I)),
    ("experience",     re.compile(r"\b(experience|work experience|professional experience|employment|work history|internship|intern|kinh\s*nghi[eệ]m|kinh\s*nghiem)\b", re.I)),
    ("projects",       re.compile(r"\b(projects?|portfolio|case stud(?:y|ies)|d[uự]\s*[aá]n)\b", re.I)),
    ("education",      re.compile(r"\b(education|academic|university|college|school|degree|gpa|h[oọ]c\s*v[aấ]n)\b", re.I)),
    ("certifications", re.compile(r"\b(certifications?|certificates?|awards?|training|licenses?|ch[uứ]ng\s*ch[iỉ])\b", re.I)),
    ("languages",      re.compile(r"\b(languages?|english|ielts|toeic|ng[oô]n\s*ng[uữ])\b", re.I)),
]

def normalize_section_label(label: str) -> str:
    label = simple_clean(label).lower()
    for section, pattern in SECTION_PATTERNS:
        if pattern.search(label):
            return section
    return label or "unknown"


def _is_plain_section_heading(line: str) -> bool:
    line = simple_clean(re.sub(r"^#+\s*", "", line))
    if not line or len(line) > 80:
        return False
    if line.startswith(("-", "*", "•")) or "@" in line or ":" in line or "|" in line:
        return False
    if re.search(r"[.!?,;:]$", line):
        return False
    # Must match a known section keyword
    label = line.lower()
    for _, pattern in SECTION_PATTERNS:
        if pattern.search(label):
            break
    else:
        return False
    words = re.findall(r"[A-Za-zÀ-ỹ0-9]+", line)
    return len(words) <= 8

def _enrich_chunks(chunks: List[Dict[str, Any]], candidate_name: str = "") -> List[Dict[str, Any]]:
    enriched: List[Dict[str, Any]] = []
    for idx, chunk in enumerate(chunks):
        item = dict(chunk)
        item["text"]     = simple_clean(item.get("text", ""))
        item["headings"] = [heading for heading in (simple_clean(h) for h in _as_list(item.get("headings"))) if heading]
        raw_heading = simple_clean(item.get("raw_heading") or (item["headings"][-1] if item["headings"] else item.get("section", "")))
        section = normalize_section_label(raw_heading)
        item.update({
            "chunk_index":   idx,
            "chunk_id":      item.get("chunk_id") or f"chunk_{idx}",
            "raw_heading":   raw_heading or "unknown",
            "section":       section,
            "embedding_text": item["text"],
        })
        if candidate_name:
            item["candidate_name"] = candidate_name
        enriched.append(item)
    return enriched


def _merge_small_chunks(chunks: List[Dict[str, Any]], min_chars: int, max_chars: int) -> List[Dict[str, Any]]:
    if not chunks:
        return []
    merged = [chunks[0].copy()]
    for cur in chunks[1:]:
        prev = merged[-1]
        if (
            cur.get("section") == prev.get("section")
            and len(cur.get("text", "")) < min_chars
            and len(prev.get("text", "")) + len(cur.get("text", "")) <= max_chars
        ):
            prev["text"]     = simple_clean(prev.get("text", "") + "\n" + cur.get("text", ""))
            prev["headings"] = _dedupe_strings(_as_list(prev.get("headings")) + _as_list(cur.get("headings")), limit=8)
        else:
            merged.append(cur.copy())
    return merged


def cv_markdown_to_chunks(
    markdown_text: str | NormalizedText,
    schema: Optional[Dict[str, Any]] = None,
    *,
    min_chars: int = 80,
    max_chars: int = 1200,
) -> List[Dict[str, Any]]:
    """Split markdown CV into section chunks for vector search."""
    markdown_text = _clean_value(markdown_text)
    candidate_name = (schema or {}).get("candidate", {}).get("name", "") if isinstance(schema, dict) else ""

    chunks: List[Dict[str, Any]] = []
    current_heading = "profile"
    current_lines: List[str] = []

    def flush() -> None:
        nonlocal current_lines
        text = simple_clean("\n".join(current_lines))
        if text:
            chunks.append({
                "chunk_index": len(chunks),
                "section":     current_heading,
                "raw_heading": current_heading,
                "text":        text,
                "headings":    [current_heading],
            })
        current_lines.clear()

    for raw_line in markdown_text.splitlines():
        md_heading = re.match(r"^\s{0,3}#{1,4}\s+(.+?)\s*$", raw_line)
        if md_heading:
            flush()
            current_heading = simple_clean(md_heading.group(1)) or "section"
        elif schema is not None and _is_plain_section_heading(raw_line):
            flush()
            current_heading = simple_clean(raw_line) or "section"
        else:
            current_lines.append(raw_line.rstrip())
    flush()

    if not chunks and markdown_text:
        chunks = [{"chunk_index": 0, "section": "profile", "raw_heading": "profile",
                   "text": markdown_text, "headings": ["profile"]}]

    chunks = _merge_small_chunks(chunks, min_chars, max_chars)
    return _enrich_chunks(chunks, candidate_name)


def cv_document_to_chunks(
    doc: Any,
    schema: Optional[Dict[str, Any]] = None,
    *,
    markdown_text: str | NormalizedText | None = None,
    max_tokens: int = 512,
    min_chars: int = 80,
    max_chars: int = 1200,
) -> List[Dict[str, Any]]:
    """Chunk a Docling document for vector search."""
    # pyrefly: ignore [missing-import]
    from docling.chunking import HybridChunker

    markdown = _clean_value(markdown_text) if markdown_text is not None else simple_clean(doc.export_to_markdown())
    chunker  = HybridChunker(max_tokens=max_tokens, merge_peers=True)
    chunks: List[Dict[str, Any]] = []

    for raw in chunker.chunk(doc):
        text = simple_clean(getattr(raw, "text", "") or "")
        if not text and hasattr(raw, "export_to_text"):
            text = simple_clean(raw.export_to_text())
        if len(text) < 20:
            continue
        headings = [simple_clean(h) for h in (getattr(getattr(raw, "meta", None), "headings", None) or [])]
        raw_heading = headings[-1] if headings else "unknown"
        chunks.append({
            "chunk_index": len(chunks),
            "section":     raw_heading,
            "raw_heading": raw_heading,
            "text":        text,
            "headings":    headings,
        })

    if not chunks:
        return cv_markdown_to_chunks(markdown, schema, min_chars=min_chars, max_chars=max_chars)

    chunks = _merge_small_chunks(chunks, min_chars, max_chars)
    candidate_name = (schema or {}).get("candidate", {}).get("name", "") if isinstance(schema, dict) else ""
    return _enrich_chunks(chunks, candidate_name)

