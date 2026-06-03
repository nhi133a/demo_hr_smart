"""pdf_utils.py – CV PDF conversion, schema extraction, and chunking."""
from __future__ import annotations

import html
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

if TYPE_CHECKING:
    from docling.document_converter import DocumentConverter

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EDUCATION_LEVELS = {"", "high_school", "bachelor", "bachelor_in_progress", "master", "phd"}
SKILL_TYPES      = {"", "technical", "tool", "process", "domain", "soft"}
CV_LLM_PARSE_MAX_CHARS = max(4000, int(os.getenv("CV_LLM_PARSE_MAX_CHARS", "12000") or 12000))

SECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("summary",        re.compile(r"\b(summary|objective|career objective|profile|about me)\b", re.I)),
    ("skills",         re.compile(r"\b(skills?|technical skills?|tools?|technologies|tech stack|competenc(?:e|ies)|expertise|k[yỹ]\s*n[aă]ng|c[oô]ng\s*c[uụ])\b", re.I)),
    ("experience",     re.compile(r"\b(experience|work experience|professional experience|employment|work history|internship|intern|kinh\s*nghi[eệ]m|kinh\s*nghiem)\b", re.I)),
    ("projects",       re.compile(r"\b(projects?|portfolio|case stud(?:y|ies)|d[uự]\s*[aá]n)\b", re.I)),
    ("education",      re.compile(r"\b(education|academic|university|college|school|degree|gpa|h[oọ]c\s*v[aấ]n)\b", re.I)),
    ("certifications", re.compile(r"\b(certifications?|certificates?|awards?|training|licenses?|ch[uứ]ng\s*ch[iỉ])\b", re.I)),
    ("languages",      re.compile(r"\b(languages?|english|ielts|toeic|ng[oô]n\s*ng[uữ])\b", re.I)),
]

# ---------------------------------------------------------------------------
# Text and type helpers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NormalizedText:
    raw: str
    clean: str


def simple_clean(text: Any) -> str:
    text = html.unescape(str(text or ""))
    # Fix spaced uppercase acronyms: "H T M L" → "HTML"
    text = re.sub(r"\b(?:[A-Z]\s){3,}[A-Z]\b", lambda m: m.group(0).replace(" ", ""), text)
    text = text.replace("\ufeff", "")
    text = re.sub(r"<!--\s*(?:image|picture|photo|avatar)\s*-->", "", text, flags=re.I)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


clean_text = simple_clean


def normalize_text(raw: Any) -> NormalizedText:
    return raw if isinstance(raw, NormalizedText) else NormalizedText(raw=str(raw or ""), clean=simple_clean(raw))


def _clean_value(value: Any) -> str:
    return value.clean if isinstance(value, NormalizedText) else simple_clean(value)


def _clip(text: str | NormalizedText, limit: int) -> str:
    text = _clean_value(text)
    return text if len(text) <= limit else simple_clean(text[:limit])


def _as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _to_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        m = re.search(r"\d+(?:\.\d+)?", value)
        if m:
            return float(m.group(0))
    return default


def _to_int(value: Any, default: int = 0) -> int:
    return max(0, int(round(_to_float(value, default))))


def _normalize_enum(value: Any, allowed: set[str]) -> str:
    v = str(value or "").strip().lower()
    return v if v in allowed else ""


def _score_100(value: Any) -> int:
    return int(round(max(0.0, min(100.0, _to_float(value, 0.0)))))


def _normalize_confidence(value: Any) -> str:
    if isinstance(value, str):
        v = value.lower().strip()
        if v in ("high", "medium", "low"):
            return v
    if isinstance(value, (int, float)):
        if value >= 0.7:
            return "high"
        if value >= 0.4:
            return "medium"
        return "low"
    return "medium"


def _dedupe_strings(items: List[str], limit: int = 80) -> List[str]:
    seen: set[str] = set()
    result: List[str] = []
    for item in items:
        item = simple_clean(item)
        key = item.lower()
        if item and key not in seen:
            result.append(item)
            seen.add(key)
    return result[:limit]


def _clean_required(value: Any) -> str:
    return simple_clean(value)


def _clean_optional(value: Any) -> str | None:
    return simple_clean(value) or None


def _chunk_ids(value: Any) -> List[str]:
    return _dedupe_strings([str(x) for x in _as_list(value)], limit=30)


def _positive_float(value: Any) -> float | None:
    value = _to_float(value, 0.0)
    return value if value > 0 else None


def _rounded_years(value: Any) -> float:
    return round(_to_float(value, 0.0), 2)


def _months(value: Any) -> int:
    return _to_int(value, 0)


def _dict_or_empty(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _dict_list(value: Any) -> List[Dict[str, Any]]:
    return [item for item in _as_list(value) if isinstance(item, dict)]


def _education_level(value: Any) -> str | None:
    return _normalize_enum(value, EDUCATION_LEVELS) or None


def _education_status(value: Any) -> str | None:
    return _normalize_enum(value, {"", "completed", "in_progress"}) or None


def _skill_type(value: Any) -> str:
    return _normalize_enum(value, SKILL_TYPES)


def _source_type(value: Any) -> str:
    return simple_clean(value) or "cv"


def _technologies(value: Any) -> List[str]:
    return _dedupe_strings([str(x) for x in _as_list(value)], limit=30)


def _filter_skills(rows: list[Any]) -> list[Any]:
    return [row for row in rows if row.name and row.confidence != "low" and row.evidence and row.source_chunk_ids][:100]


def _filter_soft_skills(rows: list[Any]) -> list[Any]:
    return [row for row in rows if row.name and row.evidence and row.source_chunk_ids][:60]


def _filter_languages(rows: list[Any]) -> list[Any]:
    return [row for row in rows if row.name and row.evidence and row.source_chunk_ids][:30]


def _filter_certifications(rows: list[Any]) -> list[Any]:
    return [row for row in rows if (row.name or row.issuer) and row.evidence and row.source_chunk_ids][:30]


def _filter_experience(rows: list[Any]) -> list[Any]:
    return [row for row in rows if (row.title or row.company) and row.evidence and row.source_chunk_ids][:40]


def _filter_projects(rows: list[Any]) -> list[Any]:
    return [row for row in rows if (row.name or row.description or row.technologies) and row.evidence and row.source_chunk_ids][:40]


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class CandidateSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str = ""
    email: str = ""
    phone: str = ""
    location: str = ""

    _clean = field_validator("name", "email", "phone", "location", mode="before")(_clean_required)

class MetadataFilterSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")
    education_min: Optional[str] = None
    exp_years_min: Optional[float] = None
    location: Optional[str] = None

    _education = field_validator("education_min", mode="before")(_education_level)
    _location = field_validator("location", mode="before")(_clean_optional)
    _years = field_validator("exp_years_min", mode="before")(_positive_float)

class SkillSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str = ""
    type: str = ""
    confidence: str = "medium"
    years: float = 0.0
    evidence: str = ""
    source_section: str = ""
    source_type: str = ""
    source_chunk_ids: List[str] = Field(default_factory=list)

    _norm = field_validator("confidence", mode="before")(_normalize_confidence)
    _clean = field_validator("name", "evidence", "source_section", "source_type", mode="before")(_clean_required)
    _type = field_validator("type", mode="before")(_skill_type)
    _years = field_validator("years", mode="before")(_rounded_years)
    _chunk_ids = field_validator("source_chunk_ids", mode="before")(_chunk_ids)

class SoftSkillSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str = ""
    confidence: str = "medium"
    evidence: str = ""
    source_chunk_ids: List[str] = Field(default_factory=list)

    _norm = field_validator("confidence", mode="before")(_normalize_confidence)
    _clean = field_validator("name", "evidence", mode="before")(_clean_required)
    _chunk_ids = field_validator("source_chunk_ids", mode="before")(_chunk_ids)

class LanguageSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str = ""
    level: str = ""
    confidence: str = "medium"
    evidence: str = ""
    source_chunk_ids: List[str] = Field(default_factory=list)

    _norm = field_validator("confidence", mode="before")(_normalize_confidence)
    _clean = field_validator("name", "level", "evidence", mode="before")(_clean_required)
    _chunk_ids = field_validator("source_chunk_ids", mode="before")(_chunk_ids)

class CertificationSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str = ""
    issuer: str = ""
    evidence: str = ""
    source_chunk_ids: List[str] = Field(default_factory=list)

    _clean = field_validator("name", "issuer", "evidence", mode="before")(_clean_required)
    _chunk_ids = field_validator("source_chunk_ids", mode="before")(_chunk_ids)

class EducationSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")
    level: Optional[str] = None
    major: Optional[str] = None
    school: Optional[str] = None
    status: Optional[str] = None
    evidence: Optional[str] = None
    source_chunk_ids: List[str] = Field(default_factory=list)

    _level = field_validator("level", mode="before")(_education_level)
    _status = field_validator("status", mode="before")(_education_status)
    _clean = field_validator("major", "school", "evidence", mode="before")(_clean_optional)
    _chunk_ids = field_validator("source_chunk_ids", mode="before")(_chunk_ids)

class ExperienceSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")
    title: str = ""
    company: str = ""
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    years: float = 0.0
    months: int = 0
    domain: str = ""
    quality_score: int = 0
    evidence: str = ""
    source_chunk_ids: List[str] = Field(default_factory=list)

    _clean = field_validator("title", "company", "start_date", "end_date", "domain", "evidence", mode="before")(_clean_required)
    _years = field_validator("years", mode="before")(_rounded_years)
    _months = field_validator("months", mode="before")(_months)
    _quality = field_validator("quality_score", mode="before")(_score_100)
    _chunk_ids = field_validator("source_chunk_ids", mode="before")(_chunk_ids)

class ProjectSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str = ""
    description: str = ""
    technologies: List[str] = Field(default_factory=list)
    quality_score: int = 0
    evidence: str = ""
    source_chunk_ids: List[str] = Field(default_factory=list)

    _clean = field_validator("name", "description", "evidence", mode="before")(_clean_required)
    _technologies = field_validator("technologies", mode="before")(_technologies)
    _quality = field_validator("quality_score", mode="before")(_score_100)
    _chunk_ids = field_validator("source_chunk_ids", mode="before")(_chunk_ids)

class ConfidenceSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")
    skills: str = "high"
    education_min: str = "medium"

    _norm = field_validator("skills", "education_min", mode="before")(_normalize_confidence)

class CVStructuredSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")
    source_type: str = "cv"
    candidate: CandidateSchema = Field(default_factory=CandidateSchema)
    metadata_filter: MetadataFilterSchema = Field(default_factory=MetadataFilterSchema)
    total_experience_years: Optional[float] = None
    experience_months: int = 0
    experience_years: float = 0.0
    skills: List[SkillSchema] = Field(default_factory=list)
    soft_skills: List[SoftSkillSchema] = Field(default_factory=list)
    languages: List[LanguageSchema] = Field(default_factory=list)
    certifications: List[CertificationSchema] = Field(default_factory=list)
    education: EducationSchema = Field(default_factory=EducationSchema)
    experience: List[ExperienceSchema] = Field(default_factory=list)
    projects: List[ProjectSchema] = Field(default_factory=list)
    confidence: ConfidenceSchema = Field(default_factory=ConfidenceSchema, alias="_confidence")

    _source_type = field_validator("source_type", mode="before")(_source_type)
    _object_or_empty = field_validator("candidate", "metadata_filter", "education", "confidence", mode="before")(_dict_or_empty)
    _dict_list = field_validator("skills", "soft_skills", "languages", "certifications", "experience", "projects", mode="before")(_dict_list)
    _total_years = field_validator("total_experience_years", mode="before")(_positive_float)

    _skills = field_validator("skills", mode="after")(_filter_skills)
    _soft_skills = field_validator("soft_skills", mode="after")(_filter_soft_skills)
    _languages = field_validator("languages", mode="after")(_filter_languages)
    _certifications = field_validator("certifications", mode="after")(_filter_certifications)
    _experience = field_validator("experience", mode="after")(_filter_experience)
    _projects = field_validator("projects", mode="after")(_filter_projects)

    @model_validator(mode="after")
    def _derive_experience(self) -> "CVStructuredSchema":
        exp_rows = [row.model_dump() for row in self.experience]
        exp_months = _sum_experience_months(exp_rows)
        llm_years = _to_float(self.total_experience_years, 0.0)
        exp_months = _to_int(llm_years * 12) if llm_years > 0 else exp_months
        exp_years = round(exp_months / 12, 2) if exp_months else 0.0
        self.experience_months = exp_months
        self.experience_years = exp_years
        self.total_experience_years = exp_years
        self.metadata_filter.exp_years_min = exp_years or None
        if not self.education.evidence or not self.education.source_chunk_ids:
            self.education = EducationSchema()
        return self

# ---------------------------------------------------------------------------
# LLM prompt
# ---------------------------------------------------------------------------

CV_SCHEMA_PROMPT = """
Extract the CV into ONE valid JSON object matching the schema.

RULES:
- Output JSON only.
- Do not add explanations or markdown.
- Missing data -> null.
- Do not hallucinate.

TRACEABILITY (required):
- Every extracted skill, experience, education, and project object must contain:
  - source_chunk_ids
  - evidence
- evidence must be the exact text span from the CV.
- If no supporting chunk exists, omit the item.

EXTRACTION:
- Group experience using title, company, and dates from the CV.
- Infer skill type and source_section only when directly supported by context.
- Preserve schema structure and field names exactly.

CV:
{cv_text}

SCHEMA:
{schema_definition}
"""
# ---------------------------------------------------------------------------
# PDF conversion
# ---------------------------------------------------------------------------

def build_converter(*, ocr: bool = False, table_structure: bool = True) -> "DocumentConverter":
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
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
    from PyPDF2 import PdfReader

    reader = PdfReader(BytesIO(pdf_source) if isinstance(pdf_source, bytes) else str(pdf_source))
    pages = [simple_clean(p.extract_text() or "") for p in reader.pages]
    markdown = simple_clean("\n\n---\n\n".join(p for p in pages if p))
    if not markdown:
        raise ValueError("PDF text fallback produced no text")
    return markdown


def _ocr_to_markdown(pdf_source: str | Path | bytes, *, zoom: float = 2.0) -> str:
    import fitz
    import numpy as np
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
                line = simple_clean(text)
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

# ---------------------------------------------------------------------------
# Section classification (heading-only, no content scoring)
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Experience duration helpers (kept: needed for total_experience_years calc)
# ---------------------------------------------------------------------------

_MONTH_NAMES: Dict[str, int] = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}


def _parse_year_month(value: Any, *, default_end: bool = False) -> Optional[int]:
    """Return year*12 + month, or None."""
    text = simple_clean(value).lower()
    if not text:
        return None
    if text in {"present", "current", "now", "ongoing", "nay", "hiện tại", "hiện nay"}:
        from datetime import datetime
        now = datetime.now()
        return now.year * 12 + now.month

    # Vietnamese: tháng 3/2022
    vn = re.search(r"th[áa]ng\s*(\d{1,2})[/\s-]*n[aă]m\s*(\d{4})", text)
    if vn:
        m, y = int(vn.group(1)), int(vn.group(2))
        if 1 <= m <= 12:
            return y * 12 + m

    # MM/YYYY or MM-YYYY
    mm_yyyy = re.search(r"(?P<m>\d{1,2})[-/](?P<y>\d{4})", text)
    if mm_yyyy:
        m, y = int(mm_yyyy.group("m")), int(mm_yyyy.group("y"))
        if 1 <= m <= 12:
            return y * 12 + m

    # "March 2022"
    named = re.search(
        r"\b(?P<mon>jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?"
        r"|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+(?P<yr>\d{4})\b",
        text,
    )
    if named:
        return int(named.group("yr")) * 12 + _MONTH_NAMES[named.group("mon")]

    # Bare year
    m = re.search(r"(?P<yr>\d{4})(?:[-/](?P<mo>\d{1,2}))?", text)
    if not m:
        return None
    year = int(m.group("yr"))
    month = int(m.group("mo") or (12 if default_end else 1))
    if not 1 <= month <= 12:
        month = 12 if default_end else 1
    return year * 12 + month


def _sum_experience_months(rows: List[Dict[str, Any]]) -> int:
    """Merge overlapping date intervals + sum loose duration fields."""
    intervals: List[tuple[int, int]] = []
    loose_months = 0
    loose_seen: set[str] = set()

    for row in rows:
        start = _parse_year_month(row.get("start_date"))
        end   = _parse_year_month(row.get("end_date"), default_end=True)
        if start is not None and end is not None and end >= start:
            intervals.append((start, end))
            continue
        years  = _to_float(row.get("years"), 0.0)
        months = _to_int(row.get("months"), 0)
        est = months if months > 12 else round(years * 12) + months
        if est <= 0:
            continue
        company = simple_clean(row.get("company")).lower()
        title = simple_clean(row.get("title")).lower()
        key = f"{company}::{title}::{est}"
        if key not in loose_seen:
            loose_seen.add(key)
            loose_months += est

    if loose_months > 600:
        loose_months = 0
    if not intervals:
        return loose_months

    intervals.sort()
    merged = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])

    return sum(e - s + 1 for s, e in merged) + loose_months

# ---------------------------------------------------------------------------
# JSON / LLM helpers
# ---------------------------------------------------------------------------

def _extract_json_object(text: str) -> Dict[str, Any]:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", simple_clean(text), flags=re.I)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
        raise ValueError("LLM JSON response must be an object")
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            raise ValueError("LLM did not return a valid JSON object")
        try:
            parsed = json.loads(m.group(0))
            if isinstance(parsed, dict):
                return parsed
            raise ValueError("LLM JSON response must be an object")
        except json.JSONDecodeError:
            raise ValueError("LLM did not return a valid JSON object")


def _llm_call(prompt: str) -> str:
    try:
        from llm_provider import generate_answer
    except Exception as exc:
        raise RuntimeError("LLM provider is not available for CV parsing") from exc
    answer = str(generate_answer(prompt) or "")
    if not answer.strip():
        raise ValueError("LLM returned an empty CV parser response")
    return answer

# ---------------------------------------------------------------------------
# Schema normalization
# ---------------------------------------------------------------------------

def _normalize_cv_schema(data: Dict[str, Any]) -> Dict[str, Any]:
    raw = data if isinstance(data, dict) else {}
    return CVStructuredSchema.model_validate(raw).model_dump(by_alias=True)

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Schema extraction pipeline
# ---------------------------------------------------------------------------

def _build_llm_input_from_chunks(chunks: List[Dict[str, Any]], fallback_text: str | NormalizedText) -> str:
    """Build a compact, section-aware LLM input from existing chunks."""
    normalized = normalize_text(fallback_text)
    if not chunks:
        return _clip(normalized, CV_LLM_PARSE_MAX_CHARS)

    priority = {"profile": 0, "summary": 0, "skills": 1, "experience": 2,
                "projects": 3, "education": 4, "certifications": 5, "languages": 6, "unknown": 7}
    section_limits = {"profile": 1200, "summary": 1200, "skills": 2200, "experience": 4200,
                      "projects": 2800, "education": 1600, "certifications": 1000, "languages": 800, "unknown": 900}

    used: Dict[str, int] = {}
    parts: List[str] = []
    total = 0

    for chunk in sorted(chunks, key=lambda c: (priority.get(str(c.get("section") or "unknown"), 8), int(c.get("chunk_index", 0) or 0))):
        section = normalize_section_label(chunk.get("section", "unknown"))
        text    = simple_clean(chunk.get("text", ""))
        if not text:
            continue
        remaining = section_limits.get(section, 900) - used.get(section, 0)
        if remaining <= 0:
            continue
        text = _clip(text, min(remaining, 1800))
        raw_heading = simple_clean(chunk.get("raw_heading") or "")
        chunk_id = simple_clean(chunk.get("chunk_id") or f"chunk_{chunk.get('chunk_index', len(parts))}")
        header = f"[chunk_id={chunk_id} | section={section}]"
        if raw_heading and raw_heading.lower() != section:
            header = f"[chunk_id={chunk_id} | section={section} | heading={raw_heading[:80]}]"
        part = simple_clean(f"{header}\n{text}")
        if total + len(part) + 2 > CV_LLM_PARSE_MAX_CHARS:
            break
        parts.append(part)
        total += len(part) + 2
        used[section] = used.get(section, 0) + len(text)

    compact = simple_clean("\n\n".join(parts))
    return compact or _clip(normalized, CV_LLM_PARSE_MAX_CHARS)


def _build_llm_input(cv_text: str | NormalizedText) -> str:
    """Build a compact, section-aware LLM input from raw CV text."""
    normalized = normalize_text(cv_text)
    chunks = cv_markdown_to_chunks(normalized, {}, min_chars=40, max_chars=1800)
    return _build_llm_input_from_chunks(chunks, normalized)


def extract_cv_schema(cv_text: str | NormalizedText, *, chunks: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
    """Parse CV text → structured schema dict via LLM + Pydantic normalization."""
    normalized = normalize_text(cv_text)
    llm_input = _build_llm_input_from_chunks(chunks, normalized) if chunks is not None else _build_llm_input(normalized)
    schema_definition = json.dumps(CVStructuredSchema.model_json_schema(by_alias=True), ensure_ascii=False)
    raw_json = _extract_json_object(
        _llm_call(CV_SCHEMA_PROMPT.format(cv_text=llm_input, schema_definition=schema_definition))
    )
    return _normalize_cv_schema(raw_json)

