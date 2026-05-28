from __future__ import annotations
import html
import json
import os
import re
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

if TYPE_CHECKING:
    from docling.document_converter import DocumentConverter


def build_converter(*, ocr: bool = False, table_structure: bool = True) -> DocumentConverter:
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    opts = PdfPipelineOptions(do_ocr=ocr, do_table_structure=table_structure)
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )

def fix_spaced_letters(text: str) -> str:
    return re.sub(r"\b(?:[A-Z]\s){3,}[A-Z]\b", lambda m: m.group(0).replace(" ", ""), text or "")


def clean_text(text: Any) -> str:
    text = html.unescape(str(text or ""))
    text = fix_spaced_letters(text)
    text = text.replace("\ufeff", "")
    text = re.sub(r"<!--\s*(?:image|picture|photo|avatar)\s*-->", "", text, flags=re.I)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _load_document(pdf_source: str | Path | bytes, converter: DocumentConverter):
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
    """Light text-only PDF fallback when Docling conversion is too expensive."""
    from io import BytesIO
    from PyPDF2 import PdfReader

    reader = PdfReader(BytesIO(pdf_source) if isinstance(pdf_source, bytes) else str(pdf_source))
    pages = [clean_text(page.extract_text() or "") for page in reader.pages]
    markdown = clean_text("\n\n---\n\n".join(page for page in pages if page))
    if not markdown:
        raise ValueError("PDF text fallback produced no text")
    return markdown


def convert_cv_pdf(
    pdf_source: str | Path | bytes,
    *,
    ocr: bool = False,
    converter: Optional[DocumentConverter] = None,
) -> tuple[Any | None, str]:
    """Convert a CV PDF; fall back to PyPDF2 text if Docling fails."""
    supplied_converter = converter is not None
    converter = converter or build_converter(ocr=ocr)
    try:
        doc = _load_document(pdf_source, converter)
        return doc, clean_text(doc.export_to_markdown())
    except Exception as docling_error:
        if not supplied_converter:
            try:
                light_doc = _load_document(
                    pdf_source,
                    build_converter(ocr=ocr, table_structure=False),
                )
                return light_doc, clean_text(light_doc.export_to_markdown())
            except Exception:
                pass
        try:
            return None, _pypdf_to_markdown(pdf_source)
        except Exception as fallback_error:
            raise RuntimeError(
                f"Docling conversion failed and PDF text fallback failed: {docling_error}"
            ) from fallback_error


def pdf_to_markdown(
    pdf_source: str | Path | bytes,
    *,
    include_toc: bool = False,
    ocr: bool = False,
    converter: Optional[DocumentConverter] = None,
) -> str:
    _, markdown = convert_cv_pdf(pdf_source, ocr=ocr, converter=converter)

    if include_toc:
        headings = re.findall(r"^(#{1,3}) (.+)", markdown, re.MULTILINE)
        if headings:
            toc = ["## Table of Contents\n"]
            for h, title in headings:
                slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
                toc.append("  " * (len(h) - 1) + f"- [{title}](#{slug})")
            markdown = "\n".join(toc) + "\n\n---\n\n" + markdown
    return markdown

EDUCATION_LEVELS = {"", "high_school", "bachelor", "bachelor_in_progress", "master", "phd"}
SKILL_TYPES      = {"", "technical", "tool", "process", "domain", "soft"}


CV_SCHEMA_PROMPT = """Parse this CV for JD matching. Return JSON only.
Rules:
- required_skills: explicit skills/tools/process/domain/soft skills written in CV.
- preferred_skills: skills directly inferred from experience/project, only with clear evidence.
- soft_skills: personal/work traits explicitly written in CV, e.g. teamwork, communication, accountability.
- languages: spoken/written languages and proficiency, e.g. English basic/intermediate.
- certifications: certificates/awards/training credentials explicitly listed.
- education: level, major/specialization, school, status, evidence.
- Normalize equivalent names; do not output low-confidence skills, job titles, desired roles, seniority labels, names, schools, locations.
- evidence: short CV quote. quality_score: 0-100 impact/complexity/technology.
- education_min: high_school|bachelor|master|phd|null.
- Experience: ONLY from Experience/Kinh nghiệm/Work History/Employment. Exclude Education, Projects, Summary, Objective, Certifications.
- Dates: YYYY-MM when possible; present/nay/hiện tại -> "present"; unknown -> null.
- years = full years, months = leftover months, not total months. If dates exist, years/months can be 0; Python recalculates.

CV TEXT:
{cv_text}

Schema:
{{"source_type":"cv","candidate":{{"name":"","email":"","phone":"","location":""}},"metadata_filter":{{"education_min":"high_school|bachelor|master|phd"|null,"location":string|null}},"education":{{"level":"high_school|bachelor|master|phd"|null,"major":string|null,"school":string|null,"status":"completed|in_progress"|null,"evidence":string|null}},"total_experience_years":number|null,"required_skills":[{{"name":"","type":"technical|tool|process|soft|domain","confidence":"high|medium|low","years":number,"evidence":""}}],"preferred_skills":[{{"name":"","type":"technical|tool|process|soft|domain","confidence":"high|medium|low","years":number,"evidence":""}}],"soft_skills":[{{"name":"","confidence":"high|medium|low","evidence":""}}],"languages":[{{"name":"","level":"","confidence":"high|medium|low","evidence":""}}],"certifications":[{{"name":"","issuer":"","evidence":""}}],"experience":[{{"title":"","company":"","start_date":"YYYY-MM"|null,"end_date":"YYYY-MM|present"|null,"years":number,"months":number,"domain":"","quality_score":0,"evidence":""}}],"projects":[{{"name":"","description":"","technologies":[],"quality_score":0}}],"_confidence":{{"required_skills":"high|medium","education_min":"high|medium|low"}}}}"""

def _as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


class CandidateSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str = ""
    email: str = ""
    phone: str = ""
    location: str = ""


class MetadataFilterSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")
    education_min: Optional[str] = None
    location: Optional[str] = None


class SkillSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str = ""
    type: str = ""
    confidence: str = "medium"
    years: float = 0.0
    evidence: str = ""

    @field_validator("confidence", mode="before")
    @classmethod
    def normalize_confidence(cls, value: Any) -> str:
        return _normalize_confidence(value)


class SoftSkillSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str = ""
    confidence: str = "medium"
    evidence: str = ""

    @field_validator("confidence", mode="before")
    @classmethod
    def normalize_confidence(cls, value: Any) -> str:
        return _normalize_confidence(value)


class LanguageSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str = ""
    level: str = ""
    confidence: str = "medium"
    evidence: str = ""

    @field_validator("confidence", mode="before")
    @classmethod
    def normalize_confidence(cls, value: Any) -> str:
        return _normalize_confidence(value)


class CertificationSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str = ""
    issuer: str = ""
    evidence: str = ""


class EducationSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")
    level: Optional[str] = None
    major: Optional[str] = None
    school: Optional[str] = None
    status: Optional[str] = None
    evidence: Optional[str] = None


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


class ProjectSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str = ""
    description: str = ""
    technologies: List[str] = Field(default_factory=list)
    quality_score: int = 0


class ConfidenceSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")
    required_skills: str = "high"
    education_min: str = "medium"

    @field_validator("required_skills", "education_min", mode="before")
    @classmethod
    def normalize_confidence(cls, value: Any) -> str:
        return _normalize_confidence(value)


class CVStructuredSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")
    source_type: str = "cv"
    candidate: CandidateSchema = Field(default_factory=CandidateSchema)
    metadata_filter: MetadataFilterSchema = Field(default_factory=MetadataFilterSchema)
    total_experience_years: Optional[float] = None
    required_skills: List[SkillSchema] = Field(default_factory=list)
    preferred_skills: List[SkillSchema] = Field(default_factory=list)
    soft_skills: List[SoftSkillSchema] = Field(default_factory=list)
    languages: List[LanguageSchema] = Field(default_factory=list)
    certifications: List[CertificationSchema] = Field(default_factory=list)
    education: EducationSchema = Field(default_factory=EducationSchema)
    experience: List[ExperienceSchema] = Field(default_factory=list)
    projects: List[ProjectSchema] = Field(default_factory=list)
    confidence: ConfidenceSchema = Field(default_factory=ConfidenceSchema, alias="_confidence")


def _dedupe_strings(items: List[str], limit: int = 80) -> List[str]:
    seen: set[str] = set()
    result: List[str] = []
    for item in items:
        item = clean_text(item)
        key = item.lower()
        if item and key not in seen:
            result.append(item)
            seen.add(key)
    return result[:limit]


def _clean_string_list(value: Any, limit: int = 40) -> List[str]:
    return _dedupe_strings(
        [clean_text(item) for item in _as_list(value) if isinstance(item, str)],
        limit=limit,
    )


def _to_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.search(r"\d+(?:\.\d+)?", value)
        if match:
            return float(match.group(0))
    return default


def _to_int(value: Any, default: int = 0) -> int:
    return max(0, int(round(_to_float(value, default))))


def _normalize_enum(value: Any, allowed: set[str]) -> str:
    value = str(value or "").strip().lower()
    return value if value in allowed else ""


def _clamp(value: Any, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, _to_float(value, lo)))


def _score_100(value: Any) -> int:
    return int(round(_clamp(value, 0, 100)))


def _parse_year_month(value: Any, *, default_end: bool = False) -> Optional[int]:
    text = clean_text(value).lower()
    if not text:
        return None
    if text in {"present", "current", "now", "ongoing", "nay", "hiện tại", "hiện nay"}:
        from datetime import datetime
        now = datetime.now()
        return now.year * 12 + now.month

    vn_match = re.search(r"th[áa]ng\s*(\d{1,2})[/\s-]*n[aă]m\s*(\d{4})", text)
    if vn_match:
        month = int(vn_match.group(1))
        year  = int(vn_match.group(2))
        if 1 <= month <= 12:
            return year * 12 + month

    month_year_match = re.search(r"(?P<month>\d{1,2})[-/](?P<year>\d{4})", text)
    if month_year_match:
        month = int(month_year_match.group("month"))
        year = int(month_year_match.group("year"))
        if 1 <= month <= 12:
            return year * 12 + month

    month_names = {
        "jan": 1,
        "january": 1,
        "feb": 2,
        "february": 2,
        "mar": 3,
        "march": 3,
        "apr": 4,
        "april": 4,
        "may": 5,
        "jun": 6,
        "june": 6,
        "jul": 7,
        "july": 7,
        "aug": 8,
        "august": 8,
        "sep": 9,
        "sept": 9,
        "september": 9,
        "oct": 10,
        "october": 10,
        "nov": 11,
        "november": 11,
        "dec": 12,
        "december": 12,
    }
    named_match = re.search(
        r"\b(?P<month>jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+(?P<year>\d{4})\b",
        text,
    )
    if named_match:
        return int(named_match.group("year")) * 12 + month_names[named_match.group("month")]

    match = re.search(r"(?P<year>\d{4})(?:[-/](?P<month>\d{1,2}))?", text)
    if not match:
        return None
    year = int(match.group("year"))
    month = int(match.group("month") or (12 if default_end else 1))
    if not 1 <= month <= 12:
        month = 12 if default_end else 1
    return year * 12 + month


def _sum_experience_months(rows: List[Dict[str, Any]]) -> int:
    intervals: List[tuple[int, int]] = []
    loose_months = 0
    loose_keys: set[str] = set()

    for row in rows:
        start = _parse_year_month(row.get("start_date"))
        end = _parse_year_month(row.get("end_date"), default_end=True)
        years = _to_float(row.get("years"), 0.0)
        months = _to_int(row.get("months"), 0)

        if start is not None and end is not None and end >= start:
            intervals.append((start, end))
            continue
        estimated_months = months if months > 12 else round(years * 12) + months
        if estimated_months <= 0:
            continue

        key = f"{clean_text(row.get('company')).lower()}::{clean_text(row.get('title')).lower()}::{estimated_months}"
        if key not in loose_keys:
            loose_keys.add(key)
            loose_months += estimated_months

    if loose_months > 600:
        loose_months = 0

    if not intervals:
        return loose_months

    intervals.sort()
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        last = merged[-1]
        if start <= last[1] + 1:
            last[1] = max(last[1], end)
        else:
            merged.append([start, end])

    return sum(end - start + 1 for start, end in merged) + loose_months


def _clean_dict_list(value: Any, allowed_keys: List[str], limit: int = 30) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in _as_list(value):
        if not isinstance(item, dict):
            continue
        row: Dict[str, Any] = {}
        for key in allowed_keys:
            cell = item.get(key)
            if isinstance(cell, str):
                cell = clean_text(cell)
            elif isinstance(cell, list):
                cell = _clean_string_list(cell, limit=30)
            elif cell is not None and not isinstance(cell, (int, float)):
                cell = str(cell)
            if cell not in (None, "", []):
                row[key] = cell
        if row:
            rows.append(row)
    return rows[:limit]


def _extract_json_object(text: str) -> Dict[str, Any]:
    text = clean_text(text)
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}


def _normalize_confidence(value: Any) -> str:
    if isinstance(value, str):
        value = value.lower().strip()
        if value in ("high", "medium", "low"):
            return value
    if isinstance(value, (int, float)):
        if value >= 0.7:
            return "high"
        elif value >= 0.4:
            return "medium"
        else:
            return "low"
    return "medium"

def _clean_skill_rows(value: Any) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in _as_list(value):
        if not isinstance(item, dict):
            continue
        name = clean_text(item.get("name"))
        if not name:
            continue

        # Normalize confidence to high/medium/low
        confidence = _normalize_confidence(item.get("confidence", "medium"))
        
        # Skip low confidence skills
        if confidence == "low":
            continue

        row: Dict[str, Any] = {
            "name":       name,
            "type":       _normalize_enum(item.get("type"), SKILL_TYPES),
            "confidence": confidence,
        }

        # Optional fields
        years = _to_float(item.get("years"), 0.0)
        if years > 0:
            row["years"] = round(years, 2)

        evidence = clean_text(item.get("evidence", ""))
        if evidence:
            row["evidence"] = evidence

        rows.append(row)
    return rows[:100]

def _normalize_cv_schema(data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        parsed = CVStructuredSchema.model_validate(data if isinstance(data, dict) else {})
        data = parsed.model_dump(by_alias=True)
    except ValidationError:
        data = data if isinstance(data, dict) else {}

    candidate = data.get("candidate") if isinstance(data.get("candidate"), dict) else {}
    metadata_filter = data.get("metadata_filter") if isinstance(data.get("metadata_filter"), dict) else {}
    education = data.get("education") if isinstance(data.get("education"), dict) else {}
    experience_rows = _clean_dict_list(
        data.get("experience"),
        ["title", "company", "start_date", "end_date", "years", "months", "domain", "quality_score", "evidence"],
        limit=40,
    )
    for row in experience_rows:
        row["years"] = round(_to_float(row.get("years"), 0.0), 2)
        if "months" in row:
            row["months"] = _to_int(row.get("months"), 0)
        row["quality_score"] = _score_100(row.get("quality_score", 0))

    python_experience_months = _sum_experience_months(experience_rows)
    llm_total_years = _to_float(data.get("total_experience_years"), 0.0)
    llm_experience_months = _to_int(llm_total_years * 12, 0) if llm_total_years > 0 else 0
    experience_months = llm_experience_months or python_experience_months
    experience_years = round(experience_months / 12, 2) if experience_months else 0.0

    schema: Dict[str, Any] = {
        "source_type": "cv",
        "candidate": {
            "name":     clean_text(candidate.get("name")),
            "email":    clean_text(candidate.get("email")),
            "phone":    clean_text(candidate.get("phone")),
            "location": clean_text(candidate.get("location")),
        },
        "metadata_filter": {
            "education_min":  _normalize_enum(metadata_filter.get("education_min"), EDUCATION_LEVELS) or None,
            "exp_years_min":  experience_years or None,
            "location":       clean_text(metadata_filter.get("location")) or None,
        },
        "education": {
            "level": _normalize_enum(education.get("level"), EDUCATION_LEVELS) or (
                _normalize_enum(metadata_filter.get("education_min"), EDUCATION_LEVELS) or None
            ),
            "major": clean_text(education.get("major")) or None,
            "school": clean_text(education.get("school")) or None,
            "status": _normalize_enum(education.get("status"), {"", "completed", "in_progress"}) or None,
            "evidence": clean_text(education.get("evidence")) or None,
        },
        
        # Required skills: explicitly mentioned or inferred as must-have
        "required_skills": _clean_skill_rows(data.get("required_skills", [])),

        # Preferred skills: nice-to-have or inferred
        "preferred_skills": _clean_skill_rows(data.get("preferred_skills", [])),
        "soft_skills": _clean_dict_list(
            data.get("soft_skills"),
            ["name", "confidence", "evidence"],
            limit=60,
        ),
        "languages": _clean_dict_list(
            data.get("languages"),
            ["name", "level", "confidence", "evidence"],
            limit=30,
        ),
        "certifications": _clean_dict_list(
            data.get("certifications"),
            ["name", "issuer", "evidence"],
            limit=30,
        ),

        "experience": experience_rows,
        "experience_months": experience_months,
        "experience_years": experience_years,
        "total_experience_years": experience_years,
        "projects": _clean_dict_list(
            data.get("projects"),
            ["name", "description", "technologies", "quality_score"],
            limit=40,
        ),
        
        # Overall confidence levels
        "_confidence": {
            "required_skills": _normalize_confidence(data.get("_confidence", {}).get("required_skills", "high")) if isinstance(data.get("_confidence"), dict) else "high",
            "education_min": _normalize_confidence(data.get("_confidence", {}).get("education_min", "medium")) if isinstance(data.get("_confidence"), dict) else "medium",
        },
    }

    for row in schema["projects"]:
        row["quality_score"] = _score_100(row.get("quality_score", 0))
    for row in schema["soft_skills"]:
        row["confidence"] = _normalize_confidence(row.get("confidence", "medium"))
    for row in schema["languages"]:
        row["confidence"] = _normalize_confidence(row.get("confidence", "medium"))

    return schema


def _duration_from_heading(label: str) -> tuple[float, int]:
    normalized = clean_text(label).lower()
    years_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:years?|yrs?|n[aă]m)\b", normalized)
    months_match = re.search(r"(\d+)\s*(?:months?|mos?|th[áa]ng)\b", normalized)
    years = _to_float(years_match.group(1), 0.0) if years_match else 0.0
    months = _to_int(months_match.group(1), 0) if months_match else 0
    return years, months


_DATE_TOKEN_RE = (
    r"(?:"
    r"\d{4}(?:[-/]\d{1,2})?"
    r"|\d{1,2}[-/]\d{4}"
    r"|(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
    r"aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{4}"
    r"|th[áa]ng\s*\d{1,2}(?:[/\s-]*n[aă]m)?\s*\d{4}"
    r")"
)
_PRESENT_TOKEN_RE = r"(?:present|current|now|ongoing|nay|hiện\s*tại|hiện\s*nay)"


def _date_range_from_label(label: str) -> tuple[Optional[str], Optional[str]]:
    match = re.search(
        rf"(?P<start>{_DATE_TOKEN_RE})\s*(?:-|–|—|to|đến|->)\s*(?P<end>{_DATE_TOKEN_RE}|{_PRESENT_TOKEN_RE})",
        clean_text(label),
        re.I,
    )
    if not match:
        return None, None

    start = clean_text(match.group("start"))
    end = clean_text(match.group("end"))
    if _parse_year_month(start) is None or _parse_year_month(end, default_end=True) is None:
        return None, None
    return start, end


def _is_experience_section_label(label: str) -> bool:
    normalized = re.sub(r"[^a-zA-ZÀ-ỹ\s]", " ", clean_text(label).lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return bool(re.fullmatch(
        r"(?:experience|work history|work experience|employment|professional experience|"
        r"kinh nghi[eệ]m(?: l[aà]m vi[eệ]c)?|kinh nghiem(?: lam viec)?)",
        normalized,
        re.I,
    ))


def _experience_row_from_label(label: str) -> Optional[Dict[str, Any]]:
    label = clean_text(re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", label))
    years, months = _duration_from_heading(label)
    start_date, end_date = _date_range_from_label(label)
    if years <= 0 and months <= 0 and not start_date:
        return None

    title_company = re.sub(
        r"\(?\s*\d+(?:\.\d+)?\s*(?:years?|yrs?|n[aă]m|months?|mos?|th[áa]ng)\s*\)?",
        "",
        label,
        flags=re.I,
    ).strip(" -()|")
    if start_date and end_date:
        title_company = re.sub(
            rf"\(?\s*{re.escape(start_date)}\s*(?:-|–|—|to|đến|->)\s*{re.escape(end_date)}\s*\)?",
            "",
            title_company,
            flags=re.I,
        ).strip(" -()|")
    title, _, company = title_company.partition(" - ")

    if not title_company:
        if years <= 0 and months <= 0:
            return None
        start_date = None
        end_date = None

    return {
        "title": title or title_company,
        "company": company,
        "start_date": start_date,
        "end_date": end_date,
        "years": years,
        "months": months,
        "evidence": label,
    }


def _explicit_experience_rows_from_markdown(markdown_text: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    in_experience = False

    for raw_line in str(markdown_text or "").splitlines():
        heading = re.match(r"^\s{0,3}#{1,6}\s+(.+?)\s*$", raw_line)
        label = clean_text(heading.group(1) if heading else raw_line)
        if not label:
            continue
        section = normalize_section_label(label)

        if _is_experience_section_label(label):
            in_experience = True
            continue

        is_section_heading = bool(heading) or bool(re.fullmatch(r"[A-Za-zÀ-ỹ\s]{2,60}", label))
        if in_experience and is_section_heading and section in {"skills", "projects", "education", "certifications", "summary"}:
            break
        if not in_experience:
            continue

        row = _experience_row_from_label(label)
        if row:
            rows.append(row)

    return rows


def _merge_explicit_experience_rows(
    rows: List[Dict[str, Any]],
    fallback_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    merged = [dict(row) for row in rows]
    seen = {
        (
            clean_text(row.get("title")).lower(),
            clean_text(row.get("company")).lower(),
            _to_int(row.get("months"), 0),
            round(_to_float(row.get("years"), 0.0), 2),
        )
        for row in merged
    }

    for fallback in fallback_rows:
        key = (
            clean_text(fallback.get("title")).lower(),
            clean_text(fallback.get("company")).lower(),
            _to_int(fallback.get("months"), 0),
            round(_to_float(fallback.get("years"), 0.0), 2),
        )
        if key not in seen:
            merged.append(fallback)
            seen.add(key)
    return merged


def _drop_llm_experience_durations(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned: List[Dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item.pop("start_date", None)
        item.pop("end_date", None)
        item["years"] = 0.0
        item["months"] = 0
        cleaned.append(item)
    return cleaned


def _llm_call(prompt: str) -> str:
    try:
        from llm_provider import generate_answer
        return generate_answer(prompt)
    except Exception:
        return ""

def _fallback_candidate_fields(text: str) -> Dict[str, str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    email = re.search(r"[\w.\-+]+@[\w.\-]+\.\w+", text)
    phone = re.search(r"(?:\+?\d[\d\s().-]{7,}\d)", text)

    name = ""
    bad_name_re = re.compile(
        r"\b(?:email|gmail|phone|facebook|linkedin|address|location|student|university|"
        r"specialization|summary|objective|experience|skill|strength|certified|"
        r"accountability|open-minded|inquisitive|seeking|residing|image|picture|"
        r"photo|avatar|personal|information|contact|profile|career|language|"
        r"proficiency|education|project|application|system|service|booking|lab|"
        r"infrastructure|objective|technical|tools|activities)\b",
        re.I,
    )
    bad_heading_keys = {
        "personal information",
        "contact information",
        "language proficiency",
        "career objective",
        "technical skills",
        "soft skills",
        "soft skills and activities",
        "work experience",
        "professional experience",
        "education",
        "project",
        "projects",
        "profile",
        "summary",
        "strength",
    }
    for line in lines[:80]:
        candidate = re.sub(r"^#+\s*", "", line).strip()
        candidate_key = re.sub(r"[^a-z0-9]+", " ", candidate.lower()).strip()
        if (
            re.search(r"<!--.*?-->", candidate)
            or re.search(r"[<>]", candidate)
            or re.fullmatch(r"[-_ ]*(?:image|picture|photo|avatar)[-_ ]*", candidate, re.I)
            or candidate_key in bad_heading_keys
            or "@" in candidate
            or re.search(r"\d", candidate)
            or ":" in candidate
            or len(candidate) > 80
            or len(candidate.split()) < 2
            or len(candidate.split()) > 5
            or re.search(r"[.!?]$", candidate)
            or bad_name_re.search(candidate)
        ):
            continue
        name = candidate
        break

    return {
        "name":     name,
        "email":    email.group(0) if email else "",
        "phone":    phone.group(0).strip() if phone else "",
        "location": "",
    }


def _extract_cv_schema_single_prompt_data(cv_text: str) -> Dict[str, Any]:
    raw = _llm_call(CV_SCHEMA_PROMPT.format(cv_text=cv_text[:24000]))
    return _extract_json_object(raw)


def _finalize_cv_schema(raw_data: Dict[str, Any], cv_text: str) -> Dict[str, Any]:
    has_llm_total = _to_float(raw_data.get("total_experience_years"), 0.0) > 0
    schema  = _normalize_cv_schema(raw_data)
    schema["experience"] = _drop_llm_experience_durations(schema.get("experience", []))
    if not has_llm_total:
        schema.pop("total_experience_years", None)
    fallback_rows = _explicit_experience_rows_from_markdown(cv_text)
    schema["experience"] = _merge_explicit_experience_rows(schema.get("experience", []), fallback_rows)
    schema = _normalize_cv_schema(schema)

    # Fallback: rescue missing candidate identity fields without overwriting LLM values.
    fallback_candidate = _fallback_candidate_fields(cv_text)
    for key, value in fallback_candidate.items():
        if value and not schema["candidate"].get(key):
            schema["candidate"][key] = value

    return schema


def extract_cv_schema(cv_text: str) -> Dict[str, Any]:
    cv_text = clean_text(cv_text)
    raw_data = _extract_cv_schema_single_prompt_data(cv_text)
    return _finalize_cv_schema(raw_data, cv_text)


def _skill_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9+#]+", "", clean_text(value).lower())


def _skill_evidence_in_text(skill: str, evidence: str, text: str) -> bool:
    skill = clean_text(skill)
    evidence = clean_text(evidence)
    text_lower = clean_text(text).lower()
    if not skill:
        return False
    if re.search(r"(?<![a-z0-9])" + re.escape(skill.lower()) + r"(?![a-z0-9])", text_lower):
        return True
    return bool(evidence and evidence.lower() in text_lower)


def _merge_skill_rows(existing: List[Dict[str, Any]], additions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged = _clean_skill_rows(existing)
    by_key = {_skill_key(row.get("name")): row for row in merged}
    rank = {"": 0, "low": 1, "medium": 2, "high": 3}

    for row in _clean_skill_rows(additions):
        key = _skill_key(row.get("name"))
        if not key:
            continue
        current = by_key.get(key)
        if not current:
            merged.append(row)
            by_key[key] = row
            continue
        if rank.get(row.get("confidence", ""), 0) > rank.get(current.get("confidence", ""), 0):
            current["confidence"] = row["confidence"]
        if row.get("evidence") and not current.get("evidence"):
            current["evidence"] = row["evidence"]
        if row.get("type") and not current.get("type"):
            current["type"] = row["type"]
    return merged[:100]


def _skill_type_for_name(name: str) -> str:
    lowered = name.lower()
    if re.search(r"\b(testing|reporting|tracking|planning|execution|analysis|design|automation|research|forecasting|budgeting|recruitment|interviewing)\b", lowered):
        return "process"
    if re.search(r"\b(sql|mysql|mongodb|postgresql|jira|trello|postman|bruno|selenium|cypress|playwright|excel|power bi|tableau|git|docker|crm|salesforce|hubspot|google ads|meta ads)\b", lowered):
        return "tool"
    if re.search(r"\b(sales|marketing|finance|accounting|hr|human resources|customer|retail|operation|logistics|b2b|b2c|seo|sem)\b", lowered):
        return "domain"
    if re.search(r"\b(communication|negotiation|leadership|teamwork|problem solving|presentation)\b", lowered):
        return "soft"
    return "technical"


def _canonical_chunk_skill_name(value: Any) -> str:
    text = clean_text(value)
    text = re.sub(r"\s*\((?:basic|beginner|intermediate|advanced|expert)\)\s*$", "", text, flags=re.I)
    text = text.strip(" -.;:,()[]")
    if not text:
        return ""
    lowered = text.lower()
    known = {
        "api": "API",
        "sql": "SQL",
        "crm": "CRM",
        "seo": "SEO",
        "sem": "SEM",
        "kpi": "KPI",
        "b2b": "B2B",
        "b2c": "B2C",
        "mysql": "MySQL",
        "mongodb": "MongoDB",
        "postgresql": "PostgreSQL",
        "node.js": "Node.js",
        "nodejs": "Node.js",
        "rest api": "REST API",
        "uat": "UAT",
        "sdlc": "SDLC",
        "ui": "UI",
        "ux": "UX",
    }
    if lowered in known:
        return known[lowered]
    replacements = {
        "api testing": "API Testing",
        "manual testing": "Manual Testing",
        "automation testing": "Automation Testing",
        "functional testing": "Functional Testing",
        "regression testing": "Regression Testing",
        "uat testing": "UAT Testing",
        "test case design": "Test Case Design",
        "bug reporting": "Bug Reporting",
        "bug tracking": "Bug Tracking",
        "defect reporting": "Defect Reporting",
        "test planning": "Test Planning",
        "test execution": "Test Execution",
        "requirements analysis": "Requirements Analysis",
        "business requirements analysis": "Business Requirements Analysis",
        "lead generation": "Lead Generation",
        "account management": "Account Management",
        "customer relationship management": "Customer Relationship Management",
        "market research": "Market Research",
        "revenue growth": "Revenue Growth",
        "google ads": "Google Ads",
        "meta ads": "Meta Ads",
        "microsoft excel": "Microsoft Excel",
        "power bi": "Power BI",
        "system analysis": "System Analysis",
        "result validation": "Result Validation",
        "web automation": "Web Automation",
    }
    return replacements.get(lowered, text if any(ch.isupper() for ch in text) else text.title())


def _valid_chunk_skill_name(name: Any) -> bool:
    text = clean_text(name)
    if not text or len(text) > 45:
        return False
    lowered = text.lower()
    if lowered in {
        "testing",
        "test cases",
        "developers",
        "speaking",
        "listening",
        "reading",
        "writing",
        "mobile",
        "web",
        "skills",
        "tools",
        "competencies",
        "expertise",
    }:
        return False
    if re.search(
        r"\b(?:future|exposure|project|application|service|team members?|engineers?|intern|users?|profiles?|"
        r"potential|risks?|expected\s+vs|actual results?|request$|response data|workflows?|professional experience yet)\b",
        lowered,
    ):
        return False
    return len(re.findall(r"[a-z0-9+#.-]+", lowered)) <= 5


def _filter_chunk_skill_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        name = _canonical_chunk_skill_name(item.get("name", ""))
        if not _valid_chunk_skill_name(name):
            continue
        item["name"] = name
        item["type"] = _skill_type_for_name(name)
        filtered.append(item)
    return _clean_skill_rows(filtered)


def _evidence_snippet(text: str, start: int, end: int, radius: int = 90) -> str:
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    return clean_text(text[left:right])


def _snippet_for_skill_name(name: str, text: str) -> str:
    cleaned_name = clean_text(name)
    aliases = {
        "Manual Testing": [r"\bmanual\s+test(?:ing| cases?)\b"],
        "Test Case Design": [r"\btest\s+case\s+design\b", r"\bdesign(?:ed|ing)?\s+(?:and\s+execut(?:ed|ing)\s+)?(?:manual\s+)?test\s+cases?\b"],
        "API Testing": [r"\bapi\s+testing\b"],
    }
    patterns = aliases.get(cleaned_name, [r"(?<![a-z0-9])" + re.escape(cleaned_name) + r"(?![a-z0-9])"])
    for raw_pattern in patterns:
        match = re.search(raw_pattern, text, re.I)
        if match:
            return _evidence_snippet(text, match.start(), match.end())
    return ""


SKILL_LINE_LABEL_RE = re.compile(
    r"^\s*(?:[-*]\s*)?(?:"
    r"technical\s+skills?|soft\s+skills?|hard\s+skills?|skills?|tools?|technolog(?:y|ies)|"
    r"core\s+competenc(?:y|ies)|competenc(?:y|ies)|expertise|areas?\s+of\s+expertise|"
    r"professional\s+skills?|key\s+skills?|domain\s+skills?|business\s+skills?|"
    r"sales\s+skills?|marketing\s+skills?|hr\s+skills?|finance\s+skills?|"
    r"backend|frontend|database|databases|programming\s+languages?|languages?|frameworks?|platforms?|"
    r"k[yỹ]\s*n[aă]ng|chuy[eê]n\s*m[oô]n|n[aă]ng\s*l[uự]c|c[oô]ng\s*c[uụ]|ng[oô]n\s*ng[uữ]"
    r")\s*[:\-]\s*(?P<items>.+)$",
    re.I,
)


TRAILING_SKILL_NOISE_RE = re.compile(
    r"\b(?:fresher|no professional experience|actively learning|seeking|looking for|career objective|objective)\b",
    re.I,
)


def _clean_skill_item_text(value: Any) -> str:
    text = clean_text(value)
    text = re.sub(r"\s*\([^)]*\)", "", text)
    text = TRAILING_SKILL_NOISE_RE.split(text, maxsplit=1)[0]
    text = re.sub(r"^(?:and|or|và|hoặc)\s+", "", text, flags=re.I)
    return clean_text(text).strip(" -.;:,()[]")


def _split_skill_items(value: str) -> List[str]:
    text = re.sub(r"\s+(?:and|or|và|hoặc)\s+", ",", clean_text(value), flags=re.I)
    parts = re.split(r"[,;|•·]+", text)
    return [_clean_skill_item_text(part) for part in parts if _clean_skill_item_text(part)]


def _generic_skill_line_rows(text: str, confidence: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for raw_line in str(text or "").splitlines():
        line = clean_text(raw_line)
        if not line:
            continue
        match = SKILL_LINE_LABEL_RE.match(line)
        if not match:
            continue
        for item in _split_skill_items(match.group("items")):
            name = _canonical_chunk_skill_name(item)
            if not _valid_chunk_skill_name(name):
                continue
            rows.append({
                "name": name,
                "type": _skill_type_for_name(name),
                "confidence": confidence,
                "evidence": line[:500],
            })
    return rows


def _regex_skill_rows_from_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    phrase_patterns = [
        r"\b(?:manual|automation|api|functional|regression|uat|unit|integration|performance|security|web|mobile)\s+testing\b",
        r"\btest\s+(?:case\s+design|cases?|planning|execution|data preparation|documentation|results?)\b",
        r"\b(?:bug|defect)\s+(?:reporting|tracking|reproduction|verification)\b",
        r"\b(?:business\s+)?requirements?\s+analysis\b",
        r"\bweb\s+automation\b",
        r"\b(?:rest\s+api|sql|mysql|mongodb|postgresql|flask|express|node\.?js|jira|trello|postman|bruno|selenium|cypress|playwright|git|docker|excel|power\s+bi|tableau)\b",
    ]
    cue_pattern = re.compile(
        r"\b(?:backend|database|databases|tools?|technolog(?:y|ies)|programming languages?|languages?|frameworks?|platforms?|using|used|with|familiarity with|knowledge of|experience with|proficiency in)\s*[:\-]?\s+([^.\n;]{2,120})",
        re.I,
    )

    for chunk in chunks or []:
        if not isinstance(chunk, dict):
            continue
        text = clean_text(chunk.get("text") or chunk.get("content") or chunk.get("embedding_text") or "")
        if not text:
            continue
        section = normalize_section_label(str(chunk.get("section") or "unknown"))
        confidence = "high" if section in {"profile", "summary", "skills"} else "medium"
        rows.extend(_generic_skill_line_rows(text, confidence))
        for pattern in phrase_patterns:
            for match in re.finditer(pattern, text, re.I):
                name = _canonical_chunk_skill_name(match.group(0))
                rows.append({
                    "name": name,
                    "type": _skill_type_for_name(name),
                    "confidence": confidence,
                    "evidence": _evidence_snippet(text, match.start(), match.end()),
                })
        for match in cue_pattern.finditer(text):
            phrase = re.sub(r"\s+(?:and|or)\s+", ",", match.group(1), flags=re.I)
            for part in re.split(r"[,/|]", phrase):
                part = _clean_skill_item_text(part)
                if not part or len(part) > 40:
                    continue
                if re.search(r"\b(?:application|project|service|system|team|engineer|intern|user|data|workflows?)\b", part, re.I):
                    continue
                if not re.fullmatch(r"(?:[A-Z][A-Za-z0-9+#.-]{1,}|[A-Z]{2,})(?:\s+[A-Z][A-Za-z0-9+#.-]{1,}){0,2}", part):
                    continue
                name = _canonical_chunk_skill_name(part)
                rows.append({
                    "name": name,
                    "type": _skill_type_for_name(name),
                    "confidence": confidence,
                    "evidence": _evidence_snippet(text, match.start(), match.end()),
                })
    return _filter_chunk_skill_rows(rows)


def reconcile_cv_schema_with_chunks(schema: Dict[str, Any], chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Verify schema skills against raw chunks and rescue explicit missing skills."""
    schema = _normalize_cv_schema(schema)
    before_required = len(_as_list(schema.get("required_skills")))
    before_preferred = len(_as_list(schema.get("preferred_skills")))
    raw_text = "\n".join(clean_text(ch.get("text") or ch.get("content") or "") for ch in chunks or [] if isinstance(ch, dict))
    evidence_text = "\n".join(
        clean_text(ch.get("text") or ch.get("content") or "")
        for ch in chunks or []
        if isinstance(ch, dict) and normalize_section_label(str(ch.get("section") or "unknown")) in {"profile", "summary", "skills", "experience", "projects", "certifications"}
    ) or raw_text

    verified_required = [
        row for row in schema.get("required_skills", [])
        if _skill_evidence_in_text(row.get("name", ""), row.get("evidence", ""), raw_text)
    ]
    verified_preferred = [
        row for row in schema.get("preferred_skills", [])
        if _skill_evidence_in_text(row.get("name", ""), row.get("evidence", ""), raw_text)
    ]

    discovered = _regex_skill_rows_from_chunks(chunks)
    for row in discovered:
        exact_evidence = _snippet_for_skill_name(row.get("name", ""), evidence_text)
        if exact_evidence:
            row["evidence"] = exact_evidence

    schema["required_skills"] = _merge_skill_rows(
        verified_required,
        [row for row in discovered if row.get("confidence") == "high"],
    )
    schema["preferred_skills"] = _merge_skill_rows(
        verified_preferred,
        [row for row in discovered if row.get("confidence") != "high"],
    )
    schema = _normalize_cv_schema(schema)
    schema["_audit"] = {
        "skill_reconciliation": {
            "schema_required_before": before_required,
            "schema_preferred_before": before_preferred,
            "schema_required_after": len(_as_list(schema.get("required_skills"))),
            "schema_preferred_after": len(_as_list(schema.get("preferred_skills"))),
            "chunk_skill_candidates": len(discovered),
            "method": "regex_evidence",
        }
    }
    return schema


SECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("summary",       re.compile(r"\b(summary|objective|career objective|profile|about me)\b", re.I)),
    ("skills",        re.compile(r"\b(skills?|technical skills?|tools?|technolog(?:y|ies)|competenc(?:e|ies))\b", re.I)),
    ("experience",    re.compile(r"\b(experience|work experience|professional experience|employment|work history|internship|intern|kinh\s*nghi[eệ]m(?:\s*l[aà]m\s*vi[eệ]c)?|kinh\s*nghiem)\b", re.I)),
    ("projects",      re.compile(r"\b(projects?|portfolio)\b", re.I)),
    ("education",     re.compile(r"\b(education|academic|university|college|school|degree|gpa)\b", re.I)),
    ("certifications",re.compile(r"\b(certifications?|certificates?|awards?)\b", re.I)),
]


def normalize_section_label(label: str) -> str:
    label = clean_text(label).lower()
    for section, pattern in SECTION_PATTERNS:
        if pattern.search(label):
            return section
    return label or "unknown"


def _build_embedding_text(chunk: Dict[str, Any], schema: Dict[str, Any]) -> str:
    return clean_text(chunk.get("text", ""))

def _enrich_chunks(chunks: List[Dict[str, Any]], schema: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Prepare light raw chunks for RAG storage."""
    candidate = schema.get("candidate", {}) if isinstance(schema, dict) else {}
    enriched: List[Dict[str, Any]] = []

    for idx, chunk in enumerate(chunks):
        item = dict(chunk)
        item["chunk_index"]          = idx
        item["section"]              = normalize_section_label(item.get("section", "unknown"))
        item["text"]                 = clean_text(item.get("text", ""))
        item["headings"]             = [clean_text(h) for h in _as_list(item.get("headings")) if clean_text(h)]
        item["embedding_text"]       = _build_embedding_text(item, schema or {})
        if candidate.get("name"):
            item["candidate_name"] = candidate["name"]
        enriched.append(item)

    return enriched

def _merge_small_chunks(
    chunks: List[Dict[str, Any]],
    min_chars: int = 80,
    max_chars: int = 1200,
) -> List[Dict[str, Any]]:
    if not chunks:
        return []
    merged = [chunks[0].copy()]
    for current in chunks[1:]:
        previous   = merged[-1]
        same_section = current.get("section") == previous.get("section")
        can_merge    = len(previous.get("text", "")) + len(current.get("text", "")) <= max_chars
        if same_section and len(current.get("text", "")) < min_chars and can_merge:
            previous["text"] = clean_text(previous.get("text", "") + "\n" + current.get("text", ""))
        else:
            merged.append(current.copy())
    return merged

def cv_markdown_to_chunks(
    markdown_text: str,
    schema: Optional[Dict[str, Any]] = None,
    *,
    min_chars: int = 80,
    max_chars: int = 1200,
) -> List[Dict[str, Any]]:
    """Split markdown CV into light raw section chunks for vector search."""
    markdown_text = clean_text(markdown_text)
    include_heading_text = schema is None
    schema = _normalize_cv_schema(schema) if schema else {}

    chunks: List[Dict[str, Any]] = []
    current_heading = "profile"
    current_lines: List[str] = []

    def flush() -> None:
        nonlocal current_lines
        text = clean_text("\n".join(current_lines))
        if text:
            if include_heading_text and current_heading and current_heading != "profile":
                text = clean_text(f"{current_heading}\n{text}")
            chunks.append({
                "chunk_index": len(chunks),
                "section":     normalize_section_label(current_heading),
                "text":        text,
                "headings":    [current_heading],
            })
        current_lines.clear()

    for raw_line in markdown_text.splitlines():
        heading = re.match(r"^\s{0,3}#{1,4}\s+(.+?)\s*$", raw_line)
        if heading:
            flush()
            current_heading = clean_text(heading.group(1)) or "section"
        else:
            current_lines.append(raw_line.rstrip())
    flush()

    if not chunks and markdown_text:
        chunks = [{"chunk_index": 0, "section": "profile", "text": markdown_text, "headings": ["profile"]}]

    chunks = _merge_small_chunks(chunks, min_chars=min_chars, max_chars=max_chars)
    return _enrich_chunks(chunks, schema)


def cv_document_to_chunks(
    doc: Any,
    schema: Optional[Dict[str, Any]] = None,
    *,
    max_tokens: int = 512,
    min_chars: int = 80,
    max_chars: int = 1200,
) -> List[Dict[str, Any]]:
    """Chunk a Docling document for vector search without injecting schema keywords."""
    from docling.chunking import HybridChunker

    markdown = clean_text(doc.export_to_markdown())
    schema = _normalize_cv_schema(schema) if schema else {}
    chunker = HybridChunker(max_tokens=max_tokens, merge_peers=True)
    chunks: List[Dict[str, Any]] = []

    for raw in chunker.chunk(doc):
        text = clean_text(getattr(raw, "text", "") or "")
        if not text and hasattr(raw, "export_to_text"):
            text = clean_text(raw.export_to_text())
        if len(text) < 20:
            continue
        headings = [
            clean_text(h)
            for h in (getattr(getattr(raw, "meta", None), "headings", None) or [])
        ]
        section = normalize_section_label(headings[-1] if headings else "unknown")
        chunks.append({
            "chunk_index": len(chunks),
            "section": section,
            "text": text,
            "headings": headings,
        })

    if not chunks:
        return cv_markdown_to_chunks(markdown, schema, min_chars=min_chars, max_chars=max_chars)

    chunks = _merge_small_chunks(chunks, min_chars=min_chars, max_chars=max_chars)
    return _enrich_chunks(chunks, schema)


def cv_pdf_to_chunks(
    pdf_source: str | Path | bytes,
    *,
    max_tokens: int = 512,
    min_chars: int = 80,
    max_chars: int = 1200,
    converter: Optional[DocumentConverter] = None,
    ocr: bool = False,
    enrich_with_llm: bool = True,
) -> List[Dict[str, Any]]:
    """Chunk a PDF for RAG only; global schema extraction is a separate pass."""
    doc, markdown = convert_cv_pdf(pdf_source, ocr=ocr, converter=converter)
    if doc is None:
        return cv_markdown_to_chunks(markdown, min_chars=min_chars, max_chars=max_chars)
    return cv_document_to_chunks(doc, max_tokens=max_tokens, min_chars=min_chars, max_chars=max_chars)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Clean Docling CV parser/chunker")
    parser.add_argument("pdf")
    parser.add_argument("-o", "--output")
    parser.add_argument("--chunk",        action="store_true")
    parser.add_argument("--json",         action="store_true")
    parser.add_argument("--ocr",          action="store_true")
    parser.add_argument("--no-llm-enrich",action="store_true")
    parser.add_argument("--toc",          action="store_true")
    parser.add_argument("--max-tokens",   type=int, default=512)
    parser.add_argument("--min-chars",    type=int, default=80)
    args = parser.parse_args()

    conv = build_converter(ocr=args.ocr)
    if args.chunk:
        result = cv_pdf_to_chunks(
            args.pdf,
            max_tokens=args.max_tokens,
            min_chars=args.min_chars,
            converter=conv,
            enrich_with_llm=not args.no_llm_enrich,
        )
        output = json.dumps(result, ensure_ascii=False, indent=2)
    else:
        output = pdf_to_markdown(args.pdf, include_toc=args.toc, converter=conv)

    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"Saved: {args.output}")
    else:
        print(output)
