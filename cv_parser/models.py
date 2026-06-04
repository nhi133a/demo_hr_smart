from __future__ import annotations

import html
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Annotated

# pyrefly: ignore [missing-import]
from pydantic import BaseModel, ConfigDict, Field, model_validator, BeforeValidator, AfterValidator

if TYPE_CHECKING:
    # pyrefly: ignore [missing-import]
    from docling.document_converter import DocumentConverter


EDUCATION_LEVELS = {"", "high_school", "bachelor", "bachelor_in_progress", "master", "phd"}
SKILL_TYPES      = {"", "technical", "tool", "process", "domain", "soft"}


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



CleanStr = Annotated[str, BeforeValidator(_clean_required)]
CleanOptionalStr = Annotated[Optional[str], BeforeValidator(_clean_optional)]
ConfidenceStr = Annotated[str, BeforeValidator(_normalize_confidence)]
ChunkIdsList = Annotated[List[str], BeforeValidator(_chunk_ids)]
PositiveFloat = Annotated[Optional[float], BeforeValidator(_positive_float)]
RoundedYears = Annotated[float, BeforeValidator(_rounded_years)]
IntMonths = Annotated[int, BeforeValidator(_months)]
Score100 = Annotated[int, BeforeValidator(_score_100)]
DictOrEmpty = Annotated[Dict[str, Any], BeforeValidator(_dict_or_empty)]
DictList = Annotated[List[Dict[str, Any]], BeforeValidator(_dict_list)]

EducationLevel = Annotated[Optional[str], BeforeValidator(_education_level)]
EducationStatus = Annotated[Optional[str], BeforeValidator(_education_status)]
SkillType = Annotated[str, BeforeValidator(_skill_type)]
SourceType = Annotated[str, BeforeValidator(_source_type)]
TechnologiesList = Annotated[List[str], BeforeValidator(_technologies)]

class CandidateSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: CleanStr = ""
    email: CleanStr = ""
    phone: CleanStr = ""
    location: CleanStr = ""
    dob: CleanOptionalStr = None
    github: CleanOptionalStr = None
    linkedin: CleanOptionalStr = None
    facebook: CleanOptionalStr = None

class MetadataFilterSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")
    education_min: EducationLevel = None
    exp_years_min: PositiveFloat = None
    location: CleanOptionalStr = None

class SkillSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: CleanStr = ""
    type: SkillType = ""
    confidence: ConfidenceStr = "medium"
    years: RoundedYears = 0.0
    evidence: CleanStr = ""
    source_section: CleanStr = ""
    source_type: CleanStr = ""
    source_chunk_ids: ChunkIdsList = Field(default_factory=list)

class SoftSkillSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: CleanStr = ""
    confidence: ConfidenceStr = "medium"
    evidence: CleanStr = ""
    source_chunk_ids: ChunkIdsList = Field(default_factory=list)

class LanguageSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: CleanStr = ""
    level: CleanStr = ""
    confidence: ConfidenceStr = "medium"
    evidence: CleanStr = ""
    source_chunk_ids: ChunkIdsList = Field(default_factory=list)

class CertificationSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: CleanStr = ""
    issuer: CleanStr = ""
    evidence: CleanStr = ""
    source_chunk_ids: ChunkIdsList = Field(default_factory=list)

class EducationSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")
    level: EducationLevel = None
    major: CleanOptionalStr = None
    school: CleanOptionalStr = None
    status: EducationStatus = None
    evidence: CleanOptionalStr = None
    source_chunk_ids: ChunkIdsList = Field(default_factory=list)

class ExperienceSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")
    title: CleanStr = ""
    company: CleanStr = ""
    start_date: CleanOptionalStr = None
    end_date: CleanOptionalStr = None
    years: RoundedYears = 0.0
    months: IntMonths = 0
    domain: CleanStr = ""
    quality_score: Score100 = 0
    evidence: CleanStr = ""
    source_chunk_ids: ChunkIdsList = Field(default_factory=list)

class ProjectSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: CleanStr = ""
    description: CleanStr = ""
    technologies: TechnologiesList = Field(default_factory=list)
    quality_score: Score100 = 0
    evidence: CleanStr = ""
    source_chunk_ids: ChunkIdsList = Field(default_factory=list)


class CVStructuredSchema(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)
    source_type: SourceType = "cv"
    candidate: Annotated[CandidateSchema, BeforeValidator(_dict_or_empty)] = Field(default_factory=CandidateSchema)
    metadata_filter: Annotated[MetadataFilterSchema, BeforeValidator(_dict_or_empty)] = Field(default_factory=MetadataFilterSchema)
    total_experience_years: PositiveFloat = None
    experience_months: int = 0
    experience_years: float = 0.0
    skills: Annotated[List[SkillSchema], BeforeValidator(_dict_list), AfterValidator(_filter_skills)] = Field(default_factory=list)
    soft_skills: Annotated[List[SoftSkillSchema], BeforeValidator(_dict_list), AfterValidator(_filter_soft_skills)] = Field(default_factory=list)
    languages: Annotated[List[LanguageSchema], BeforeValidator(_dict_list), AfterValidator(_filter_languages)] = Field(default_factory=list)
    certifications: Annotated[List[CertificationSchema], BeforeValidator(_dict_list), AfterValidator(_filter_certifications)] = Field(default_factory=list)
    education: Annotated[EducationSchema, BeforeValidator(_dict_or_empty)] = Field(default_factory=EducationSchema)
    experience: Annotated[List[ExperienceSchema], BeforeValidator(_dict_list), AfterValidator(_filter_experience)] = Field(default_factory=list)
    projects: Annotated[List[ProjectSchema], BeforeValidator(_dict_list), AfterValidator(_filter_projects)] = Field(default_factory=list)

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
        # Relaxed Education Validation: Keep education data if at least school or major is extracted.
        if not self.education.school and not self.education.major and not self.education.evidence:
            self.education = EducationSchema()
        return self



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
    match = re.search(r"(?P<yr>\d{4})(?:[-/](?P<mo>\d{1,2}))?", text)
    if not match:
        return None
    year = int(match.group("yr"))
    month = int(match.group("mo") or (12 if default_end else 1))
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

