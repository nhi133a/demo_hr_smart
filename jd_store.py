import hashlib
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Tuple

# pyrefly: ignore [missing-import]
import certifi
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv
# pyrefly: ignore [missing-import]
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
# pyrefly: ignore [missing-import]
from pymongo import MongoClient, UpdateOne

from bedrock_utils import get_embedding
from jd_parser_prompt import get_jd_parser_prompt

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

LLM_PARSE_ENABLED     = os.getenv("JD_LLM_PARSE_ENABLED", "true").strip().lower() in {"1", "true", "yes"}
LLM_TIMEOUT_SECONDS   = max(3,    int(os.getenv("JD_LLM_PARSE_TIMEOUT_SECONDS", "25") or 25))
LLM_MAX_CHARS         = max(1000, int(os.getenv("JD_LLM_PARSE_MAX_CHARS",       "8000") or 8000))

MONGO_CLIENT_OPTS = {"maxPoolSize": 5, "minPoolSize": 1, "maxIdleTimeMS": 45000, "retryWrites": False}
JD_COLLECTION_NAME = "job_descriptions"
VECTOR_INDEX_NAME = "jd_vector_index"
JD_DB_NAME = "aws_rag_db"

LEVEL_ALIASES: Dict[str, str] = {
    "intern": "intern", "internship": "intern", "trainee": "intern", "thuc tap": "intern",
    "fresher": "fresher", "fresh graduate": "fresher", "entry level": "fresher", "entry-level": "fresher",
    "junior": "junior", "jr": "junior",
    "mid": "mid", "middle": "mid", "mid level": "mid", "mid-level": "mid",
    "senior": "senior", "sr": "senior",
    "lead": "lead", "tech lead": "lead", "team lead": "lead",
    "manager": "manager",
}

# Sections that become individual chunks
_SECTION_ORDER = ["required_skills", "preferred_skills", "responsibilities", "soft_skills", "experience", "education"]

SAMPLE_JDS = [
    {
        "id": "jd_001",
        "title": "Tester Intern / QA Intern",
        "text": (
            "Requirements: Manual testing, test case design, bug reporting, API testing.\n"
            "Experience: No professional experience required; real project or internship is preferred.\n"
            "Skills: Jira, Postman, basic Selenium, technical document reading.\n"
            "Soft skills: Careful, detail-oriented, basic English reading."
        ),
    },
    {
        "id": "jd_002",
        "title": "Backend Developer Intern",
        "text": (
            "Requirements: REST API design, database modeling, server-side logic.\n"
            "Experience: Personal project is preferred; professional experience is not required.\n"
            "Skills: Python or Node.js, SQL, Git, HTTP and JSON understanding.\n"
            "Soft skills: Teamwork, self-learning, good communication."
        ),
    },
    {
        "id": "jd_003",
        "title": "Frontend Developer Intern",
        "text": (
            "Requirements: Build responsive web UI and integrate APIs.\n"
            "Experience: Portfolio or real project is preferred; professional experience is not required.\n"
            "Skills: HTML, CSS, JavaScript, React or Vue.\n"
            "Soft skills: Creative, UI/UX attention, good communication."
        ),
    },
    {
        "id": "jd_004",
        "title": "Data Analyst Intern",
        "text": (
            "Requirements: Analyze data, build reports and dashboards.\n"
            "Experience: Data analysis project is preferred; professional experience is not required.\n"
            "Skills: Excel, SQL, Python with Pandas and Matplotlib, Power BI or Tableau.\n"
            "Soft skills: Logical thinking, accuracy, clear result presentation."
        ),
    },
]

_client = None


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class JDParseError(RuntimeError):
    def __init__(self, jd_id: str, parse_error: str):
        self.jd_id = jd_id
        self.parse_error = parse_error
        super().__init__(f"JD parse failed for '{jd_id}': {parse_error}")


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class JDFilterModel(BaseModel):
    """Computed locally for hard-filter at query time. Never sent to LLM."""
    model_config = ConfigDict(extra="ignore")
    required_skills:        List[str] = Field(default_factory=list)
    experience_min_months:  int | None = None
    target_level:           str | None = None


class JDMetadataFilter(BaseModel):
    model_config = ConfigDict(extra="ignore")

    education_min: str | None = None
    exp_years_min: float | None = None
    job_type: str | None = None
    location: str | None = None

    @field_validator("education_min", "job_type", "location", mode="before")
    @classmethod
    def _clean_optional_text(cls, v: Any) -> str | None:
        return _clean_text(v) or None

    @field_validator("exp_years_min", mode="before")
    @classmethod
    def _clean_years(cls, v: Any) -> float | None:
        if v in (None, ""):
            return None
        try:
            return max(0.0, float(v))
        except (TypeError, ValueError):
            return None


class JDRequirementUnit(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str
    category: Literal[
        "skill",
        "soft_skill",
        "experience",
        "education",
        "responsibility",
        "certification",
        "competency",
    ] = "skill"
    importance: Literal["required", "preferred", "responsibility"] = "required"
    source_section: str | None = None
    evidence: str = ""
    confidence: Literal["high", "medium", "low"] = "medium"
    alternative_group: str | None = None

    @field_validator("name", "evidence", mode="before")
    @classmethod
    def _clean_required_text(cls, v: Any) -> str:
        return _clean_text(v)

    @field_validator("source_section", "alternative_group", mode="before")
    @classmethod
    def _clean_optional_text(cls, v: Any) -> str | None:
        return _clean_text(v) or None

    @field_validator("category", mode="before")
    @classmethod
    def _normalize_category(cls, v: Any) -> str:
        key = _slug_text(v or "skill").replace(" ", "_")
        aliases = {
            "required_skill": "skill",
            "preferred_skill": "skill",
            "technical_skill": "skill",
            "softskill": "soft_skill",
            "domain": "competency",
            "process": "competency",
            "cert": "certification",
            "certificate": "certification",
        }
        return aliases.get(key, key or "skill")

    @field_validator("importance", mode="before")
    @classmethod
    def _normalize_importance(cls, v: Any) -> str:
        key = _slug_text(v or "required")
        if key in {"preferred", "nice to have", "plus", "advantage", "bonus"}:
            return "preferred"
        if key in {"responsibility", "task", "duty"}:
            return "responsibility"
        return "required"


class JDSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # Identity
    job_title:       str | None = None
    target_level:    str | None = Field(default=None, description="intern|fresher|junior|mid|senior|lead|manager")

    # Skills
    required_skills:  List[str] = Field(default_factory=list)
    preferred_skills: List[str] = Field(default_factory=list)
    competencies:     List[str] = Field(default_factory=list)   # domain/process, not tools
    soft_skills:      List[str] = Field(default_factory=list)

    # Responsibilities
    responsibilities: List[str] = Field(default_factory=list)

    # Experience
    exp_min_months: int  = Field(default=0,    description="Lower bound in months. 1 year = 12.")
    exp_max_months: int | None = Field(default=None, description="Upper bound in months if stated.")

    # Education + meta
    education_level: str | None = Field(default=None, description="high_school|bachelor|master|phd")
    job_type:        str | None = Field(default=None, description="full-time|part-time|contract|remote")
    location:        str | None = None

    metadata_filter: JDMetadataFilter = Field(default_factory=JDMetadataFilter)
    requirement_units: List[JDRequirementUnit] = Field(default_factory=list)

    # Computed — not from LLM
    filter: JDFilterModel = Field(default_factory=JDFilterModel, exclude=True)

    # Parse status — set by extract_jd_schema, never by LLM
    parse_status: Literal["success", "llm_disabled", "failed"] = Field(
        default="success", exclude=True,
        description="success: LLM parsed OK | llm_disabled: parse skipped | failed: all attempts exhausted"
    )
    parse_error: str | None = Field(default=None, exclude=True)

    # --- validators ---

    @field_validator("job_title", "job_type", "location", "education_level", mode="before")
    @classmethod
    def _clean_str(cls, v: Any) -> str | None:
        return _clean_text(v) or None

    @field_validator("target_level", mode="before")
    @classmethod
    def _norm_level(cls, v: Any) -> str | None:
        return normalize_level(v) or None

    @field_validator("required_skills", "preferred_skills", "competencies", "soft_skills", "responsibilities", mode="before")
    @classmethod
    def _clean_list(cls, v: Any) -> List[str]:
        return _clean_strings(v)

    @field_validator("exp_min_months", "exp_max_months", mode="before")
    @classmethod
    def _non_neg(cls, v: Any) -> int | None:
        if v in (None, ""):
            return None
        try:
            return max(0, int(float(v)))
        except (TypeError, ValueError):
            return None

    @field_validator("exp_min_months", mode="after")
    @classmethod
    def _default_min(cls, v: int | None) -> int:
        return v or 0

    @model_validator(mode="after")
    def _build_filter(self) -> "JDSchema":
        self._sync_from_requirement_units()
        self.filter = JDFilterModel(
            required_skills=list(self.required_skills),
            experience_min_months=self.exp_min_months or None,
            target_level=self.target_level,
        )
        return self

    def _sync_from_requirement_units(self) -> None:
        if not self.requirement_units:
            self.requirement_units = _units_from_legacy_fields(self)
        if not self.required_skills:
            self.required_skills = _unit_names(self.requirement_units, "skill", "required")
            self.required_skills.extend(_unit_names(self.requirement_units, "certification", "required"))
            self.required_skills = _clean_strings(self.required_skills)
        if not self.preferred_skills:
            self.preferred_skills = _unit_names(self.requirement_units, "skill", "preferred")
            self.preferred_skills.extend(_unit_names(self.requirement_units, "certification", "preferred"))
            self.preferred_skills = _clean_strings(self.preferred_skills)
        if not self.competencies:
            self.competencies = _unit_names(self.requirement_units, "competency", None)
        if not self.soft_skills:
            self.soft_skills = _unit_names(self.requirement_units, "soft_skill", None)
        if not self.responsibilities:
            self.responsibilities = _unit_names(self.requirement_units, "responsibility", None)
        self._sync_experience_and_education()

    def _sync_experience_and_education(self) -> None:
        if self.metadata_filter.exp_years_min is not None and not self.exp_min_months:
            self.exp_min_months = int(round(self.metadata_filter.exp_years_min * 12))
        for unit in self.requirement_units:
            text = f"{unit.name} {unit.evidence}"
            if unit.category == "experience" and not self.exp_min_months:
                months = _months_from_text(text)
                if months:
                    self.exp_min_months = months
            if unit.category == "education" and not self.education_level:
                self.education_level = _education_level_from_text(text)
        if self.metadata_filter.education_min and not self.education_level:
            self.education_level = _education_level_from_text(self.metadata_filter.education_min)
        self.job_type = self.job_type or self.metadata_filter.job_type
        self.location = self.location or self.metadata_filter.location

    def to_storage_dict(self) -> Dict:
        data = self.model_dump()
        data["source_type"] = "jd"
        data["metadata_filter"] = {
            "education_min": self.education_level or self.metadata_filter.education_min,
            "exp_years_min": round(self.exp_min_months / 12, 2) if self.exp_min_months else self.metadata_filter.exp_years_min,
            "job_type": self.job_type or self.metadata_filter.job_type,
            "location": self.location or self.metadata_filter.location,
            "target_level": self.target_level,
        }
        data["experience"] = {
            "min_months": self.exp_min_months,
            "max_months": self.exp_max_months,
            "required": bool(self.exp_min_months),
            "exclude_internship": False,
            "evidence": _first_unit_evidence(self.requirement_units, "experience"),
        }
        data["education"] = {
            "level": self.education_level,
            "major": None,
            "required": bool(self.education_level),
            "evidence": _first_unit_evidence(self.requirement_units, "education") or None,
        }
        data["work_context"] = {
            "role": self.job_title,
            "seniority": self.target_level,
            "team_size": None,
            "report_to": None,
            "industry": None,
            "company_type": None,
            "domains": [],
            "platforms": [],
            "tools": _clean_strings([*self.required_skills, *self.preferred_skills]),
            "processes": self.competencies,
        }
        data["scoring_config"] = _default_scoring_config()
        data["filter"] = self.filter.model_dump()
        return data


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _clean_strings(value: Any, limit: int = 60) -> List[str]:
    seen: set[str] = set()
    result: List[str] = []
    for item in (value if isinstance(value, list) else []):
        text = _clean_text(item)
        key = text.lower()
        if text and key not in seen:
            result.append(text)
            seen.add(key)
    return result[:limit]


def _unit_names(units: List[JDRequirementUnit], category: str, importance: str | None) -> List[str]:
    names = []
    for unit in units:
        if unit.category != category:
            continue
        if importance and unit.importance != importance:
            continue
        if unit.name:
            names.append(unit.name)
    return _clean_strings(names)


def _make_unit(
    name: str,
    category: str,
    importance: str,
    *,
    evidence: str = "",
    source_section: str | None = None,
) -> JDRequirementUnit | None:
    name = _clean_text(name)
    if not name:
        return None
    return JDRequirementUnit.model_validate(
        {
            "name": name,
            "category": category,
            "importance": importance,
            "source_section": source_section,
            "evidence": evidence or name,
            "confidence": "medium",
        }
    )


def _units_from_legacy_fields(schema: JDSchema) -> List[JDRequirementUnit]:
    units: List[JDRequirementUnit] = []
    specs = [
        (schema.required_skills, "skill", "required", "required_skills"),
        (schema.preferred_skills, "skill", "preferred", "preferred_skills"),
        (schema.competencies, "competency", "required", "competencies"),
        (schema.soft_skills, "soft_skill", "required", "soft_skills"),
        (schema.responsibilities, "responsibility", "responsibility", "responsibilities"),
    ]
    for rows, category, importance, source_section in specs:
        for item in rows:
            unit = _make_unit(item, category, importance, source_section=source_section)
            if unit:
                units.append(unit)
    if schema.exp_min_months:
        unit = _make_unit(
            f"{schema.exp_min_months} months experience",
            "experience",
            "required",
            source_section="experience",
        )
        if unit:
            units.append(unit)
    if schema.education_level:
        unit = _make_unit(schema.education_level, "education", "required", source_section="education")
        if unit:
            units.append(unit)
    return _dedupe_units(units)


def _dedupe_units(units: List[JDRequirementUnit]) -> List[JDRequirementUnit]:
    seen = set()
    result = []
    for unit in units:
        key = (unit.category, unit.importance, _slug_text(unit.name))
        if not unit.name or key in seen:
            continue
        seen.add(key)
        result.append(unit)
    return result


def _months_from_text(value: str) -> int:
    text = _slug_text(value)
    match = re.search(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)\b", text)
    if match:
        return int(round(float(match.group(1)) * 12))
    match = re.search(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:months?|mos?)\b", text)
    if match:
        return int(round(float(match.group(1))))
    return 0


def _education_level_from_text(value: str) -> str | None:
    text = _slug_text(value)
    if re.search(r"\b(phd|doctor|doctorate)\b", text):
        return "phd"
    if re.search(r"\b(master|msc|ms)\b", text):
        return "master"
    if re.search(r"\b(bachelor|university|college|degree|bs|bsc|ba)\b", text):
        return "bachelor"
    if re.search(r"\b(high school|secondary)\b", text):
        return "high_school"
    return _clean_text(value) or None


def _first_unit_evidence(units: List[JDRequirementUnit], category: str) -> str:
    for unit in units:
        if unit.category == category:
            return unit.evidence or unit.name
    return ""


def _default_scoring_config() -> Dict:
    return {
        "weights": {
            "required_skills": 0.50,
            "preferred_skills": 0.15,
            "competencies": 0.10,
            "experience": 0.15,
            "education": 0.05,
            "quality": 0.03,
            "context": 0.02,
        },
        "required_skill_min_match_rate": 0.0,
        "hard_filters": {"metadata": False, "required_skills": False, "experience": False, "education": False},
        "retrieval_weights": {"dense": 0.70, "lexical": 0.20, "schema": 0.10, "cross_encoder": 0.0, "llm_judge": 0.0},
    }


def _slug_text(value: str) -> str:
    import unicodedata
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).lower()
    text = re.sub(r"[_/&-]+", " ", text)
    text = re.sub(r"[^a-z0-9+#.\s]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_level(value: str | None) -> str:
    return LEVEL_ALIASES.get(_slug_text(value or ""), "")


# ---------------------------------------------------------------------------
# Step 1 — Text normalization
# ---------------------------------------------------------------------------

def normalize_jd_text(raw: str) -> str:
    """Strip HTML, normalize whitespace and bullet chars. Preserves line structure."""
    text = str(raw or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"<\s*br\s*/?\s*>",              "\n", text, flags=re.I)
    text = re.sub(r"</\s*(p|div|li|h[1-6]|tr)\s*>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.translate(str.maketrans({"•": "-", "·": "-", "–": "-", "—": "-"}))

    lines: List[str] = []
    for raw_line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", raw_line).strip()
        if not line:
            if lines and lines[-1]:   # collapse consecutive blanks
                lines.append("")
            continue
        lines.append(line)
    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# Step 2 — LLM parse → JDSchema
# ---------------------------------------------------------------------------

_PARSE_SYSTEM = """Extract structured data from the job description. Return ONLY valid JSON, no markdown, no explanation.

{
  "job_title":        string | null,
  "target_level":     "intern"|"fresher"|"junior"|"mid"|"senior"|"lead"|"manager" | null,
  "required_skills":  [string],
  "preferred_skills": [string],
  "competencies":     [string],
  "soft_skills":      [string],
  "responsibilities": [string],
  "exp_min_months":   integer | null,
  "exp_max_months":   integer | null,
  "education_level":  "high_school"|"bachelor"|"master"|"phd" | null,
  "job_type":         "full-time"|"part-time"|"contract"|"remote" | null,
  "location":         string | null
}

Rules:
- required_skills: ALL technical items stated as required (languages, frameworks, tools, DBs, platforms). Be exhaustive — never omit a skill mentioned as a requirement.
- preferred_skills: items explicitly marked "nice to have", "plus", or "preferred" only.
- competencies: domain/process knowledge, not tools. e.g. "System design", "Agile", "REST API design".
- soft_skills: interpersonal traits only. e.g. "Teamwork", "Attention to detail". Never put technical skills here.
- target_level: infer from (1) explicit label in text, (2) years required [0=fresher, 1-2=junior, 3-5=mid, 5+=senior, 7+/leading=lead], (3) scope signals ["lead a team"→lead, "no experience required"→fresher]. Return null only if no signal exists.
- exp_min_months: years × 12, lower bound of ranges. e.g. "3-5 years" → 36.
- responsibilities: one action per item, start with a verb.

Example:
Input: "Backend Engineer. Must know Python, FastAPI, PostgreSQL, Docker. Redis is a plus. 5+ years required. Will design microservices, lead code reviews, mentor juniors."
Output:
{"job_title":"Backend Engineer","target_level":"senior","required_skills":["Python","FastAPI","PostgreSQL","Docker"],"preferred_skills":["Redis"],"competencies":["Microservices architecture"],"soft_skills":[],"responsibilities":["Design microservices","Lead code reviews","Mentor junior engineers"],"exp_min_months":60,"exp_max_months":null,"education_level":null,"job_type":null,"location":null}
"""


def _build_parse_prompt(jd_text: str) -> str:
    return get_jd_parser_prompt(jd_text)


def _extract_json(raw: str) -> Dict:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", str(raw or "").strip(), flags=re.I | re.M)
    match = re.search(r"\{.*\}", text, re.S)
    candidates = [match.group(0)] if match else []
    candidates.append(text)
    for c in candidates:
        try:
            parsed = json.loads(c)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue
    raise ValueError("LLM did not return a valid JSON object")


def _call_llm(prompt: str) -> str:
    try:
        from llm_provider import generate_answer
    except Exception as exc:
        raise RuntimeError("LLM provider unavailable") from exc

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(generate_answer, prompt)
        try:
            answer = str(future.result(timeout=LLM_TIMEOUT_SECONDS) or "").strip()
        except FuturesTimeoutError:
            future.cancel()
            raise TimeoutError(f"LLM parse timed out after {LLM_TIMEOUT_SECONDS}s")

    if not answer:
        raise ValueError("LLM returned empty response")
    return answer


def extract_jd_schema(jd_text: str) -> JDSchema:
    text = normalize_jd_text(jd_text)
    if not text or not LLM_PARSE_ENABLED:
        return JDSchema(parse_status="llm_disabled")

    last_error: str = ""
    for attempt, char_limit in enumerate([LLM_MAX_CHARS, LLM_MAX_CHARS // 2], start=1):
        try:
            raw = _call_llm(_build_parse_prompt(text[:char_limit]))
            schema = JDSchema.model_validate(_extract_json(raw))
            schema.parse_status = "success"
            return schema

        except (TimeoutError, ValueError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt == 1:
                print(f"[WARN] JD parse attempt 1 failed ({last_error}). Retrying with {char_limit // 2} chars.")
            else:
                print(f"[WARN] JD parse failed after 2 attempts ({last_error}).")

        except Exception as exc:
            # Bad JSON, Pydantic validation error — retrying won't help
            last_error = f"{type(exc).__name__}: {exc}"
            print(f"[WARN] JD parse non-retryable error ({last_error}).")
            break

    return JDSchema(parse_status="failed", parse_error=last_error)


def _parse_jd_schema(data: Dict | JDSchema) -> JDSchema:
    if isinstance(data, JDSchema):
        return data
    if not isinstance(data, dict):
        raise ValueError("JD schema must be a dict")
    return JDSchema.model_validate(data)


def _normalize_jd_schema(data: Dict | JDSchema, *, fallback_text: str = "") -> Dict:
    return _parse_jd_schema(data).to_storage_dict()


# ---------------------------------------------------------------------------
# Step 3 — Schema → chunks
# ---------------------------------------------------------------------------

def _render_section(section: str, schema: JDSchema) -> str:
    if section == "required_skills":
        parts = []
        if schema.required_skills:
            parts.append("Required skills:\n" + "\n".join(f"- {s}" for s in schema.required_skills))
        if schema.competencies:
            parts.append("Competencies:\n" + "\n".join(f"- {c}" for c in schema.competencies))
        return "\n\n".join(parts)

    if section == "preferred_skills":
        if not schema.preferred_skills:
            return ""
        return "Preferred skills:\n" + "\n".join(f"- {s}" for s in schema.preferred_skills)

    if section == "responsibilities":
        return "\n".join(f"- {r}" for r in schema.responsibilities)

    if section == "soft_skills":
        return "\n".join(f"- {s}" for s in schema.soft_skills)

    if section == "experience":
        lines = []
        if schema.exp_min_months:
            lines.append(f"Minimum: {schema.exp_min_months} months")
        if schema.exp_max_months is not None:
            lines.append(f"Maximum: {schema.exp_max_months} months")
        return "\n".join(lines)

    if section == "education":
        lines = []
        if schema.education_level:
            lines.append(f"Level: {schema.education_level}")
        return "\n".join(lines)

    return ""


def _build_embedding_text(title: str, section: str, content: str, schema: JDSchema) -> str:
    parts = [f"[JD: {title}]", f"[SECTION: {section}]", content.strip()]

    req = schema.required_skills
    pref = schema.preferred_skills

    if section == "required_skills" and req:
        parts.append("[REQUIRED_SKILLS]\n" + "\n".join(f"- {s}" for s in req))
        if pref:
            parts.append("[PREFERRED_SKILLS]\n" + "\n".join(f"- {s}" for s in pref))

    if section == "preferred_skills" and pref:
        if req:
            parts.append("[REQUIRED_SKILLS]\n" + "\n".join(f"- {s}" for s in req))
        parts.append("[PREFERRED_SKILLS]\n" + "\n".join(f"- {s}" for s in pref))

    return "\n\n".join(p for p in parts if p)


def schema_to_chunks(jd_id: str, title: str, schema: JDSchema) -> List[Dict]:
    chunks = []
    for idx, section in enumerate(_SECTION_ORDER):
        content = _render_section(section, schema)
        if not content.strip():
            continue

        embedding_text = _build_embedding_text(title, section, content, schema)
        chunks.append({
            "chunk_id":      f"{jd_id}_{idx:02d}_{section}",
            "jd_id":         jd_id,
            "source":        jd_id,
            "title":         title,
            "section":       section,
            "chunk_index":   idx,
            "content":       content,
            "embedding_text": embedding_text,
            "content_hash":  hashlib.sha256(embedding_text.encode()).hexdigest()[:16],
            "jd_schema":     schema.to_storage_dict(),
            "target_level":  schema.target_level or "",
            "llm_skills":    list(schema.required_skills) if section == "required_skills" else [],
            "skill_text":    "\n".join(f"- {s}" for s in schema.required_skills) if section == "required_skills" else "",
            "filter":        schema.filter.model_dump(),
        })
    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _resolve_jd_input(jd: Dict) -> tuple[str, str, str]:
    jd_id = str(jd.get("id") or jd.get("jd_id") or jd.get("title") or "jd").strip()
    title = str(jd.get("title") or jd_id).strip()

    if isinstance(jd.get("sections"), dict):
        parts = [f"{k}:\n{v}" for k, v in jd["sections"].items() if str(v).strip()]
        raw_text = "\n\n".join(parts)
    else:
        raw_text = str(jd.get("text") or jd.get("description") or jd.get("content") or "").strip()

    # Prepend title if not already present in text
    if title and raw_text and not re.search(r"^\s*(?:job\s+title|title|position|role)\s*:", raw_text, re.I | re.M):
        raw_text = f"Job Title: {title}\n\n{raw_text}"

    return jd_id, title, raw_text


def process_jd(jd: Dict) -> Tuple[JDSchema, List[Dict]]:
    jd_id, title, raw_text = _resolve_jd_input(jd)

    # Override level if caller supplied it explicitly
    explicit_level = normalize_level(jd.get("target_level") or jd.get("level"))

    schema = extract_jd_schema(raw_text)

    if schema.parse_status == "failed":
        raise JDParseError(jd_id=jd_id, parse_error=schema.parse_error or "unknown error")

    if explicit_level:
        schema.target_level = explicit_level
        schema.filter.target_level = explicit_level

    chunks = schema_to_chunks(jd_id, title, schema)
    return schema, chunks


def jd_to_section_chunks(jd: Dict) -> List[Dict]:
    return process_jd(jd)[1]


def get_mongo_client():
    global _client
    if _client is None:
        _client = MongoClient(os.getenv("MONGO_URI"), tlsCAFile=certifi.where(), **MONGO_CLIENT_OPTS)
    return _client


def get_jd_store():
    return get_mongo_client()[JD_DB_NAME][JD_COLLECTION_NAME]


def count_indexed_jds() -> int:
    return get_jd_store().count_documents({})


def _project_fields(include_score: bool = False) -> Dict:
    fields = {
        "_id": 0,
        "chunk_id": 1,
        "jd_id": 1,
        "title": 1,
        "source": 1,
        "section": 1,
        "chunk_index": 1,
        "content": 1,
        "embedding_text": 1,
        "content_hash": 1,
        "jd_schema": 1,
        "target_level": 1,
        "llm_skills": 1,
        "skill_text": 1,
        "filter": 1,
    }
    if include_score:
        fields["score"] = {"$meta": "vectorSearchScore"}
    return fields


def search_similar_jds(query_embedding: list, k: int = 5) -> List[Dict]:
    pipeline = [
        {
            "$vectorSearch": {
                "index": VECTOR_INDEX_NAME,
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": max(k * 10, 50),
                "limit": k,
            }
        },
        {"$project": _project_fields(include_score=True)},
    ]
    return list(get_jd_store().aggregate(pipeline))


def list_indexed_jds() -> List[Dict]:
    pipeline = [
        {"$sort": {"chunk_index": 1}},
        {
            "$group": {
                "_id": "$jd_id",
                "title": {"$first": "$title"},
                "target_level": {"$first": "$target_level"},
                "chunk_count": {"$sum": 1},
            }
        },
        {"$sort": {"title": 1}},
        {"$project": {"_id": 0, "jd_id": "$_id", "title": 1, "target_level": 1, "chunk_count": 1}},
    ]
    return list(get_jd_store().aggregate(pipeline))


def get_jd_chunks(jd_id: str) -> List[Dict]:
    return list(get_jd_store().find({"jd_id": jd_id}, _project_fields()).sort("chunk_index", 1))


def ingest_jds(jds: List[Dict] | None = None, *, prune_missing: bool = True) -> int:
    collection = get_jd_store()
    operations: List[UpdateOne] = []
    valid_chunk_ids: List[str] = []

    for jd in jds or SAMPLE_JDS:
        for chunk in jd_to_section_chunks(jd):
            chunk_id = chunk["chunk_id"]
            content_hash = chunk["content_hash"]
            valid_chunk_ids.append(chunk_id)
            operations.append(
                UpdateOne(
                    {"chunk_id": chunk_id, "content_hash": {"$ne": content_hash}},
                    {
                        "$set": {
                            **chunk,
                            "embedding": get_embedding(chunk["embedding_text"]),
                            "updated_at": datetime.now(timezone.utc),
                        }
                    },
                    upsert=True,
                )
            )

    if operations:
        collection.bulk_write(operations, ordered=False)
    if prune_missing and valid_chunk_ids:
        collection.delete_many({"chunk_id": {"$nin": valid_chunk_ids}})
    return len(operations)


def _infer_jd_title(text: str, fallback: str = "Pasted Job Description") -> str:
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip(" #:-\t")
        if line:
            return line[:80]
    return fallback


def ingest_jd_text(
    jd_text: str,
    *,
    title: str | None = None,
    jd_id: str | None = None,
    target_level: str | None = None,
) -> Dict:
    text = str(jd_text or "").strip()
    if not text:
        raise ValueError("JD text is empty")

    digest = hashlib.sha256(text.encode()).hexdigest()[:12]
    resolved_title = (title or "").strip() or _infer_jd_title(text)
    resolved_id = (jd_id or f"pasted_jd_{digest}").strip()
    resolved_level = normalize_level(target_level)

    chunk_count = ingest_jds(
        [{"id": resolved_id, "title": resolved_title, "text": text, "target_level": resolved_level}],
        prune_missing=False,
    )
    return {
        "jd_id": resolved_id,
        "title": resolved_title,
        "target_level": resolved_level,
        "chunk_count": chunk_count,
    }
