import hashlib
import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal

import certifi
from dotenv import load_dotenv
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator
from pymongo import MongoClient, UpdateOne

from bedrock_utils import get_embedding
from jd_parser_prompt import get_jd_parser_prompt

load_dotenv()

MONGO_CLIENT_OPTS = {"maxPoolSize": 5, "minPoolSize": 1, "maxIdleTimeMS": 45000, "retryWrites": False}

JD_LLM_PARSE_ENABLED = os.getenv("JD_LLM_PARSE_ENABLED", "true").strip().lower() in {"1", "true", "yes"}
JD_LLM_PARSE_TIMEOUT_SECONDS = max(3, int(os.getenv("JD_LLM_PARSE_TIMEOUT_SECONDS", "25") or 25))
JD_LLM_PARSE_MAX_CHARS = max(1000, int(os.getenv("JD_LLM_PARSE_MAX_CHARS", "8000") or 8000))

JD_COLLECTION_NAME = "job_descriptions"
VECTOR_INDEX_NAME = "jd_vector_index"
JD_DB_NAME = "aws_rag_db"

CANONICAL_SECTIONS = {
    "requirements",
    "responsibilities",
    "preferred_skills",
    "skills",
    "experience",
    "education",
    "soft_skills",
    "benefits",
}

SECTION_ALIASES = {
    "requirement": "requirements",
    "requirements": "requirements",
    "job requirements": "requirements",
    "required skills": "requirements",
    "must have": "requirements",
    "must-have": "requirements",
    "yeu cau": "requirements",
    "yeu cau cong viec": "requirements",
    "responsibility": "responsibilities",
    "responsibilities": "responsibilities",
    "job responsibilities": "responsibilities",
    "what you will do": "responsibilities",
    "duties": "responsibilities",
    "tasks": "responsibilities",
    "mo ta cong viec": "responsibilities",
    "preferred": "preferred_skills",
    "preferred skills": "preferred_skills",
    "nice to have": "preferred_skills",
    "nice-to-have": "preferred_skills",
    "bonus": "preferred_skills",
    "plus": "preferred_skills",
    "uu tien": "preferred_skills",
    "skills": "skills",
    "technical skills": "skills",
    "technical stack": "skills",
    "technology stack": "skills",
    "tech stack": "skills",
    "tools": "skills",
    "ky nang": "skills",
    "cong nghe": "skills",
    "experience": "experience",
    "work experience": "experience",
    "years of experience": "experience",
    "kinh nghiem": "experience",
    "education": "education",
    "degree": "education",
    "hoc van": "education",
    "soft skills": "soft_skills",
    "soft skill": "soft_skills",
    "ky nang mem": "soft_skills",
    "benefits": "benefits",
    "benefit": "benefits",
    "salary": "benefits",
    "compensation": "benefits",
    "quyen loi": "benefits",
    "phuc loi": "benefits",
}

LEVEL_ALIASES = {
    "intern": "intern",
    "internship": "intern",
    "trainee": "intern",
    "thuc tap": "intern",
    "fresher": "fresher",
    "fresh graduate": "fresher",
    "entry level": "fresher",
    "entry-level": "fresher",
    "junior": "junior",
    "jr": "junior",
    "mid": "mid",
    "middle": "mid",
    "mid level": "mid",
    "mid-level": "mid",
    "senior": "senior",
    "sr": "senior",
    "lead": "lead",
    "tech lead": "lead",
    "team lead": "lead",
    "manager": "manager",
}

SAMPLE_JDS = [
    {
        "id": "jd_001",
        "title": "Tester Intern / QA Intern",
        "sections": {
            "requirements": "Manual testing, test case design, bug reporting, API testing.",
            "experience": "No professional experience required; real project or internship is preferred.",
            "skills": "Jira, Postman, basic Selenium, technical document reading.",
            "soft_skills": "Careful, detail-oriented, basic English reading.",
        },
    },
    {
        "id": "jd_002",
        "title": "Backend Developer Intern",
        "sections": {
            "requirements": "REST API design, database modeling, server-side logic.",
            "experience": "Personal project is preferred; professional experience is not required.",
            "skills": "Python or Node.js, SQL, Git, HTTP and JSON understanding.",
            "soft_skills": "Teamwork, self-learning, good communication.",
        },
    },
    {
        "id": "jd_003",
        "title": "Frontend Developer Intern",
        "sections": {
            "requirements": "Build responsive web UI and integrate APIs.",
            "experience": "Portfolio or real project is preferred; professional experience is not required.",
            "skills": "HTML, CSS, JavaScript, React or Vue.",
            "soft_skills": "Creative, UI/UX attention, good communication.",
        },
    },
    {
        "id": "jd_004",
        "title": "Data Analyst Intern",
        "sections": {
            "requirements": "Analyze data, build reports and dashboards.",
            "experience": "Data analysis project is preferred; professional experience is not required.",
            "skills": "Excel, SQL, Python with Pandas and Matplotlib, Power BI or Tableau.",
            "soft_skills": "Logical thinking, accuracy, clear result presentation.",
        },
    },
]

_client = None


class JDFilterModel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    required_skills: List[str] = Field(default_factory=list)
    experience_min_months: int | None = None
    target_level: str | None = None


class JDMetadataFilter(BaseModel):
    model_config = ConfigDict(extra="ignore")

    education_min: str | None = None
    exp_years_min: float | None = None
    job_type: str | None = None
    location: str | None = None
    target_level: str | None = None

    @field_validator("education_min", "job_type", "location", "target_level", mode="before")
    @classmethod
    def _clean_optional_text(cls, value: Any) -> str | None:
        text = _clean_text(value)
        return text or None

    @field_validator("target_level")
    @classmethod
    def _normalize_target_level(cls, value: str | None) -> str | None:
        return normalize_level(value) or None


class JDWeights(BaseModel):
    model_config = ConfigDict(extra="ignore")

    required_skills: float = 0.45
    preferred_skills: float = 0.10
    competencies: float = 0.10
    experience: float = 0.15
    education: float = 0.05
    quality: float = 0.05
    context: float = 0.10


class JDHardFilters(BaseModel):
    model_config = ConfigDict(extra="ignore")

    metadata: bool = False
    required_skills: bool = False
    experience: bool = False
    education: bool = False




class JDRetrievalWeights(BaseModel):
    model_config = ConfigDict(extra="ignore")

    dense: float = 0.70
    lexical: float = 0.20
    schema_weight: float = Field(default=0.10, validation_alias=AliasChoices("schema_weight", "schema"))
    cross_encoder: float = 0.0
    llm_judge: float = 0.0


class JDScoringConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    weights: JDWeights = Field(default_factory=JDWeights)
    required_skill_min_match_rate: float = 0.0
    hard_filters: JDHardFilters = Field(default_factory=JDHardFilters)
    retrieval_weights: JDRetrievalWeights = Field(default_factory=JDRetrievalWeights)




class JDSkill(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str
    type: str | None = None
    confidence: Literal["high", "medium", "low"] = "medium"
    importance: Literal["required", "preferred"] = "required"
    evidence: str = ""
    alternative_group: str | None = None

    @field_validator("name", mode="before")
    @classmethod
    def _clean_name(cls, value: Any) -> str:
        return _clean_text(value)

    @field_validator("evidence", mode="before")
    @classmethod
    def _clean_evidence(cls, value: Any) -> str:
        return _clean_text(value)

    @field_validator("type", "alternative_group", mode="before")
    @classmethod
    def _clean_optional_fields(cls, value: Any) -> str | None:
        text = _clean_text(value)
        return text or None


class JDCompetency(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str
    category: Literal["technical", "process", "domain"] | None = None
    confidence: Literal["high", "medium", "low"] = "medium"
    importance: Literal["required", "preferred"] = "required"
    evidence: str = ""

    @field_validator("name", "evidence", mode="before")
    @classmethod
    def _clean_text_fields(cls, value: Any) -> str:
        return _clean_text(value)


class JDSoftSkill(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str
    confidence: Literal["high", "medium", "low"] = "medium"
    evidence: str = ""

    @field_validator("name", "evidence", mode="before")
    @classmethod
    def _clean_text_fields(cls, value: Any) -> str:
        return _clean_text(value)


class JDExperience(BaseModel):
    model_config = ConfigDict(extra="ignore")

    min_months: int = 0
    max_months: int | None = None
    required: bool = False
    exclude_internship: bool = False
    evidence: str = ""

    @field_validator("min_months", "max_months", mode="before")
    @classmethod
    def _non_negative_int(cls, value: Any) -> int | None:
        if value in (None, ""):
            return None
        if isinstance(value, str):
            match = re.search(r"\d+(?:\.\d+)?", value)
            if not match:
                raise ValueError("experience month value must be numeric")
            value = float(match.group(0))
        elif not isinstance(value, (int, float)):
            raise ValueError("experience month value must be numeric")
        return max(0, int(round(value)))

    @field_validator("min_months", mode="after")
    @classmethod
    def _default_min_months(cls, value: int | None) -> int:
        return value or 0

    @field_validator("evidence", mode="before")
    @classmethod
    def _clean_evidence(cls, value: Any) -> str:
        return _clean_text(value)


class JDEducation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    level: str | None = None
    major: str | None = None
    required: bool = False
    evidence: str | None = None

    @field_validator("level", "major", "evidence", mode="before")
    @classmethod
    def _clean_optional_text(cls, value: Any) -> str | None:
        text = _clean_text(value)
        return text or None


class JDWorkContext(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: str | None = None
    seniority: str | None = None
    team_size: str | None = None
    report_to: str | None = None
    industry: str | None = None
    company_type: str | None = None
    domains: List[str] = Field(default_factory=list)
    platforms: List[str] = Field(default_factory=list)
    tools: List[str] = Field(default_factory=list)
    processes: List[str] = Field(default_factory=list)

    @field_validator("role", "seniority", "team_size", "report_to", "industry", "company_type", mode="before")
    @classmethod
    def _clean_optional_text(cls, value: Any) -> str | None:
        text = _clean_text(value)
        return text or None

    @field_validator("seniority")
    @classmethod
    def _normalize_seniority(cls, value: str | None) -> str | None:
        return normalize_level(value) or None

    @field_validator("domains", "platforms", "tools", "processes", mode="before")
    @classmethod
    def _clean_list(cls, value: Any) -> List[str]:
        return _clean_strings(value)


class JDRequirementUnit(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str
    category: Literal["skill", "soft_skill", "experience", "education", "responsibility", "certification"] | None = None
    importance: Literal["required", "preferred", "responsibility"] | None = None
    source_section: str | None = None
    evidence: str = ""
    confidence: Literal["high", "medium", "low"] = "medium"
    alternative_group: str | None = None

    @field_validator("category", mode="before")
    @classmethod
    def _normalize_category(cls, value: Any) -> str | None:
        text = _slug_text(value)
        aliases = {
            "required_skill": "skill",
            "required skill": "skill",
            "preferred_skill": "skill",
            "preferred skill": "skill",
            "technical_skill": "skill",
            "technical skill": "skill",
            "hard_skill": "skill",
            "hard skill": "skill",
            "softskill": "soft_skill",
            "soft_skill": "soft_skill",
            "soft skill": "soft_skill",
            "cert": "certification",
            "certificate": "certification",
        }
        return aliases.get(text, text or None)

    @field_validator("name", "evidence", mode="before")
    @classmethod
    def _clean_required_text_fields(cls, value: Any) -> str:
        return _clean_text(value)

    @field_validator("source_section", "alternative_group", mode="before")
    @classmethod
    def _clean_optional_text_fields(cls, value: Any) -> str | None:
        text = _clean_text(value)
        return text or None


class JDConfidence(BaseModel):
    model_config = ConfigDict(extra="ignore")

    job_title: Literal["high", "medium", "low"] | None = None
    metadata_filter: Literal["high", "medium", "low"] | None = None
    required_skills: Literal["high", "medium", "low"] | None = None
    preferred_skills: Literal["high", "medium", "low"] | None = None
    responsibilities: Literal["high", "medium", "low"] | None = None
    soft_skills: Literal["high", "medium", "low"] | None = None
    work_context: Literal["high", "medium", "low"] | None = None


class JDSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    source_type: Literal["jd"] = "jd"
    job_title: str | None = None
    metadata_filter: JDMetadataFilter = Field(default_factory=JDMetadataFilter)
    scoring_config: JDScoringConfig = Field(default_factory=JDScoringConfig)
    required_skills: List[JDSkill] = Field(default_factory=list)
    preferred_skills: List[JDSkill] = Field(default_factory=list)
    competencies: List[JDCompetency] = Field(default_factory=list)
    responsibilities: List[str] = Field(default_factory=list)
    soft_skills: List[JDSoftSkill] = Field(default_factory=list)
    experience: JDExperience = Field(default_factory=JDExperience)
    education: JDEducation = Field(default_factory=JDEducation)
    work_context: JDWorkContext = Field(default_factory=JDWorkContext)
    requirement_units: List[JDRequirementUnit] = Field(default_factory=list)
    recruiter_expectations: List[str] = Field(default_factory=list)
    confidence: JDConfidence = Field(default_factory=JDConfidence, alias="_confidence")
    filter: JDFilterModel = Field(default_factory=JDFilterModel)

    @field_validator("job_title", mode="before")
    @classmethod
    def _clean_job_title(cls, value: Any) -> str | None:
        text = _clean_text(value)
        return text or None

    @field_validator("responsibilities", "recruiter_expectations", mode="before")
    @classmethod
    def _clean_string_list(cls, value: Any) -> List[str]:
        return _clean_strings(value)

    @model_validator(mode="after")
    def _attach_filter(self) -> "JDSchema":
        for row in self.required_skills:
            row.importance = "required"
        for row in self.preferred_skills:
            row.importance = "preferred"
        target_level = self.work_context.seniority or self.metadata_filter.target_level
        self.filter = JDFilterModel(
            required_skills=[row.name for row in self.required_skills],
            experience_min_months=self.experience.min_months or None,
            target_level=target_level,
        )
        return self

    def to_storage_dict(self) -> Dict:
        return self.model_dump(by_alias=True)


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _strip_accents(value: str) -> str:
    import unicodedata

    normalized = unicodedata.normalize("NFKD", str(value or ""))
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _slug_text(value: str) -> str:
    text = _strip_accents(value).lower()
    text = re.sub(r"[_/&-]+", " ", text)
    text = re.sub(r"[^a-z0-9+#.\s]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_level(value: str | None) -> str:
    key = _slug_text(value or "")
    return LEVEL_ALIASES.get(key, "")


def infer_target_level(title: str = "", text: str = "", seniority: str = "") -> str:
    explicit = normalize_level(seniority)
    if explicit:
        return explicit
    haystack = _slug_text(f"{title} {text}")
    for raw, normalized in LEVEL_ALIASES.items():
        if re.search(rf"\b{re.escape(raw)}\b", haystack):
            return normalized
    return ""


def get_mongo_client():
    global _client
    if _client is None:
        _client = MongoClient(os.getenv("MONGO_URI"), tlsCAFile=certifi.where(), **MONGO_CLIENT_OPTS)
    return _client


def get_jd_store():
    return get_mongo_client()[JD_DB_NAME][JD_COLLECTION_NAME]


def count_indexed_jds() -> int:
    return get_jd_store().count_documents({})


def _section_name(value: str) -> str:
    text = _slug_text(value)
    text = re.sub(r"^\d+(?:\.\d+)*\s*", "", text)
    return SECTION_ALIASES.get(text, text.replace(" ", "_") or "requirements")


def normalize_jd_text(raw_text: str) -> str:
    text = str(raw_text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"<\s*br\s*/?\s*>", "\n", text, flags=re.I)
    text = re.sub(r"</\s*(p|div|li|h[1-6]|tr)\s*>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.translate(str.maketrans({"•": "-", "·": "-", "–": "-", "—": "-"}))

    lines = []
    for raw_line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", raw_line).strip()
        if not line:
            if lines and lines[-1]:
                lines.append("")
            continue
        heading = line.strip(" #:-*`")
        section = _section_name(heading)
        if section in CANONICAL_SECTIONS and len(heading) <= 80:
            lines.append(f"## {section}")
        else:
            lines.append(line)
    return "\n".join(lines).strip()


def _split_jd_text_by_headers(text: str) -> Dict[str, str]:
    sections: Dict[str, List[str]] = {}
    current = "requirements"

    for raw_line in normalize_jd_text(text).splitlines():
        line = raw_line.strip()
        if not line:
            continue

        inline_match = re.match(r"^(?:#{1,6}\s*)?([^:]{2,80})\s*:\s*(.+)$", line)
        if inline_match:
            section = _section_name(inline_match.group(1))
            if section in CANONICAL_SECTIONS:
                current = section
                sections.setdefault(current, []).append(inline_match.group(2).strip())
                continue

        header_match = re.match(r"^#{1,6}\s*(.+)$", line)
        if header_match:
            section = _section_name(header_match.group(1))
            if section in CANONICAL_SECTIONS:
                current = section
                sections.setdefault(current, [])
                continue

        sections.setdefault(current, []).append(line)

    return {section: "\n".join(lines).strip() for section, lines in sections.items() if "\n".join(lines).strip()}


def _extract_json_object(raw: str) -> Dict:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", str(raw or "").strip(), flags=re.I)
    match = re.search(r"\{.*\}", text, re.S)
    candidates = [match.group(0)] if match else []
    candidates.append(text)
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("LLM did not return a valid JD JSON object")


def _generate_answer_with_timeout(prompt: str, timeout_seconds: int) -> str:
    from concurrent.futures import ThreadPoolExecutor, TimeoutError

    try:
        from llm_provider import generate_answer
    except Exception as exc:
        raise RuntimeError("LLM provider is not available for JD parsing") from exc

    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(generate_answer, prompt)
    try:
        answer = str(future.result(timeout=timeout_seconds) or "")
        if not answer.strip():
            raise ValueError("LLM returned an empty JD parser response")
        return answer
    except TimeoutError:
        future.cancel()
        raise TimeoutError(f"JD parser timed out after {timeout_seconds} seconds")
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def _clean_strings(value: Any, limit: int = 80) -> List[str]:
    result = []
    seen = set()
    for item in _as_list(value):
        text = _clean_text(item)
        key = text.lower()
        if text and key not in seen:
            result.append(text)
            seen.add(key)
    return result[:limit]


def _normalize_jd_schema_model(data: Dict | JDSchema) -> JDSchema:
    if isinstance(data, JDSchema):
        return data
    if not isinstance(data, dict):
        raise ValueError("JD schema must be a dictionary")
    return JDSchema.model_validate(data)


def _normalize_jd_schema(data: Dict | JDSchema, *, fallback_text: str = "") -> Dict:
    return _normalize_jd_schema_model(data).to_storage_dict()


def extract_jd_schema(jd_text: str, *, already_normalized: bool = False) -> JDSchema:
    text = jd_text if already_normalized else normalize_jd_text(jd_text)
    if not text or not JD_LLM_PARSE_ENABLED:
        return _normalize_jd_schema_model({})

    try:
        prompt = get_jd_parser_prompt(text[:JD_LLM_PARSE_MAX_CHARS])
        parsed = _extract_json_object(_generate_answer_with_timeout(prompt, JD_LLM_PARSE_TIMEOUT_SECONDS))
        return _normalize_jd_schema_model(parsed)
    except Exception as exc:
        print(f"[WARN] JD schema extraction failed ({type(exc).__name__}: {exc}). Falling back to empty schema.")
        return _normalize_jd_schema_model({})


def _schema_skill_names(schema: JDSchema) -> List[str]:
    names = []
    for row in [*schema.required_skills, *schema.preferred_skills, *schema.competencies]:
        if row.name:
            names.append(row.name)
    return list(dict.fromkeys(names))


def _format_rows(rows: Any) -> List[str]:
    lines = []
    for row in rows:
        if row.name:
            lines.append(f"- {row.name}")
    return lines


def _schema_to_section_text(schema: JDSchema, section: str) -> str:
    if section in {"requirements", "skills"}:
        lines = []
        required = _format_rows(schema.required_skills)
        preferred = _format_rows(schema.preferred_skills)
        competencies = _format_rows(schema.competencies)
        if required:
            lines.extend(["Required skills:", *required])
        if preferred:
            lines.extend(["Preferred skills:", *preferred])
        if competencies:
            lines.extend(["Competencies:", *competencies])
        return "\n".join(lines).strip()

    if section == "responsibilities":
        return "\n".join(f"- {item}" for item in schema.responsibilities).strip()

    if section == "soft_skills":
        return "\n".join(_format_rows(schema.soft_skills)).strip()

    if section == "experience":
        lines = []
        if schema.experience.min_months:
            lines.append(f"Minimum experience months: {schema.experience.min_months}")
        if schema.experience.max_months is not None:
            lines.append(f"Maximum experience months: {schema.experience.max_months}")
        if schema.experience.evidence:
            lines.append(f"Evidence: {schema.experience.evidence}")
        return "\n".join(lines).strip()

    if section == "education":
        lines = []
        if schema.education.level:
            lines.append(f"Minimum education: {schema.education.level}")
        if schema.education.major:
            lines.append(f"Major: {schema.education.major}")
        if schema.education.evidence:
            lines.append(f"Evidence: {schema.education.evidence}")
        return "\n".join(lines).strip()

    return ""


def _schema_sections(schema: JDSchema, source_text: str) -> Dict[str, str]:
    sections = _split_jd_text_by_headers(source_text)
    for section in ("requirements", "skills", "responsibilities", "experience", "education", "soft_skills"):
        if sections.get(section):
            continue
        text = _schema_to_section_text(schema, section)
        if text:
            sections[section] = text
    return sections


def _embedding_text(title: str, section: str, content: str, schema: JDSchema) -> str:
    parts = [f"[JD: {title}]", f"[SECTION: {section}]", content.strip()]
    if section in {"requirements", "skills", "preferred_skills"}:
        required = [r.name for r in schema.required_skills if r.name]
        preferred = [r.name for r in schema.preferred_skills if r.name]
        if required:
            parts.append("[REQUIRED_SKILLS]\n" + "\n".join(f"- {n}" for n in required))
        if preferred and section == "preferred_skills":
            parts.append("[PREFERRED_SKILLS]\n" + "\n".join(f"- {n}" for n in preferred))
        elif preferred and section in {"requirements", "skills"}:
            # inject tất cả skill (required + preferred) vào requirements/skills chunk
            all_names = list(dict.fromkeys(required + preferred))
            if all_names:
                parts.append("[JD_REQUIRED_SKILLS]\n" + "\n".join(f"- {n}" for n in all_names))
    return "\n\n".join(part for part in parts if part)


def _set_schema_context(schema: JDSchema, *, title: str = "", target_level: str = "") -> JDSchema:
    if title and not schema.job_title:
        schema.job_title = title
    if target_level:
        schema.work_context.seniority = target_level
        schema.metadata_filter.target_level = target_level
    schema.filter = JDFilterModel(
        required_skills=[row.name for row in schema.required_skills],
        experience_min_months=schema.experience.min_months or None,
        target_level=schema.work_context.seniority or schema.metadata_filter.target_level,
    )
    return schema


def jd_to_section_chunks(jd: Dict) -> List[Dict]:
    jd_id = str(jd.get("id") or jd.get("jd_id") or jd.get("title") or "jd").strip()
    title = str(jd.get("title") or jd_id).strip()
    explicit_target_level = normalize_level(jd.get("target_level") or jd.get("level"))

    section_label_map: Dict[str, str] = {}  # canonical → raw label
    if isinstance(jd.get("sections"), dict):
        parts = []
        for name, text in jd["sections"].items():
            if str(text).strip():
                canonical = _section_name(name)
                section_label_map.setdefault(canonical, name)  # giữ label đầu tiên gặp
                parts.append(f"{canonical}:\n{text}")
        raw_text = "\n\n".join(parts)
    else:
        raw_text = str(jd.get("text") or jd.get("description") or jd.get("content") or "").strip()

    if title and raw_text and not re.search(r"^\s*(?:job\s+title|title|position|role)\s*:", raw_text, re.I | re.M):
        raw_text = f"Job Title: {title}\n\n{raw_text}"
    raw_text = normalize_jd_text(raw_text)

    schema = extract_jd_schema(raw_text, already_normalized=True)
    target_level = explicit_target_level or schema.work_context.seniority or schema.metadata_filter.target_level
    _set_schema_context(schema, title=title, target_level=target_level or "")
    schema_dict = schema.to_storage_dict()

    chunks = []
    for index, (section, content) in enumerate(_schema_sections(schema, raw_text).items()):
        chunk_id = f"{jd_id}_{index:02d}_{section}"
        skills = _schema_skill_names(schema) if section in {"requirements", "skills"} else []
        skill_text = "\n".join(f"- {skill}" for skill in skills)
        embedding_text = _embedding_text(title, section, content, schema)
        chunks.append(
            {
                "chunk_id": chunk_id,
                "jd_id": jd_id,
                "source": jd_id,
                "title": title,
                "section": section,
                "section_original": section_label_map.get(section, section),
                "chunk_index": index,
                "content": content,
                "embedding_text": embedding_text,
                "content_hash": hashlib.sha256(embedding_text.encode()).hexdigest()[:16],
                "jd_schema": schema_dict,
                "target_level": target_level,
                "llm_skills": skills,
                "skill_text": skill_text,
                "filter": schema.filter.model_dump(),
            }
        )
    return chunks


def _infer_jd_title(text: str, fallback: str = "Pasted Job Description") -> str:
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip(" #:-\t")
        if line and _section_name(line) not in CANONICAL_SECTIONS:
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

    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    resolved_title = (title or "").strip() or _infer_jd_title(text)
    resolved_id = (jd_id or f"pasted_jd_{digest}").strip()
    resolved_level = normalize_level(target_level)
    chunk_count = ingest_jds(
        [{"id": resolved_id, "title": resolved_title, "text": text, "target_level": resolved_level}],
        prune_missing=False,
    )

    return {"jd_id": resolved_id, "title": resolved_title, "target_level": resolved_level, "chunk_count": chunk_count}


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
    return list(get_jd_store().find({"jd_id": jd_id}, _project_fields(include_score=False)).sort("chunk_index", 1))


def ingest_jds(jds: List[Dict] | None = None, *, prune_missing: bool = True) -> int:
    collection = get_jd_store()
    operations = []
    valid_chunk_ids = []

    for jd in jds or SAMPLE_JDS:
        for chunk in jd_to_section_chunks(jd):
            valid_chunk_ids.append(chunk["chunk_id"])
            operations.append(
                UpdateOne(
                    {"chunk_id": chunk["chunk_id"]},
                    {"$set": {**chunk, "embedding": get_embedding(chunk["embedding_text"]), "updated_at": datetime.now(timezone.utc)}},
                    upsert=True,
                )
            )

    if operations:
        collection.bulk_write(operations, ordered=False)
    if prune_missing and valid_chunk_ids:
        collection.delete_many({"chunk_id": {"$nin": valid_chunk_ids}})
    return len(operations)


def refresh_jd_embedding_texts(jd_id: str | None = None) -> int:
    collection = get_jd_store()
    query = {"jd_id": jd_id} if jd_id else {}
    operations = []

    for doc in collection.find(query, _project_fields(include_score=False)):
        chunk_id = doc.get("chunk_id")
        if not chunk_id:
            continue
        title = str(doc.get("title") or doc.get("jd_id") or "JD").strip()
        section = str(doc.get("section") or "requirements").strip()
        content = str(doc.get("content") or "").strip()
        schema = _normalize_jd_schema_model(doc.get("jd_schema") if isinstance(doc.get("jd_schema"), dict) else {})
        target_level = normalize_level(doc.get("target_level")) or schema.work_context.seniority or schema.metadata_filter.target_level
        _set_schema_context(schema, title=title, target_level=target_level or "")
        skills = _schema_skill_names(schema) if section in {"requirements", "skills"} else []
        embedding_text = _embedding_text(title, section, content, schema)
        schema_dict = schema.to_storage_dict()
        operations.append(
            UpdateOne(
                {"chunk_id": chunk_id},
                {
                    "$set": {
                        "embedding_text": embedding_text,
                        "embedding": get_embedding(embedding_text),
                        "jd_schema": schema_dict,
                        "target_level": target_level,
                        "llm_skills": skills,
                        "skill_text": "\n".join(f"- {skill}" for skill in skills),
                        "filter": schema.filter.model_dump(),
                        "updated_at": datetime.now(timezone.utc),
                    }
                },
            )
        )

    if operations:
        collection.bulk_write(operations, ordered=False)
    return len(operations)
