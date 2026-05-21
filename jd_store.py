import certifi
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from typing import Dict, List

from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne

from bedrock_utils import get_embedding
from jd_parser_prompt import get_jd_parser_prompt

load_dotenv()

MONGO_CLIENT_OPTS = {"maxPoolSize": 5, "minPoolSize": 1, "maxIdleTimeMS": 45000, "retryWrites": False}

JD_COLLECTION_NAME = "job_descriptions"
VECTOR_INDEX_NAME = "jd_vector_index"
JD_DB_NAME = "aws_rag_db"

CANONICAL_SECTIONS = {"requirements", "skills", "experience", "education", "soft_skills", "benefits"}

JD_SECTION_ALIASES = {
    **dict.fromkeys(["requirements", "requirement", "responsibilities", "responsibility", "job description", "description", "summary", "objective", "yeu cau", "mo ta cong viec", "trach nhiem"], "requirements"),
    **dict.fromkeys(["skills", "skill", "technical skills", "technologies", "tools", "ky nang", "ki nang", "cong nghe", "cong cu"], "skills"),
    **dict.fromkeys(["experience", "work experience", "qualification", "qualifications", "kinh nghiem", "kinh nghiem lam viec"], "experience"),
    **dict.fromkeys(["education", "degree", "hoc van", "bang cap"], "education"),
    **dict.fromkeys(["soft skills", "soft skill", "personal skills", "ky nang mem", "ki nang mem"], "soft_skills"),
    **dict.fromkeys(["benefits", "benefit"], "benefits"),
}

SAMPLE_JDS = [
    {"id": "jd_001", "title": "Tester Intern / QA Intern", "sections": {"requirements": "Manual testing, test case design, bug reporting, API testing.", "experience": "No professional experience required; real project or internship is preferred.", "skills": "Jira, Postman, basic Selenium, technical document reading.", "soft_skills": "Careful, detail-oriented, basic English reading."}},
    {"id": "jd_002", "title": "Backend Developer Intern", "sections": {"requirements": "REST API design, database modeling, server-side logic.", "experience": "Personal project is preferred; professional experience is not required.", "skills": "Python or Node.js, SQL, Git, HTTP and JSON understanding.", "soft_skills": "Teamwork, self-learning, good communication."}},
    {"id": "jd_003", "title": "Frontend Developer Intern", "sections": {"requirements": "Build responsive web UI and integrate APIs.", "experience": "Portfolio or real project is preferred; professional experience is not required.", "skills": "HTML, CSS, JavaScript, React or Vue.", "soft_skills": "Creative, UI/UX attention, good communication."}},
    {"id": "jd_004", "title": "Data Analyst Intern", "sections": {"requirements": "Analyze data, build reports and dashboards.", "experience": "Data analysis project is preferred; professional experience is not required.", "skills": "Excel, SQL, Python with Pandas and Matplotlib, Power BI or Tableau.", "soft_skills": "Logical thinking, accuracy, clear result presentation."}},
]

# DEPRECATED: JD_SCHEMA_PROMPT is replaced by get_jd_parser_prompt() from jd_parser_prompt.py
# That function provides:
# - System prompt with clear rules (required vs preferred skills, confidence levels)
# - Few-shot examples (3 cases: short JD, long prose, mixed Viet-Anh)
# - Automatic handling of all input formats
# See jd_parser_prompt.py for full prompt structure

_client = None


def get_mongo_client():
    global _client
    if _client is None:
        _client = MongoClient(os.getenv("MONGO_URI"), tlsCAFile=certifi.where(), **MONGO_CLIENT_OPTS)
    return _client


def get_jd_store():
    return get_mongo_client()[JD_DB_NAME][JD_COLLECTION_NAME]


def count_indexed_jds() -> int:
    try:
        return get_jd_store().count_documents({})
    except Exception:
        return 0


def _clean_text(value) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _as_list(value) -> List:
    return value if isinstance(value, list) else []


def _to_int(value, default: int = 0) -> int:
    if isinstance(value, (int, float)):
        return max(0, int(round(value)))
    match = re.search(r"\d+(?:\.\d+)?", str(value or ""))
    return max(0, int(round(float(match.group(0))))) if match else default


def _strip_accents(value: str) -> str:
    import unicodedata

    normalized = unicodedata.normalize("NFKD", str(value or ""))
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _normalize_jd_section(section_name: str) -> str:
    key = _strip_accents(section_name).lower()
    key = re.sub(r"[^\w\s_&/-]+", " ", key, flags=re.UNICODE)
    key = re.sub(r"[_&/-]+", " ", key)
    key = re.sub(r"\s+", " ", key).strip()
    return JD_SECTION_ALIASES.get(key, key.replace(" ", "_") or "requirements")


def _split_jd_text_by_headers(text: str) -> Dict[str, str]:
    sections: Dict[str, List[str]] = {}
    current = "requirements"
    saw_header = False

    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        inline_match = re.match(r"^([^:]{2,60})\s*:\s*(.+)$", line, re.UNICODE)
        if inline_match:
            candidate = _normalize_jd_section(inline_match.group(1))
            if candidate in CANONICAL_SECTIONS:
                current = candidate
                saw_header = True
                sections.setdefault(current, []).append(inline_match.group(2).strip())
                continue

        header_match = re.match(r"^(?:#{1,4}\s*)?([^:]{2,60})\s*:?\s*$", line, re.UNICODE)
        if header_match:
            candidate = _normalize_jd_section(header_match.group(1))
            if candidate in CANONICAL_SECTIONS:
                current = candidate
                saw_header = True
                sections.setdefault(current, [])
                continue

        sections.setdefault(current, []).append(line)

    if not saw_header:
        stripped = str(text or "").strip()
        return {"requirements": stripped} if stripped else {}

    return {
        section: "\n".join(lines).strip()
        for section, lines in sections.items()
        if "\n".join(lines).strip()
    }


def _skill_key(value: str) -> str:
    return re.sub(r"[^a-z0-9+#]+", "", str(value or "").lower())


def _dedupe_skills(skills: List[str]) -> List[str]:
    result = []
    seen = set()
    for skill in skills:
        value = re.sub(r"\s+", " ", str(skill or "")).strip(" -,*.;:")
        key = _skill_key(value)
        if value and key and key not in seen:
            result.append(value)
            seen.add(key)
    return result


def _extract_json_object(raw: str) -> Dict:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", str(raw or "").strip(), flags=re.IGNORECASE)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    for candidate in [match.group(0) if match else "", text]:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def _clean_item_rows(value, allowed_keys: List[str], limit: int = 60) -> List[Dict]:
    rows = []
    seen = set()
    for item in _as_list(value):
        if isinstance(item, str):
            item = {"name": item}
        if not isinstance(item, dict):
            continue
        name = _clean_text(item.get("name"))
        key = _skill_key(name)
        if not key or key in seen:
            continue
        row = {"name": name}
        seen.add(key)
        for allowed_key in allowed_keys:
            if allowed_key == "name":
                continue
            cell = item.get(allowed_key)
            if isinstance(cell, list):
                cell = [_clean_text(x) for x in cell if _clean_text(x)]
            elif cell is not None:
                cell = _clean_text(cell)
            if cell not in (None, "", []):
                row[allowed_key] = cell
        rows.append(row)
    return rows[:limit]


def _clean_expectations(value) -> List[str]:
    expectations = []
    for item in _as_list(value):
        if isinstance(item, dict):
            item = item.get("name") or item.get("expectation") or item.get("evidence") or item.get("description")
        text = _clean_text(item)
        if text:
            expectations.append(text)
    return list(dict.fromkeys(expectations))[:30]


DEFAULT_SCORING_CONFIG = {
    "weights": {
        "required_skills": 0.45,
        "preferred_skills": 0.10,
        "competencies": 0.10,
        "experience": 0.15,
        "education": 0.10,
        "quality": 0.05,
        "context": 0.05,
    },
    "required_skill_min_match_rate": 0.60,
    "hard_filters": {
        "metadata": True,
        "required_skills": True,
        "experience": False,
        "education": False,
    },
    "retrieval_weights": {
        "dense": 0.55,
        "lexical": 0.30,
        "schema": 0.15,
        "cross_encoder": 0.0,
        "llm_judge": 0.0,
    },
}


def _normalize_weight_map(value, defaults: Dict[str, float]) -> Dict[str, float]:
    raw = value if isinstance(value, dict) else {}
    weights = {}
    for key, default in defaults.items():
        try:
            weights[key] = max(0.0, float(raw.get(key, default)))
        except Exception:
            weights[key] = default
    total = sum(weights.values())
    if total <= 0:
        return defaults.copy()
    return {key: round(val / total, 4) for key, val in weights.items()}


def _normalize_scoring_config(value) -> Dict:
    raw = value if isinstance(value, dict) else {}
    hard_filters = raw.get("hard_filters") if isinstance(raw.get("hard_filters"), dict) else {}
    try:
        required_rate = float(raw.get("required_skill_min_match_rate", DEFAULT_SCORING_CONFIG["required_skill_min_match_rate"]) or 0)
    except Exception:
        required_rate = DEFAULT_SCORING_CONFIG["required_skill_min_match_rate"]
    config = {
        "weights": _normalize_weight_map(raw.get("weights"), DEFAULT_SCORING_CONFIG["weights"]),
        "required_skill_min_match_rate": max(0.0, min(1.0, required_rate)),
        "hard_filters": {
            key: bool(hard_filters.get(key, default))
            for key, default in DEFAULT_SCORING_CONFIG["hard_filters"].items()
        },
        "retrieval_weights": _normalize_weight_map(raw.get("retrieval_weights"), DEFAULT_SCORING_CONFIG["retrieval_weights"]),
    }
    return config


def _infer_role(text: str) -> str:
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip(" #:-\t")
        if not line or line.startswith("[") or line.startswith("-"):
            continue
        if _normalize_jd_section(line) not in CANONICAL_SECTIONS and len(line) <= 100:
            return line
    return ""


def _infer_experience(text: str) -> Dict:
    normalized = _strip_accents(text).lower()
    match = re.search(r"(\d+(?:\.\d+)?)\s*\+?\s*(years?|yrs?|year|nam|months?|mos?|thang)", normalized)
    if not match:
        return {"min_months": 0, "required": False, "evidence": ""}
    number = float(match.group(1))
    unit = match.group(2)
    months = round(number * 12) if unit.startswith(("year", "yr", "nam")) else round(number)
    line = next((ln.strip(" -") for ln in str(text).splitlines() if match.group(0) in _strip_accents(ln).lower()), "")
    return {"min_months": months, "required": True, "evidence": line}


def _infer_education(text: str) -> Dict:
    normalized = _strip_accents(text).lower()
    level = ""
    if re.search(r"\b(phd|doctor|doctorate|tien si)\b", normalized):
        level = "phd"
    elif re.search(r"\b(master|msc|ma|thac si)\b", normalized):
        level = "master"
    elif re.search(r"\b(bachelor|ba|bs|bsc|degree|university|college|cu nhan|dai hoc)\b", normalized):
        level = "bachelor"
    elif re.search(r"\b(high school|secondary|thpt)\b", normalized):
        level = "high_school"
    if not level:
        return {"level": "", "major": "", "required": False}
    line = next(
        (
            ln.strip(" -")
            for ln in str(text).splitlines()
            if re.search(r"degree|bachelor|master|phd|university|college", _strip_accents(ln), re.I)
        ),
        "",
    )
    major = ""
    major_match = re.search(r"\bin\s+(.+?)(?:\.|$)", line, re.I)
    if major_match:
        major = major_match.group(1).strip()
    return {"level": level, "major": major, "required": True}


def _normalize_jd_schema(data: Dict, *, fallback_text: str = "") -> Dict:
    context = data.get("work_context") if isinstance(data.get("work_context"), dict) else {}
    education = data.get("education") if isinstance(data.get("education"), dict) else {}
    experience = data.get("experience") if isinstance(data.get("experience"), dict) else {}
    
    # Preserve confidence level in skills
    required = _clean_item_rows(data.get("required_skills"), ["name", "type", "confidence", "importance", "evidence"])
    preferred = _clean_item_rows(data.get("preferred_skills"), ["name", "type", "confidence", "importance", "evidence"])
    competencies = _clean_item_rows(data.get("competencies"), ["name", "category", "confidence", "importance", "evidence"])
    
    fallback_education = _infer_education(fallback_text)
    fallback_experience = _infer_experience(fallback_text)
    role = _clean_text(context.get("role")) or _infer_role(fallback_text)
    
    # Extract overall confidence levels from the LLM response
    overall_confidence = data.get("_confidence", {})
    if not isinstance(overall_confidence, dict):
        overall_confidence = {}

    schema = {
        "source_type": "jd",
        "job_title": _clean_text(data.get("job_title")) or role,
        "required_skills": required[:60],
        "preferred_skills": preferred,
        "competencies": competencies,
        "responsibilities": [_clean_text(x) for x in _as_list(data.get("responsibilities")) if _clean_text(x)][:30],
        "recruiter_expectations": _clean_expectations(data.get("recruiter_expectations")),
        "soft_skills": _clean_item_rows(data.get("soft_skills"), ["name", "confidence", "evidence"]),
        "work_context": {
            "role": role,
            "seniority": _clean_text(context.get("seniority")).lower(),
            "team_size": _clean_text(context.get("team_size")),
            "report_to": _clean_text(context.get("report_to")),
            "industry": _clean_text(context.get("industry")),
            "company_type": _clean_text(context.get("company_type")),
            "domains": [_clean_text(x) for x in _as_list(context.get("domains")) if _clean_text(x)][:20],
            "platforms": [_clean_text(x) for x in _as_list(context.get("platforms")) if _clean_text(x)][:20],
            "tools": ([_clean_text(x) for x in _as_list(context.get("tools")) if _clean_text(x)] or [r["name"] for r in required + preferred])[:30],
            "processes": ([_clean_text(x) for x in _as_list(context.get("processes")) if _clean_text(x)] or [r["name"] for r in competencies])[:30],
        },
        "metadata_filter": {
            "education_min": _clean_text(data.get("metadata_filter", {}).get("education_min")).lower() or (fallback_education.get("level") or None),
            "exp_years_min": _to_int(data.get("metadata_filter", {}).get("exp_years_min")) or (fallback_experience.get("min_months", 0) // 12 if fallback_experience.get("min_months", 0) else None),
            "job_type": _clean_text(data.get("metadata_filter", {}).get("job_type")).lower() or None,
            "location": _clean_text(data.get("metadata_filter", {}).get("location")) or None,
        },
        "scoring_config": _normalize_scoring_config(data.get("scoring_config")),
        "education": {
            "level": _clean_text(education.get("level")).lower() or fallback_education["level"],
            "major": _clean_text(education.get("major")) or fallback_education["major"],
            "required": bool(education.get("required", False)) or fallback_education["required"],
        },
        "experience": {
            "min_months": _to_int(experience.get("min_months"), 0) or fallback_experience["min_months"],
            "required": bool(experience.get("required", False)) or fallback_experience["required"],
            "evidence": _clean_text(experience.get("evidence")) or fallback_experience["evidence"],
        },
        # Include confidence levels from LLM parsing
        "_confidence": {
            "job_title": overall_confidence.get("job_title", "medium"),
            "metadata_filter": overall_confidence.get("metadata_filter", "medium"),
            "required_skills": overall_confidence.get("required_skills", "high"),
            "preferred_skills": overall_confidence.get("preferred_skills", "medium"),
            "responsibilities": overall_confidence.get("responsibilities", "medium"),
            "soft_skills": overall_confidence.get("soft_skills", "medium"),
            "work_context": overall_confidence.get("work_context", "medium"),
        }
    }
    
    # Clean up None values in metadata_filter
    schema["metadata_filter"] = {k: v for k, v in schema["metadata_filter"].items() if v is not None}

    return schema


def extract_jd_schema(jd_text: str) -> Dict:
    text = str(jd_text or "").strip()
    if not text:
        return _normalize_jd_schema({}, fallback_text="")
    try:
        from local_llm import generate_answer

        # Use optimized prompt with few-shot examples and confidence levels
        prompt = get_jd_parser_prompt(text[:16000])
        raw = generate_answer(prompt)
    except Exception:
        raw = ""
    return _normalize_jd_schema(_extract_json_object(raw), fallback_text=text)


def _jd_schema_names(schema: Dict) -> List[str]:
    names = []
    for key in ("required_skills", "preferred_skills", "competencies"):
        names.extend(row.get("name", "") for row in _as_list(schema.get(key)) if isinstance(row, dict))
    return _dedupe_skills(names)


def _format_schema_for_embedding(schema: Dict, *, section: str = "") -> str:
    parts = []
    section = str(section or "").lower()

    if section in {"requirements", "skills", "experience"}:
        for label, key in (
            ("JD_REQUIRED_SKILLS", "required_skills"),
            ("JD_PREFERRED_SKILLS", "preferred_skills"),
            ("JD_COMPETENCIES", "competencies"),
        ):
            names = [row["name"] for row in _as_list(schema.get(key)) if isinstance(row, dict) and row.get("name")]
            if names:
                parts.append(f"[{label}]\n" + "\n".join(f"- {name}" for name in names))

        expectations = [_clean_text(x) for x in _as_list(schema.get("recruiter_expectations")) if _clean_text(x)]
        if expectations:
            parts.append("[RECRUITER_EXPECTATIONS]\n" + "\n".join(f"- {x}" for x in expectations))

    context = schema.get("work_context") if isinstance(schema.get("work_context"), dict) else {}
    context_lines = []
    for key in ("role", "seniority"):
        if context.get(key):
            context_lines.append(f"{key}: {context[key]}")
    for key in ("domains", "platforms", "tools", "processes"):
        values = [_clean_text(x) for x in _as_list(context.get(key)) if _clean_text(x)]
        if values:
            context_lines.append(f"{key}: {', '.join(values)}")
    if context_lines:
        parts.append("[WORK_CONTEXT]\n" + "\n".join(f"- {line}" for line in context_lines))

    if section == "education":
        education = schema.get("education") if isinstance(schema.get("education"), dict) else {}
        edu_lines = [
            f"level: {education.get('level', '')}",
            f"major: {education.get('major', '')}",
            f"required: {education.get('required', False)}",
        ]
        parts.append("[JD_EDUCATION]\n" + "\n".join(f"- {line}" for line in edu_lines if not line.endswith(": ")))

    if section == "experience":
        experience = schema.get("experience") if isinstance(schema.get("experience"), dict) else {}
        exp_lines = [
            f"min_months: {experience.get('min_months', 0)}",
            f"required: {experience.get('required', False)}",
            f"evidence: {experience.get('evidence', '')}",
        ]
        parts.append("[JD_EXPERIENCE]\n" + "\n".join(f"- {line}" for line in exp_lines if not line.endswith(": ")))

    return "\n\n".join(parts)


def _format_named_rows(rows, *, include_evidence: bool = True) -> List[str]:
    lines = []
    for row in _as_list(rows):
        if isinstance(row, str):
            text = _clean_text(row)
            if text:
                lines.append(f"- {text}")
            continue
        if not isinstance(row, dict):
            continue
        name = _clean_text(row.get("name"))
        if not name:
            continue
        details = []
        if row.get("type"):
            details.append(f"type={_clean_text(row.get('type'))}")
        if row.get("confidence"):
            details.append(f"confidence={_clean_text(row.get('confidence'))}")
        if include_evidence and row.get("evidence"):
            details.append(f"evidence={_clean_text(row.get('evidence'))}")
        lines.append(f"- {name}" + (f" ({'; '.join(details)})" if details else ""))
    return lines


def _schema_to_jd_sections(schema: Dict, *, fallback_text: str = "") -> Dict[str, str]:
    """Build storage/retrieval chunks from the normalized LLM JD schema."""
    if not isinstance(schema, dict) or schema.get("source_type") != "jd":
        return _split_jd_text_by_headers(fallback_text) if fallback_text.strip() else {}

    sections: Dict[str, str] = {}
    title = _clean_text(schema.get("job_title"))

    requirement_lines = []
    if title:
        requirement_lines.append(f"Job title: {title}")
    requirement_lines.extend(_format_named_rows(schema.get("required_skills")))
    responsibilities = [_clean_text(x) for x in _as_list(schema.get("responsibilities")) if _clean_text(x)]
    if responsibilities:
        requirement_lines.append("Responsibilities:")
        requirement_lines.extend(f"- {item}" for item in responsibilities)
    expectations = [_clean_text(x) for x in _as_list(schema.get("recruiter_expectations")) if _clean_text(x)]
    if expectations:
        requirement_lines.append("Recruiter expectations:")
        requirement_lines.extend(f"- {item}" for item in expectations)
    if requirement_lines:
        sections["requirements"] = "\n".join(requirement_lines)

    skill_lines = []
    required_skill_lines = _format_named_rows(schema.get("required_skills"))
    preferred_skill_lines = _format_named_rows(schema.get("preferred_skills"))
    competency_lines = _format_named_rows(schema.get("competencies"))
    if required_skill_lines:
        skill_lines.append("Required skills:")
        skill_lines.extend(required_skill_lines)
    if preferred_skill_lines:
        skill_lines.append("Preferred skills:")
        skill_lines.extend(preferred_skill_lines)
    if competency_lines:
        skill_lines.append("Competencies:")
        skill_lines.extend(competency_lines)
    if skill_lines:
        sections["skills"] = "\n".join(skill_lines)

    experience = schema.get("experience") if isinstance(schema.get("experience"), dict) else {}
    metadata = schema.get("metadata_filter") if isinstance(schema.get("metadata_filter"), dict) else {}
    context = schema.get("work_context") if isinstance(schema.get("work_context"), dict) else {}
    experience_lines = []
    min_months = _to_int(experience.get("min_months"), 0) or (_to_int(metadata.get("exp_years_min"), 0) * 12)
    if min_months:
        experience_lines.append(f"Minimum experience months: {min_months}")
    if experience.get("required"):
        experience_lines.append(f"Experience required: {bool(experience.get('required'))}")
    if experience.get("evidence"):
        experience_lines.append(f"Evidence: {_clean_text(experience.get('evidence'))}")
    if context.get("seniority"):
        experience_lines.append(f"Seniority: {_clean_text(context.get('seniority'))}")
    if experience_lines:
        sections["experience"] = "\n".join(experience_lines)

    education = schema.get("education") if isinstance(schema.get("education"), dict) else {}
    education_lines = []
    edu_level = _clean_text(education.get("level")) or _clean_text(metadata.get("education_min"))
    if edu_level:
        education_lines.append(f"Minimum education: {edu_level}")
    if education.get("major"):
        education_lines.append(f"Major: {_clean_text(education.get('major'))}")
    if education.get("required"):
        education_lines.append(f"Education required: {bool(education.get('required'))}")
    if education_lines:
        sections["education"] = "\n".join(education_lines)

    soft_skill_lines = _format_named_rows(schema.get("soft_skills"))
    if soft_skill_lines:
        sections["soft_skills"] = "\n".join(soft_skill_lines)

    context_lines = []
    for key in ("role", "seniority", "industry", "company_type", "job_type", "location"):
        value = context.get(key) if key not in {"job_type", "location"} else metadata.get(key)
        if value:
            context_lines.append(f"{key}: {_clean_text(value)}")
    for key in ("domains", "platforms", "tools", "processes"):
        values = [_clean_text(x) for x in _as_list(context.get(key)) if _clean_text(x)]
        if values:
            context_lines.append(f"{key}: {', '.join(values)}")
    if context_lines:
        sections.setdefault("requirements", "")
        sections["requirements"] = "\n".join([part for part in [sections["requirements"], "Work context:", *context_lines] if part])

    if not sections and fallback_text.strip():
        return _split_jd_text_by_headers(fallback_text)
    return sections


def _build_jd_skill_text(skills: List[str]) -> str:
    skills = _dedupe_skills(skills)
    return "[JD_REQUIRED_SKILLS]\n" + "\n".join(f"- {skill}" for skill in skills) if skills else ""


def jd_to_section_chunks(jd: Dict) -> List[Dict]:
    jd_id = str(jd.get("id") or jd.get("jd_id") or jd.get("title") or "jd").strip()
    title = str(jd.get("title") or jd_id).strip()

    if isinstance(jd.get("sections"), dict):
        raw_sections = {
            _normalize_jd_section(name): str(text).strip()
            for name, text in jd["sections"].items()
            if str(text).strip()
        }
        raw_text = "\n\n".join(f"[{section}]\n{text}" for section, text in raw_sections.items())
    else:
        raw_text = str(jd.get("text") or jd.get("description") or jd.get("content") or "").strip()

    jd_schema = extract_jd_schema(raw_text)
    sections = _schema_to_jd_sections(jd_schema, fallback_text=raw_text)
    schema_names = _jd_schema_names(jd_schema)

    chunks = []
    for index, (section, text) in enumerate(sections.items()):
        skills = schema_names if section in {"requirements", "skills", "experience"} else []
        skill_text = _build_jd_skill_text(skills)
        chunk_id = f"{jd_id}_{index:02d}_{section}"
        schema_text = _format_schema_for_embedding(jd_schema, section=section)
        embedding_text = "\n".join(
            part
            for part in [f"[JD: {title}]", f"[SECTION: {section}]", text, schema_text, skill_text]
            if part
        )
        chunks.append(
            {
                "chunk_id": chunk_id,
                "jd_id": jd_id,
                "source": jd_id,
                "title": title,
                "section": section,
                "section_original": section,
                "chunk_index": index,
                "content": text,
                "embedding_text": embedding_text,
                "jd_schema": jd_schema,
                "llm_skills": skills,
                "skill_text": skill_text,
            }
        )
    return chunks


def _infer_jd_title(text: str, fallback: str = "Pasted Job Description") -> str:
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip(" #:-\t")
        if line and _normalize_jd_section(line) not in CANONICAL_SECTIONS:
            return line[:80]
    return fallback


def ingest_jd_text(jd_text: str, *, title: str | None = None, jd_id: str | None = None) -> Dict:
    text = str(jd_text or "").strip()
    if not text:
        raise ValueError("JD text is empty")

    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    resolved_title = (title or "").strip() or _infer_jd_title(text)
    resolved_id = (jd_id or f"pasted_jd_{digest}").strip()
    chunk_count = ingest_jds([{"id": resolved_id, "title": resolved_title, "text": text}], prune_missing=False)

    return {"jd_id": resolved_id, "title": resolved_title, "chunk_count": chunk_count}


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
        "llm_skills": 1,
        "skill_text": 1,
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
                "numCandidates": min(max(k * 10, 50), 200),
                "limit": k,
            }
        },
        {"$project": _project_fields(include_score=True)},
    ]
    return list(get_jd_store().aggregate(pipeline))


def search_similar_jd_skills(query_embedding: list, k: int = 5) -> List[Dict]:
    pipeline = [
        {
            "$vectorSearch": {
                "index": VECTOR_INDEX_NAME,
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": min(max(k * 12, 60), 200),
                "limit": k * 4,
            }
        },
        {"$project": _project_fields(include_score=True)},
    ]
    hits = list(get_jd_store().aggregate(pipeline))
    preferred = [hit for hit in hits if str(hit.get("section") or "").strip().lower() in {"skills", "requirements", "experience"}]
    return (preferred or hits)[:k]


def list_indexed_jds() -> List[Dict]:
    pipeline = [
        {"$sort": {"chunk_index": 1}},
        {"$group": {"_id": "$jd_id", "title": {"$first": "$title"}, "chunk_count": {"$sum": 1}}},
        {"$sort": {"title": 1}},
        {"$project": {"_id": 0, "jd_id": "$_id", "title": 1, "chunk_count": 1}},
    ]
    return list(get_jd_store().aggregate(pipeline))


def get_jd_chunks(jd_id: str) -> List[Dict]:
    return list(
        get_jd_store()
        .find({"jd_id": jd_id}, _project_fields(include_score=False))
        .sort("chunk_index", 1)
    )


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
