import re
import unicodedata
import logging
import math
import os
from difflib import SequenceMatcher
from typing import Any, Dict, List

from bedrock_utils import get_embedding
from jd_store import _normalize_jd_schema, count_indexed_jds, get_jd_chunks

logger = logging.getLogger(__name__)

CROSS_ENCODER_ENABLED = os.getenv("CROSS_ENCODER_ENABLED", "true").strip().lower() in {
    "1",
    "true",
    "yes",
}
CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
CROSS_ENCODER_LOCAL_FILES_ONLY = os.getenv("CROSS_ENCODER_LOCAL_FILES_ONLY", "true").strip().lower() in {
    "1",
    "true",
    "yes",
}
CROSS_ENCODER_MAX_CHUNKS = max(1, int(os.getenv("CROSS_ENCODER_MAX_CHUNKS", "6") or 6))
CROSS_ENCODER_TOP_AVG = max(1, int(os.getenv("CROSS_ENCODER_TOP_AVG", "3") or 3))
CROSS_ENCODER_MAX_JD_CHARS = max(500, int(os.getenv("CROSS_ENCODER_MAX_JD_CHARS", "3500") or 3500))
CROSS_ENCODER_MAX_CHUNK_CHARS = max(300, int(os.getenv("CROSS_ENCODER_MAX_CHUNK_CHARS", "1200") or 1200))
CROSS_ENCODER_WEIGHT = max(0.0, min(1.0, float(os.getenv("CROSS_ENCODER_WEIGHT", "0.25") or 0.25)))
REQUIREMENT_EVIDENCE_ENABLED = os.getenv("REQUIREMENT_EVIDENCE_ENABLED", "true").strip().lower() in {
    "1",
    "true",
    "yes",
}
REQUIREMENT_EVIDENCE_THRESHOLD = max(
    0.0,
    min(1.0, float(os.getenv("REQUIREMENT_EVIDENCE_THRESHOLD", "0.55") or 0.55)),
)
REQUIREMENT_EVIDENCE_MAX_REQUIREMENTS = max(1, int(os.getenv("REQUIREMENT_EVIDENCE_MAX_REQUIREMENTS", "18") or 18))
REQUIREMENT_EVIDENCE_MAX_CHUNKS = max(1, int(os.getenv("REQUIREMENT_EVIDENCE_MAX_CHUNKS", "4") or 4))

_cross_encoder_model = None
_cross_encoder_load_attempted = False

SCORE_LIMITS = {
    "technical_score": 40,
    "experience_score": 30,
    "education_score": 20,
    "fit_score": 10,
}

DEFAULT_WEIGHTS = {
    "required_skills": 0.50,
    "preferred_skills": 0.15,
    "competencies": 0.10,
    "experience": 0.15,
    "education": 0.05,
    "quality": 0.03,
    "context": 0.02,
}

EDUCATION_RANK = {
    "": 0,
    "highschool": 1,
    "high_school": 1,
    "college": 2,
    "bachelorinprogress": 2,
    "bachelor_in_progress": 2,
    "bachelor": 3,
    "master": 4,
    "phd": 5,
}


def _empty_evaluation(summary: str = "Error") -> Dict:
    return {
        "score": 0,
        "technical_score": 0,
        "experience_score": 0,
        "education_score": 0,
        "fit_score": 0,
        "matched_skills": [],
        "missing_skills": [],
        "recommendation": "Error",
        "summary": summary,
    }


def _error_result(jd_id: str, jd_title: str, summary: str, *, method: str = "") -> List[Dict]:
    return [{
        "cv_source": "error",
        "candidate_name": "",
        "jd_title": jd_title,
        "jd_id": jd_id,
        "similarity_score": 0,
        "dense_score": 0,
        "bm25_score": 0,
        "cross_encoder_score": 0,
        "rerank_score": 0,
        "rerank_method": method,
        "retrieval_method": method,
        "section_scores": {},
        "evaluation": _empty_evaluation(summary),
        "cv_profile": {},
        "match_evidence": [],
    }]


def _strip_accents(value: str) -> str:
    text = str(value or "").translate(str.maketrans({"đ": "d", "Đ": "D"}))
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _norm_text(value: str) -> str:
    value = _strip_accents(value).lower()
    value = re.sub(r"[^a-z0-9.+#]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _skill_key(value: str) -> str:
    return re.sub(r"[^a-z0-9+#]+", "", _norm_text(value))


def _dedupe_keep_order(items: List[Any]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        text = re.sub(r"\s+", " ", str(item or "")).strip(" -,*.;:")
        key = _skill_key(text)
        if text and key and key not in seen:
            seen.add(key)
            result.append(text)
    return result


def _rows(schema: Dict, key: str) -> List[Dict]:
    value = schema.get(key) if isinstance(schema, dict) else []
    rows = []
    for item in value if isinstance(value, list) else []:
        if isinstance(item, str):
            item = {"name": item}
        if isinstance(item, dict) and str(item.get("name") or "").strip():
            rows.append(item)
    return rows


def _first_schema(chunks: List[Dict], key: str) -> Dict:
    for chunk in chunks or []:
        schema = chunk.get(key) if isinstance(chunk, dict) else None
        if isinstance(schema, dict) and schema:
            return schema
    return {}


def _normalize_weights(schema: Dict) -> Dict[str, float]:
    raw_config = schema.get("scoring_config") if isinstance(schema, dict) else {}
    raw_weights = raw_config.get("weights") if isinstance(raw_config, dict) else {}
    weights = {}
    for key, default in DEFAULT_WEIGHTS.items():
        try:
            weights[key] = max(0.0, float((raw_weights or {}).get(key, default)))
        except Exception:
            weights[key] = default
    total = sum(weights.values())
    return DEFAULT_WEIGHTS.copy() if total <= 0 else {key: val / total for key, val in weights.items()}


def _required_skill_threshold(schema: Dict) -> float:
    raw_config = schema.get("scoring_config") if isinstance(schema, dict) else {}
    try:
        value = float((raw_config or {}).get("required_skill_min_match_rate", 0.0))
    except Exception:
        value = 0.0
    return max(0.0, min(1.0, value))


def _hard_filters(schema: Dict) -> Dict[str, bool]:
    raw_config = schema.get("scoring_config") if isinstance(schema, dict) else {}
    raw_filters = raw_config.get("hard_filters") if isinstance(raw_config, dict) else {}
    return {
        "metadata": bool((raw_filters or {}).get("metadata", True)),
        "required_skills": bool((raw_filters or {}).get("required_skills", False)),
    }


def _row_name(row: Dict) -> str:
    return str(row.get("name") or "").strip()


def _row_weight(row: Dict) -> float:
    importance = str(row.get("importance") or "").lower()
    confidence = str(row.get("confidence") or "").lower()
    weight = 1.25 if importance in {"must", "required", "high"} else 1.0
    if confidence == "low":
        weight *= 0.75
    elif confidence == "high":
        weight *= 1.10
    return weight


def _token_set(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9+#]+", _norm_text(value)))


SKILL_TEXT_EVIDENCE = {
    "restapidesign": (
        r"\brest(?:ful)?\s+apis?\b",
        r"\brest\s+endpoints?\b",
        r"\bapi\s+(?:design|development|endpoints?)\b",
    ),
    "databasemodeling": (
        r"\bdatabase\s+(?:model(?:ing|s)?|design|layouts?|schemas?)\b",
        r"\b(?:relational|non relational)\s+(?:data\s+)?model(?:ing|s)?\b",
        r"\b(?:entity|table)\s+(?:relationships?|schemas?)\b",
    ),
    "serversidelogic": (
        r"\bserver\s+side\s+(?:logic|components?|services?|development)\b",
        r"\bbackend\s+(?:logic|components?|services?|development)\b",
    ),
    "teamwork": (
        r"\bteam(?:work| collaboration)\b",
        r"\bcollaborat(?:e|ed|ing|ion)\b",
        r"\bworked\s+(?:with|in)\s+(?:a\s+)?team\b",
    ),
    "selflearning": (
        r"\bself\s+(?:learning|study|taught)\b",
        r"\beager(?:ness)?\s+to\s+learn\b",
        r"\blearn(?:ed|ing)?\s+new\s+(?:technologies|tools|skills)\b",
    ),
    "communication": (
        r"\bcommunicat(?:e|ed|ing|ion)\b",
        r"\bclear\s+(?:presentation|reporting|documentation)\b",
    ),
    "goodcommunication": (
        r"\bcommunicat(?:e|ed|ing|ion)\b",
        r"\bclear\s+(?:presentation|reporting|documentation)\b",
    ),
    "manualtesting": (
        r"\bmanual\s+test(?:ing| cases?)\b",
    ),
    "manualtestingprocesses": (
        r"\bmanual\s+test(?:ing| cases?)\b",
    ),
    "functionaltesting": (
        r"\bfunctional\s+testing\b",
        r"\bfunctional\b(?:\s+[a-z0-9]+){0,5}\s+testing\b",
    ),
    "defectreporting": (
        r"\b(?:report(?:ed|ing)?|document(?:ed|ation)?|track(?:ed|ing)?)\b(?:\s+[a-z0-9]+){0,6}\s+defects?\b",
        r"\bdefects?\b(?:\s+[a-z0-9]+){0,6}\s+(?:report(?:ed|ing)?|document(?:ed|ation)?|track(?:ed|ing)?)\b",
    ),
    "bugtracking": (
        r"\b(?:track(?:ed|ing)?|reproduc(?:e|ed|ing))\b(?:\s+[a-z0-9]+){0,6}\s+(?:bugs?|defects?)\b",
        r"\b(?:bugs?|defects?)\b(?:\s+[a-z0-9]+){0,6}\s+track(?:ed|ing)?\b",
    ),
    "proactiveness": (
        r"\bproactiv(?:e|ely|eness)\b",
    ),
    "dataanalysis": (
        r"\bdata\s+(?:analysis|analytics|processing|preprocessing|visuali[sz]ation)\b",
        r"\banaly[sz](?:e|ed|ing)\s+data\b",
        r"\b(?:pandas|numpy)\b",
    ),
    "reporting": (
        r"\breports?\b",
        r"\breporting\b",
        r"\bdashboards?\b",
        r"\b(?:power\s*bi|tableau)\b",
    ),
    "dashboards": (
        r"\bdashboards?\b",
        r"\bdata\s+visuali[sz]ation\b",
        r"\b(?:power\s*bi|tableau)\b",
    ),
    "datavisualization": (
        r"\bdata\s+visuali[sz]ation\b",
        r"\b(?:matplotlib|seaborn|power\s*bi|tableau)\b",
    ),
}


def _skills_match(jd_skill: str, cv_skill: str) -> bool:
    left_key = _skill_key(jd_skill)
    right_key = _skill_key(cv_skill)
    if not left_key or not right_key:
        return False
    if left_key == right_key:
        return True
    if len(left_key) >= 4 and (left_key in right_key or right_key in left_key):
        return True
    left_tokens = _token_set(jd_skill)
    right_tokens = _token_set(cv_skill)
    if left_tokens and right_tokens and left_tokens <= right_tokens:
        return True
    return SequenceMatcher(None, left_key, right_key).ratio() >= 0.90


def _snippet_for_pattern(text: str, pattern: str) -> str:
    """Return a compact raw CV snippet that contains the normalized match."""
    for snippet in re.split(r"(?:\r?\n)+|(?<=[.!?])\s+", str(text or "")):
        if re.search(pattern, _norm_text(snippet), re.I):
            return re.sub(r"\s+", " ", snippet).strip()[:500]
    return ""


def _skill_text_match(jd_skill: str, cv_text: str) -> Dict[str, str] | None:
    """Find skill evidence in raw CV text when canonical skill names differ."""
    normalized_cv_text = _norm_text(cv_text)
    normalized_skill = _norm_text(jd_skill)
    if not normalized_cv_text or not normalized_skill:
        return None

    exact_pattern = rf"(?<![a-z0-9]){re.escape(normalized_skill)}(?![a-z0-9])"
    if re.search(exact_pattern, normalized_cv_text):
        return {
            "name": jd_skill,
            "evidence": _snippet_for_pattern(cv_text, exact_pattern) or jd_skill,
            "section": "raw_text",
        }

    for pattern in SKILL_TEXT_EVIDENCE.get(_skill_key(jd_skill), ()):
        if re.search(pattern, normalized_cv_text, re.I):
            return {
                "name": jd_skill,
                "evidence": _snippet_for_pattern(cv_text, pattern) or jd_skill,
                "section": "raw_text",
            }
    return None


def _match_rows(jd_rows: List[Dict], cv_rows: List[Dict], cv_text: str) -> tuple[float, List[str], List[str], Dict[str, Dict]]:
    if not jd_rows:
        return 1.0, [], [], {}

    matched = []
    missing = []
    evidence_by_skill = {}
    total_weight = 0.0
    matched_weight = 0.0

    for jd_row in jd_rows:
        name = _row_name(jd_row)
        if not name:
            continue
        total_weight += _row_weight(jd_row)
        match_row = next((row for row in cv_rows if _skills_match(name, _row_name(row))), None)
        if not match_row:
            match_row = _skill_text_match(name, cv_text)

        if match_row:
            matched.append(name)
            matched_weight += _row_weight(jd_row)
            evidence_by_skill[name] = match_row
        else:
            missing.append(name)

    return matched_weight / max(total_weight, 1.0), _dedupe_keep_order(matched), _dedupe_keep_order(missing), evidence_by_skill


def _education_rank(value: Any) -> int:
    return EDUCATION_RANK.get(_skill_key(str(value or "").replace("_", " ")), 0)


def _score_experience(jd_schema: Dict, cv_schema: Dict) -> tuple[float, int, int]:
    jd_exp = jd_schema.get("experience") if isinstance(jd_schema, dict) else {}
    required_months = int((jd_exp or {}).get("min_months") or 0)
    cv_months = int((cv_schema or {}).get("experience_months") or 0)
    if required_months <= 0:
        return 1.0, cv_months, required_months
    if cv_months >= required_months:
        return 1.0, cv_months, required_months
    if cv_months <= 0:
        return 0.25, cv_months, required_months
    return max(0.30, min(0.95, cv_months / required_months)), cv_months, required_months


def _score_education(jd_schema: Dict, cv_schema: Dict) -> tuple[float, str, str]:
    jd_edu = jd_schema.get("education") if isinstance(jd_schema, dict) else {}
    cv_edu = cv_schema.get("education") if isinstance(cv_schema, dict) else {}
    if not isinstance(jd_edu, dict) or not jd_edu.get("required"):
        return 1.0, "", str((cv_edu or {}).get("level") or "")

    required_level = str(jd_edu.get("level") or "")
    cv_level = str((cv_edu or {}).get("level") or "")
    if _education_rank(cv_level) >= _education_rank(required_level) > 0:
        return 1.0, required_level, cv_level
    return 0.35, required_level, cv_level


def _quality_score(cv_schema: Dict) -> float:
    scores = []
    for key in ("experience", "projects"):
        rows = cv_schema.get(key) if isinstance(cv_schema, dict) else []
        for row in rows if isinstance(rows, list) else []:
            if isinstance(row, dict) and isinstance(row.get("quality_score"), (int, float)):
                scores.append(float(row["quality_score"]) / 100)
    return max(scores) if scores else 0.60


def _context_score(jd_schema: Dict, cv_rows: List[Dict], cv_text: str) -> float:
    context = jd_schema.get("work_context") if isinstance(jd_schema, dict) else {}
    terms = []
    if isinstance(context, dict):
        for key in ("domains", "platforms", "tools", "processes"):
            values = context.get(key)
            terms.extend(values if isinstance(values, list) else [])
    terms = _dedupe_keep_order(terms)
    if not terms:
        return 1.0
    ratio, _, _, _ = _match_rows([{"name": term} for term in terms], cv_rows, cv_text)
    return ratio


def _recommendation(score: float) -> str:
    if score >= 80:
        return "Strong fit"
    if score >= 65:
        return "Good fit"
    if score >= 45:
        return "Consider"
    return "Not a fit"


def _cv_text(chunks: List[Dict]) -> str:
    return "\n".join(str(ch.get("text") or ch.get("embedding_text") or ch.get("content") or "") for ch in chunks)


def _cv_skill_rows(chunks: List[Dict], cv_schema: Dict) -> List[Dict]:
    rows = []
    rows.extend(_rows(cv_schema, "skills_must"))
    rows.extend(_rows(cv_schema, "skills_nice"))
    rows.extend(_rows(cv_schema, "required_skills"))
    rows.extend(_rows(cv_schema, "preferred_skills"))
    if rows:
        return rows

    fallback = []
    for chunk in chunks:
        info = chunk.get("extracted_info") if isinstance(chunk, dict) else {}
        if isinstance(info, dict):
            fallback.extend(info.get("technical_skills") or [])
            fallback.extend(info.get("chunk_skills") or [])
            fallback.extend(info.get("inferred_skills") or [])
        fallback.extend(chunk.get("llm_skills") or [])
        for line in str(chunk.get("skill_text") or "").splitlines():
            if line.strip().startswith("-"):
                fallback.append(line.strip("- ").strip())
    return [{"name": skill, "confidence": "medium"} for skill in _dedupe_keep_order(fallback)]


def extract_cv_profile(cv_chunks: List[Dict]) -> Dict[str, Any]:
    cv_schema = _first_schema(cv_chunks, "cv_schema")
    cv_rows = _cv_skill_rows(cv_chunks, cv_schema)
    stored_experience = next(
        (
            chunk.get("cv_experience")
            for chunk in cv_chunks
            if isinstance(chunk, dict) and isinstance(chunk.get("cv_experience"), dict)
        ),
        {},
    )
    months = int(cv_schema.get("experience_months") or round(float(stored_experience.get("years") or 0) * 12))
    years = round(months / 12, 2) if months else 0
    skills = _dedupe_keep_order([_row_name(row) for row in cv_rows])
    return {
        "skills": skills[:15],
        "all_skills": skills,
        "total_skills": len(skills),
        "experience_years": years,
        "experience_duration": str(stored_experience.get("display") or _format_months(months)),
        "experience_level": "Senior" if months >= 60 else "Mid-level" if months >= 24 else "Junior/Intern",
    }


def _format_months(months: int) -> str:
    if months <= 0:
        return "0 years"
    if months < 12:
        return f"{months} month" + ("" if months == 1 else "s")
    years = round(months / 12, 1)
    return f"{years:g} year" + ("" if years == 1 else "s")


def _candidate_name(source: str, chunks: List[Dict], get_candidate_name) -> str:
    for chunk in chunks:
        if chunk.get("candidate_name"):
            return str(chunk["candidate_name"])
    return get_candidate_name(source) or source


def _build_evidence(
    matched_required: List[str],
    matched_preferred: List[str],
    evidence_rows: Dict[str, Dict],
    cv_chunks: List[Dict],
    limit: int = 8,
) -> List[Dict]:
    cv_text = _cv_text(cv_chunks)
    evidence = []
    for skill in matched_required + matched_preferred:
        row = evidence_rows.get(skill, {})
        snippet = str(row.get("evidence") or row.get("source") or row.get("inferred_from") or "")
        if not snippet:
            snippet = next((str(ch.get("text") or "")[:500] for ch in cv_chunks if _skills_match(skill, str(ch.get("text") or ""))), "")
        evidence.append({
            "skill": skill,
            "jd_requirement_type": "required" if skill in matched_required else "preferred",
            "jd_section": "REQUIREMENTS" if skill in matched_required else "SKILLS",
            "cv_section": str(row.get("section") or "schema"),
            "cv_text": snippet or cv_text[:500],
            "score": 1.0,
        })
        if len(evidence) >= limit:
            break
    return evidence


def _retrieval_query_text(jd_schema: Dict, jd_content: str) -> str:
    parts = [jd_content]
    for label, key in (
        ("REQUIRED_SKILLS", "required_skills"),
        ("PREFERRED_SKILLS", "preferred_skills"),
        ("COMPETENCIES", "competencies"),
        ("SOFT_SKILLS", "soft_skills"),
    ):
        names = [_row_name(row) for row in _rows(jd_schema, key)]
        if names:
            parts.append(f"[{label}]\n" + "\n".join(f"- {name}" for name in names))

    exp = jd_schema.get("experience") if isinstance(jd_schema, dict) else {}
    if isinstance(exp, dict) and exp.get("min_months"):
        parts.append(f"[EXPERIENCE]\nminimum_months: {exp.get('min_months')}")

    context = jd_schema.get("work_context") if isinstance(jd_schema, dict) else {}
    if isinstance(context, dict):
        context_lines = []
        for key in ("role", "seniority", "industry", "company_type"):
            if context.get(key):
                context_lines.append(f"{key}: {context[key]}")
        for key in ("domains", "platforms", "tools", "processes"):
            values = context.get(key)
            if isinstance(values, list) and values:
                context_lines.append(f"{key}: {', '.join(str(v) for v in values)}")
        if context_lines:
            parts.append("[WORK_CONTEXT]\n" + "\n".join(context_lines))

    return "\n\n".join(part for part in parts if str(part or "").strip())


def _retrieval_evidence(hits: List[Dict], limit: int = 5) -> List[Dict]:
    evidence = []
    for hit in hits[:limit]:
        evidence.append({
            "jd_requirement_type": "retrieval",
            "jd_section": "RAG",
            "cv_section": hit.get("section", "unknown"),
            "cv_text": str(hit.get("content") or hit.get("original_text") or hit.get("embedding_text") or "")[:500],
            "score": float(hit.get("score", 0.0)),
        })
    return evidence


def _get_cross_encoder_model():
    global _cross_encoder_model, _cross_encoder_load_attempted
    if not CROSS_ENCODER_ENABLED:
        return None
    if _cross_encoder_model is not None:
        return _cross_encoder_model
    if _cross_encoder_load_attempted:
        return None

    _cross_encoder_load_attempted = True
    try:
        from sentence_transformers import CrossEncoder

        _cross_encoder_model = CrossEncoder(
            CROSS_ENCODER_MODEL,
            local_files_only=CROSS_ENCODER_LOCAL_FILES_ONLY,
        )
        return _cross_encoder_model
    except Exception as exc:
        logger.warning("Cross-encoder unavailable; continuing without it: %s", exc)
        return None


def _truncate_for_cross_encoder(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(text) <= limit:
        return text
    truncated = text[:limit]
    sentence_end = max(truncated.rfind(". "), truncated.rfind("; "), truncated.rfind("\n"))
    if sentence_end >= int(limit * 0.65):
        return truncated[: sentence_end + 1].strip()
    return truncated.strip()


def _cross_encoder_probability(raw_score: Any) -> float:
    try:
        score = float(raw_score)
    except Exception:
        return 0.0
    if 0.0 <= score <= 1.0:
        return score
    return 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, score))))


def _cross_encoder_chunk_text(chunk: Dict) -> str:
    return str(
        chunk.get("content")
        or chunk.get("text")
        or chunk.get("original_text")
        or chunk.get("embedding_text")
        or ""
    )


def _candidate_cross_encoder_chunks(cv_chunks: List[Dict], retrieval_hits: List[Dict] | None = None) -> List[Dict]:
    candidates: List[Dict] = []
    candidates.extend(hit for hit in (retrieval_hits or []) if isinstance(hit, dict))
    candidates.extend(chunk for chunk in (cv_chunks or []) if isinstance(chunk, dict))

    selected = []
    seen = set()
    for chunk in candidates:
        text = _cross_encoder_chunk_text(chunk)
        clean = re.sub(r"\s+", " ", text).strip()
        if len(clean) < 20:
            continue
        key = clean[:500].lower()
        if key in seen:
            continue
        seen.add(key)
        selected.append(
            {
                "section": chunk.get("section", "unknown"),
                "text": clean,
                "retrieval_score": float(chunk.get("score", 0.0) or 0.0),
            }
        )
        if len(selected) >= CROSS_ENCODER_MAX_CHUNKS:
            break
    return selected


def _cross_encoder_score(
    jd_content: str,
    cv_chunks: List[Dict],
    *,
    retrieval_hits: List[Dict] | None = None,
) -> tuple[float, List[Dict]]:
    model = _get_cross_encoder_model()
    if model is None:
        return 0.0, []

    jd_text = _truncate_for_cross_encoder(jd_content, CROSS_ENCODER_MAX_JD_CHARS)
    chunks = _candidate_cross_encoder_chunks(cv_chunks, retrieval_hits)
    if not jd_text or not chunks:
        return 0.0, []

    pairs = [
        [jd_text, _truncate_for_cross_encoder(chunk["text"], CROSS_ENCODER_MAX_CHUNK_CHARS)]
        for chunk in chunks
    ]
    try:
        predictions = model.predict(pairs, show_progress_bar=False)
    except TypeError:
        predictions = model.predict(pairs)
    except Exception as exc:
        logger.warning("Cross-encoder scoring failed: %s", exc)
        return 0.0, []

    scored = []
    for chunk, raw_score in zip(chunks, list(predictions)):
        probability = _cross_encoder_probability(raw_score)
        scored.append({**chunk, "score": probability})
    scored.sort(key=lambda item: item["score"], reverse=True)

    top = scored[:CROSS_ENCODER_TOP_AVG]
    score = round(100 * sum(item["score"] for item in top) / max(1, len(top)), 1)
    evidence = [
        {
            "jd_requirement_type": "cross_encoder",
            "jd_section": "CROSS_ENCODER",
            "cv_section": item.get("section", "unknown"),
            "cv_text": item["text"][:500],
            "score": round(item["score"], 4),
        }
        for item in scored[:3]
    ]
    return score, evidence


def _requirement_units(jd_schema: Dict) -> List[Dict]:
    def is_label(value: str) -> bool:
        return bool(re.match(r"^\s*(?:job\s+title|title|position|role)\s*:", str(value or ""), re.I))

    schema_units = jd_schema.get("requirement_units") if isinstance(jd_schema, dict) else []
    units: List[Dict] = []
    seen = set()
    for unit in schema_units if isinstance(schema_units, list) else []:
        if not isinstance(unit, dict):
            continue
        name = str(unit.get("name") or "").strip()
        evidence = str(unit.get("evidence") or "")
        category = str(unit.get("category") or "skill").strip().lower()
        importance = str(unit.get("importance") or "required").strip().lower()
        if category in {"skill", "certification"}:
            source = "preferred_skill" if importance == "preferred" else "required_skill"
        elif category == "soft_skill":
            source = "soft_skill"
        elif category == "responsibility":
            source = "responsibility"
        else:
            continue
        dedupe_key = (source, _skill_key(name), _skill_key(evidence))
        if not name or is_label(name) or is_label(evidence) or dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        units.append(
            {
                "name": name,
                "source": source,
                "importance": importance or source,
                "evidence": evidence,
                "row": unit,
            }
        )
        if len(units) >= REQUIREMENT_EVIDENCE_MAX_REQUIREMENTS:
            return units
    if units:
        return units

    specs = [
        ("required_skills", "required_skill", "required"),
        ("preferred_skills", "preferred_skill", "preferred"),
        ("competencies", "competency", "competency"),
        ("soft_skills", "soft_skill", "competency"),
    ]
    units = []
    seen = set()
    for key, source, importance in specs:
        for row in _rows(jd_schema, key):
            name = _row_name(row)
            dedupe_key = (source, _skill_key(name))
            evidence = str(row.get("evidence") or "")
            if not name or is_label(name) or is_label(evidence) or dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            units.append(
                {
                    "name": name,
                    "source": source,
                    "importance": str(row.get("importance") or importance),
                    "evidence": str(row.get("evidence") or ""),
                    "row": row,
                }
            )
            if len(units) >= REQUIREMENT_EVIDENCE_MAX_REQUIREMENTS:
                return units
    return units


def _requirement_prompt(unit: Dict, jd_schema: Dict) -> str:
    role = ""
    context = jd_schema.get("work_context") if isinstance(jd_schema, dict) else {}
    if isinstance(context, dict):
        role = str(context.get("role") or "")
    title = str(jd_schema.get("job_title") or role or "").strip()
    parts = [
        f"Requirement: {unit.get('name', '')}",
        f"Type: {unit.get('source', '')}",
    ]
    if title:
        parts.append(f"Role: {title}")
    if unit.get("evidence"):
        parts.append(f"JD evidence: {unit['evidence']}")
    return "\n".join(parts)


def _chunk_candidate_score(unit: Dict, chunk: Dict, cv_rows: List[Dict]) -> float:
    text = _cross_encoder_chunk_text(chunk)
    name = str(unit.get("name") or "")
    if not text or not name:
        return 0.0
    score = 0.0
    if _skill_text_match(name, text):
        score += 4.0
    req_tokens = _token_set(name)
    text_tokens = _token_set(text)
    if req_tokens and text_tokens:
        score += len(req_tokens & text_tokens) / max(1, len(req_tokens))
    for row in cv_rows:
        row_name = _row_name(row)
        evidence = str(row.get("evidence") or "")
        if _skills_match(name, row_name):
            score += 3.0
            if evidence and evidence.lower() in text.lower():
                score += 2.0
    return score


def _candidate_requirement_chunks(unit: Dict, cv_chunks: List[Dict], cv_rows: List[Dict]) -> List[Dict]:
    candidates = []
    for chunk in cv_chunks or []:
        if not isinstance(chunk, dict):
            continue
        text = _cross_encoder_chunk_text(chunk)
        clean = re.sub(r"\s+", " ", text).strip()
        if len(clean) < 20:
            continue
        candidates.append(
            {
                "section": chunk.get("section", "unknown"),
                "text": clean,
                "candidate_score": _chunk_candidate_score(unit, chunk, cv_rows),
            }
        )
    candidates.sort(key=lambda item: (item["candidate_score"], len(item["text"])), reverse=True)
    return candidates[:REQUIREMENT_EVIDENCE_MAX_CHUNKS]


def _requirement_evidence_matches(
    jd_schema: Dict,
    cv_chunks: List[Dict],
) -> tuple[List[Dict], List[Dict]]:
    if not REQUIREMENT_EVIDENCE_ENABLED:
        return [], []
    model = _get_cross_encoder_model()
    if model is None:
        return [], []

    cv_schema = _first_schema(cv_chunks, "cv_schema")
    cv_rows = _cv_skill_rows(cv_chunks, cv_schema)
    cv_text = _cv_text(cv_chunks)
    units = _requirement_units(jd_schema)
    if not units:
        return [], []

    best_by_requirement: Dict[tuple[str, str], Dict] = {}
    for unit in units:
        name = str(unit.get("name") or "")
        match_row = next((row for row in cv_rows if _skills_match(name, _row_name(row))), None)
        if not match_row:
            match_row = _skill_text_match(name, cv_text)
        if match_row:
            key = (str(unit.get("source") or ""), _skill_key(name))
            best_by_requirement[key] = {
                "requirement": name,
                "source": unit.get("source", ""),
                "importance": unit.get("importance", ""),
                "status": "matched",
                "confidence": 0.95,
                "evidence": str(match_row.get("evidence") or match_row.get("source") or name)[:500],
                "cv_section": str(match_row.get("section") or "schema"),
                "method": "schema_or_raw_evidence",
            }

    work_items = []
    pairs = []
    for unit in units:
        requirement_text = _truncate_for_cross_encoder(_requirement_prompt(unit, jd_schema), 700)
        for chunk in _candidate_requirement_chunks(unit, cv_chunks, cv_rows):
            work_items.append((unit, chunk))
            pairs.append([requirement_text, _truncate_for_cross_encoder(chunk["text"], CROSS_ENCODER_MAX_CHUNK_CHARS)])

    if not pairs:
        return [], []

    try:
        predictions = model.predict(pairs, show_progress_bar=False)
    except TypeError:
        predictions = model.predict(pairs)
    except Exception as exc:
        logger.warning("Requirement evidence scoring failed: %s", exc)
        return [], []

    for (unit, chunk), raw_score in zip(work_items, list(predictions)):
        confidence = _cross_encoder_probability(raw_score)
        key = (str(unit.get("source") or ""), _skill_key(str(unit.get("name") or "")))
        current = best_by_requirement.get(key)
        if not current or confidence > current["confidence"]:
            best_by_requirement[key] = {
                "requirement": unit.get("name", ""),
                "source": unit.get("source", ""),
                "importance": unit.get("importance", ""),
                "status": "matched" if confidence >= REQUIREMENT_EVIDENCE_THRESHOLD else "missing",
                "confidence": round(confidence, 4),
                "evidence": chunk["text"][:500] if confidence >= REQUIREMENT_EVIDENCE_THRESHOLD else "",
                "cv_section": chunk.get("section", "unknown"),
                "method": "cross_encoder_requirement_evidence",
            }

    details = []
    for unit in units:
        key = (str(unit.get("source") or ""), _skill_key(str(unit.get("name") or "")))
        details.append(
            best_by_requirement.get(
                key,
                {
                    "requirement": unit.get("name", ""),
                    "source": unit.get("source", ""),
                    "importance": unit.get("importance", ""),
                    "status": "missing",
                    "confidence": 0.0,
                    "evidence": "",
                    "cv_section": "",
                    "method": "cross_encoder_requirement_evidence",
                },
            )
        )

    evidence = [
        {
            "skill": item["requirement"],
            "jd_requirement_type": item["source"],
            "jd_section": "REQUIREMENT_EVIDENCE",
            "cv_section": item.get("cv_section", "unknown"),
            "cv_text": item.get("evidence", ""),
            "score": item.get("confidence", 0.0),
        }
        for item in details
        if item.get("status") == "matched" and item.get("evidence")
    ][:8]
    return details, evidence


def _ratio_from_requirement_details(details: List[Dict], *sources: str) -> tuple[float, List[str], List[str]]:
    selected = [item for item in details if item.get("source") in sources]
    if not selected:
        return 1.0, [], []
    matched = [str(item.get("requirement") or "") for item in selected if item.get("status") == "matched"]
    missing = [str(item.get("requirement") or "") for item in selected if item.get("status") != "matched"]
    return len(matched) / max(1, len(selected)), _dedupe_keep_order(matched), _dedupe_keep_order(missing)


def _apply_requirement_evidence_refinement(
    jd_schema: Dict,
    cv_chunks: List[Dict],
    evaluation: Dict,
    section_scores: Dict,
    evidence: List[Dict],
) -> tuple[float, Dict, Dict, List[Dict]]:
    details, requirement_evidence = _requirement_evidence_matches(jd_schema, cv_chunks)
    if not details:
        return float(evaluation.get("score", 0)), evaluation, section_scores, evidence

    required_ratio, matched_required, missing_required = _ratio_from_requirement_details(details, "required_skill")
    preferred_ratio, matched_preferred, missing_preferred = _ratio_from_requirement_details(details, "preferred_skill")
    competency_ratio, matched_comp, missing_comp = _ratio_from_requirement_details(details, "competency", "soft_skill")

    weights = section_scores.get("weights") if isinstance(section_scores.get("weights"), dict) else _normalize_weights(jd_schema)
    old_required = section_scores.get("required_skills", 100) / 100
    old_preferred = section_scores.get("preferred_skills", 100) / 100
    old_competency = section_scores.get("competencies", 100) / 100

    if matched_required or missing_required:
        section_scores["required_skills"] = round(100 * required_ratio)
    else:
        required_ratio = old_required
    if matched_preferred or missing_preferred:
        section_scores["preferred_skills"] = round(100 * preferred_ratio)
    else:
        preferred_ratio = old_preferred
    if matched_comp or missing_comp:
        section_scores["competencies"] = round(100 * competency_ratio)
    else:
        competency_ratio = old_competency

    skill_weight = weights.get("required_skills", 0) + weights.get("preferred_skills", 0) + weights.get("competencies", 0)
    skill_ratio = (
        weights.get("required_skills", 0) * required_ratio
        + weights.get("preferred_skills", 0) * preferred_ratio
        + weights.get("competencies", 0) * competency_ratio
    ) / max(skill_weight, 0.01)
    quality_ratio = section_scores.get("quality", 0) / 100
    context_ratio = section_scores.get("context", 0) / 100
    fit_ratio = (quality_ratio + context_ratio) / 2

    evaluation = evaluation.copy()
    has_required_details = any(item.get("source") == "required_skill" for item in details)
    has_preferred_details = any(item.get("source") == "preferred_skill" for item in details)
    has_competency_details = any(item.get("source") in {"competency", "soft_skill"} for item in details)
    evaluation["matched_required_skills"] = matched_required if has_required_details else evaluation.get("matched_required_skills", [])
    evaluation["missing_required_skills"] = missing_required if has_required_details else evaluation.get("missing_required_skills", [])
    evaluation["matched_preferred_skills"] = matched_preferred if has_preferred_details else evaluation.get("matched_preferred_skills", [])
    evaluation["missing_preferred_skills"] = missing_preferred if has_preferred_details else evaluation.get("missing_preferred_skills", [])
    evaluation["matched_competencies"] = matched_comp if has_competency_details else evaluation.get("matched_competencies", [])
    evaluation["missing_competencies"] = missing_comp if has_competency_details else evaluation.get("missing_competencies", [])
    evaluation["matched_skills"] = _dedupe_keep_order(
        evaluation.get("matched_required_skills", [])
        + evaluation.get("matched_preferred_skills", [])
        + evaluation.get("matched_competencies", [])
    )
    evaluation["missing_skills"] = _dedupe_keep_order(
        evaluation.get("missing_required_skills", [])
        + evaluation.get("missing_preferred_skills", [])
        + evaluation.get("missing_competencies", [])
    )
    evaluation["requirement_match_details"] = details
    evaluation["technical_score"] = round(40 * skill_ratio)
    evaluation["fit_score"] = round(10 * fit_ratio)
    evaluation["score"] = sum(evaluation[key] for key in SCORE_LIMITS)
    evaluation["recommendation"] = _recommendation(evaluation["score"])
    cv_schema = _first_schema(cv_chunks, "cv_schema")
    cv_months = int((cv_schema or {}).get("experience_months") or 0)
    required_months = int(((jd_schema or {}).get("experience") or {}).get("min_months") or 0)
    evaluation["summary"] = _summary(
        evaluation["score"],
        evaluation.get("matched_required_skills", []),
        evaluation.get("missing_required_skills", []),
        cv_months,
        required_months,
    )
    return float(evaluation["score"]), evaluation, section_scores, (requirement_evidence + evidence)[:8]


def _combine_rerank_score(schema_score: float, retrieval_score: float, cross_encoder_score: float) -> tuple[float, str]:
    retrieval_pct = max(0.0, min(100.0, retrieval_score * 100))
    if cross_encoder_score > 0:
        remaining = max(0.0, 1.0 - CROSS_ENCODER_WEIGHT)
        schema_weight = remaining * 0.80
        retrieval_weight = remaining * 0.20
        score = (
            schema_score * schema_weight
            + cross_encoder_score * CROSS_ENCODER_WEIGHT
            + retrieval_pct * retrieval_weight
        )
        return round(score, 1), "schema_score+cross_encoder+rag"
    return round((schema_score * 0.85) + (retrieval_pct * 0.15), 1), "schema_score+rag_tiebreak"


def _retrieve_candidate_sources(
    jd_schema: Dict,
    jd_content: str,
    *,
    top_k: int,
    source_whitelist: List[str] | None,
    search_similar_cv_chunks,
) -> tuple[List[str], Dict[str, float], Dict[str, List[Dict]], str]:
    query_text = _retrieval_query_text(jd_schema, jd_content)
    if not query_text.strip():
        return [], {}, {}, "empty_query"

    try:
        embedding = get_embedding(query_text)
        hits = search_similar_cv_chunks(embedding, k=max(top_k * 12, 50))
    except Exception as exc:
        logger.warning("RAG retrieval failed, falling back to schema scan: %s", exc)
        return [], {}, {}, "retrieval_failed"

    allowed = set(source_whitelist or []) if source_whitelist is not None else None
    scores: Dict[str, float] = {}
    hits_by_source: Dict[str, List[Dict]] = {}
    for hit in hits or []:
        source = hit.get("source")
        if not source or (allowed is not None and source not in allowed):
            continue
        score = float(hit.get("score", 0.0))
        scores[source] = max(scores.get(source, 0.0), score)
        hits_by_source.setdefault(source, []).append(hit)

    ranked_sources = sorted(scores, key=lambda src: scores[src], reverse=True)
    return ranked_sources[:max(top_k * 5, top_k, 10)], scores, hits_by_source, "rag_vector"


def _dedupe_sources(sources: List[str]) -> List[str]:
    seen = set()
    result = []
    for source in sources:
        if source and source not in seen:
            seen.add(source)
            result.append(source)
    return result


def _score_candidate(jd_schema: Dict, jd_content: str, cv_chunks: List[Dict]) -> tuple[float, Dict, Dict, List[Dict], Dict]:
    cv_schema = _first_schema(cv_chunks, "cv_schema")
    cv_rows = _cv_skill_rows(cv_chunks, cv_schema)
    cv_text = _cv_text(cv_chunks)

    required_rows = _rows(jd_schema, "required_skills")
    preferred_rows = _rows(jd_schema, "preferred_skills")
    competency_rows = _rows(jd_schema, "competencies") + _rows(jd_schema, "soft_skills")

    required_ratio, matched_required, missing_required, req_evidence = _match_rows(required_rows, cv_rows, cv_text)
    preferred_ratio, matched_preferred, missing_preferred, pref_evidence = _match_rows(preferred_rows, cv_rows, cv_text)
    competency_ratio, matched_comp, missing_comp, _ = _match_rows(competency_rows, cv_rows, cv_text)
    experience_ratio, cv_months, required_months = _score_experience(jd_schema, cv_schema)
    education_ratio, required_edu, cv_edu = _score_education(jd_schema, cv_schema)
    quality_ratio = _quality_score(cv_schema)
    context_ratio = _context_score(jd_schema, cv_rows, cv_text)

    weights = _normalize_weights(jd_schema)
    ratios = {
        "required_skills": required_ratio,
        "preferred_skills": preferred_ratio,
        "competencies": competency_ratio,
        "experience": experience_ratio,
        "education": education_ratio,
        "quality": quality_ratio,
        "context": context_ratio,
    }
    score = round(sum(100 * weights[key] * ratios[key] for key in weights))

    skill_weight = weights["required_skills"] + weights["preferred_skills"] + weights["competencies"]
    skill_ratio = (
        weights["required_skills"] * required_ratio
        + weights["preferred_skills"] * preferred_ratio
        + weights["competencies"] * competency_ratio
    ) / max(skill_weight, 0.01)
    fit_ratio = (quality_ratio + context_ratio) / 2

    matched_skills = _dedupe_keep_order(matched_required + matched_preferred + matched_comp)
    missing_skills = _dedupe_keep_order(missing_required + missing_preferred + missing_comp)
    evaluation = {
        "score": max(0, min(100, score)),
        "technical_score": round(40 * skill_ratio),
        "experience_score": round(30 * experience_ratio),
        "education_score": round(20 * education_ratio),
        "fit_score": round(10 * fit_ratio),
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "matched_required_skills": matched_required,
        "missing_required_skills": missing_required,
        "matched_preferred_skills": matched_preferred,
        "missing_preferred_skills": missing_preferred,
        "matched_competencies": matched_comp,
        "missing_competencies": missing_comp,
        "recommendation": _recommendation(score),
        "summary": _summary(score, matched_required, missing_required, cv_months, required_months),
    }
    evaluation["score"] = sum(evaluation[key] for key in SCORE_LIMITS)
    evaluation["recommendation"] = _recommendation(evaluation["score"])

    section_scores = {
        "required_skills": round(100 * required_ratio),
        "preferred_skills": round(100 * preferred_ratio),
        "competencies": round(100 * competency_ratio),
        "experience": round(100 * experience_ratio),
        "education": round(100 * education_ratio),
        "quality": round(100 * quality_ratio),
        "context": round(100 * context_ratio),
        "weights": {key: round(val, 4) for key, val in weights.items()},
    }
    evidence_rows = {**req_evidence, **pref_evidence}
    evidence = _build_evidence(matched_required, matched_preferred, evidence_rows, cv_chunks)
    cv_profile = extract_cv_profile(cv_chunks)
    return float(evaluation["score"]), evaluation, section_scores, evidence, cv_profile


def _schema_based_evaluation(
    jd_schema: Dict,
    jd_content: str,
    cv_chunks: List[Dict],
    cv_profile: Dict | None = None,
    cv_context: Dict | None = None,
) -> tuple[float, Dict, Dict]:
    score, evaluation, section_scores, _, _ = _score_candidate(jd_schema, jd_content, cv_chunks)
    weights = section_scores.get("weights") if isinstance(section_scores.get("weights"), dict) else {}
    weighted_score = round(sum(section_scores.get(key, 0) * weights.get(key, 0.0) for key in DEFAULT_WEIGHTS))
    return float(weighted_score or score), evaluation, section_scores


def _extract_skills_from_text(text: str) -> List[str]:
    block = str(text or "")
    match = re.search(r"\[SKILLS\]\s*([\s\S]*?)(?:\n\[|\Z)", block, re.I)
    if match:
        block = match.group(1)
    skills = []
    for item in re.split(r",|/|\bor\b|\band\b|&|\n", block, flags=re.I):
        item = item.strip(" -*.;:\t")
        if item and 1 <= len(item.split()) <= 5:
            skills.append(item)
    return _dedupe_keep_order(skills)


def _deterministic_evaluation(
    jd_content: str,
    cv_chunks: List[Dict],
    cv_profile: Dict,
    similarity_score: float = 0.0,
    cv_context: Dict | None = None,
) -> Dict:
    jd_required = _extract_skills_from_text(jd_content)
    if isinstance(cv_context, dict):
        cv_text = str(cv_context.get("text") or "")
        cv_skills = _dedupe_keep_order(list(cv_context.get("skills") or []))
    else:
        cv_schema = _first_schema(cv_chunks, "cv_schema")
        cv_text = _cv_text(cv_chunks)
        cv_skills = _dedupe_keep_order([_row_name(row) for row in _cv_skill_rows(cv_chunks, cv_schema)] + list(cv_profile.get("skills") or []))

    matched = []
    missing = []
    for skill in jd_required:
        if any(_skills_match(skill, cv_skill) for cv_skill in cv_skills) or _skill_text_match(skill, cv_text):
            matched.append(skill)
        else:
            missing.append(skill)

    evaluation = {
        "technical_score": round(40 * len(matched) / max(1, len(jd_required))) if jd_required else 40,
        "experience_score": 30,
        "education_score": 20,
        "fit_score": round(max(0, min(10, float(similarity_score or 0) * 10))),
        "matched_skills": _dedupe_keep_order(matched),
        "missing_skills": _dedupe_keep_order(missing),
        "matched_required_skills": _dedupe_keep_order(matched),
        "missing_required_skills": _dedupe_keep_order(missing),
        "matched_preferred_skills": [],
        "missing_preferred_skills": [],
    }
    evaluation["score"] = sum(evaluation[key] for key in SCORE_LIMITS)
    evaluation["recommendation"] = _recommendation(evaluation["score"])
    evaluation["summary"] = _summary(evaluation["score"], evaluation["matched_required_skills"], evaluation["missing_required_skills"], 0, 0)
    return evaluation


def _postprocess_evaluation(
    evaluation: Dict,
    jd_content: str,
    cv_chunks: List[Dict],
    cv_profile: Dict,
    cv_context: Dict | None = None,
) -> Dict:
    fixed = _deterministic_evaluation(jd_content, cv_chunks, cv_profile, cv_context=cv_context)
    summary = str((evaluation or {}).get("summary") or "").strip()
    if summary:
        fixed["summary"] = summary
    return fixed


def _required_filter_status(
    jd_schema: Dict,
    cv_chunks: List[Dict],
    cv_profile: Dict,
    cv_context: Dict | None = None,
) -> tuple[bool, float, List[str]]:
    _, evaluation, _ = _schema_based_evaluation(jd_schema, "", cv_chunks, cv_profile, cv_context)
    total = len(evaluation.get("matched_required_skills", [])) + len(evaluation.get("missing_required_skills", []))
    ratio = len(evaluation.get("matched_required_skills", [])) / max(1, total)
    threshold = _required_skill_threshold(jd_schema)
    return ratio >= threshold, ratio, list(evaluation.get("missing_required_skills", []))


def _passes_metadata_filter(jd_schema: Dict, cv_chunks: List[Dict], cv_profile: Dict) -> tuple[bool, List[str]]:
    cv_schema = _first_schema(cv_chunks, "cv_schema")
    months = int(cv_schema.get("experience_months") or round(float(cv_profile.get("experience_years") or 0) * 12))
    profile = dict(cv_profile)
    profile["experience_years"] = months / 12 if months else float(cv_profile.get("experience_years") or 0)
    return _passes_hard_filters(jd_schema, {"matched_required_skills": [], "missing_required_skills": []}, profile)


def _summary(score: float, matched_required: List[str], missing_required: List[str], cv_months: int, required_months: int) -> str:
    matched = ", ".join(matched_required[:5]) if matched_required else "no required skills"
    missing = ", ".join(missing_required[:5]) if missing_required else "no major required skill gaps"
    exp = ""
    if required_months:
        exp = f" Experience: {_format_months(cv_months)} against {_format_months(required_months)} required."
    return f"{_recommendation(score)} based on schema match: matched {matched}; missing {missing}.{exp}"


def _passes_hard_filters(jd_schema: Dict, evaluation: Dict, cv_profile: Dict) -> tuple[bool, List[str]]:
    filters = _hard_filters(jd_schema)
    reasons = []
    metadata = jd_schema.get("metadata_filter") if isinstance(jd_schema, dict) else {}
    if filters["metadata"] and isinstance(metadata, dict):
        min_years = metadata.get("exp_years_min")
        try:
            if min_years is not None and float(cv_profile.get("experience_years") or 0) < float(min_years):
                reasons.append(f"experience below metadata minimum ({min_years} years)")
        except Exception:
            pass

    threshold = _required_skill_threshold(jd_schema)
    required_total = len(evaluation.get("matched_required_skills", [])) + len(evaluation.get("missing_required_skills", []))
    required_ratio = len(evaluation.get("matched_required_skills", [])) / max(1, required_total)
    if filters["required_skills"] and threshold > 0 and required_total and required_ratio < threshold:
        reasons.append(f"required skill match rate {required_ratio:.0%} below threshold")
    return not reasons, reasons


def _jd_content_and_schema(jd_chunks: List[Dict]) -> tuple[str, Dict]:
    parts = []
    for chunk in jd_chunks:
        section = str(chunk.get("section") or "requirements").upper()
        content = str(chunk.get("content") or chunk.get("embedding_text") or "")
        if content:
            parts.append(f"[{section}]\n{content}")
    content = "\n\n".join(parts)
    schema = _first_schema(jd_chunks, "jd_schema")
    return content, _normalize_jd_schema(schema, fallback_text=content) if schema else {}


def match_jd_to_cvs(
    jd_id: str,
    top_k: int = 5,
    *,
    use_llm: bool = False,
    source_whitelist: List[str] | None = None,
) -> List[Dict]:
    from mongo_utils import (
        get_candidate_name,
        get_chunks_by_source_for_matching,
        get_distinct_sources,
        search_similar_cv_chunks,
    )

    target_k = max(1, int(top_k or 1))

    if count_indexed_jds() == 0:
        return _error_result(jd_id, "JDs not indexed yet", "Please index JDs first using the Index JDs button")

    jd_chunks = get_jd_chunks(jd_id)
    if not jd_chunks:
        return _error_result(jd_id, "JD not found", "JD chunks not available")

    jd_title = str(jd_chunks[0].get("title") or jd_id)
    jd_content, jd_schema = _jd_content_and_schema(jd_chunks)
    if not jd_schema:
        return _error_result(jd_id, jd_title, "JD schema is missing. Re-index this JD before matching.", method="schema_required")

    all_sources = get_distinct_sources() or []
    if source_whitelist is not None:
        allowed = set(source_whitelist or [])
        all_sources = [source for source in all_sources if source in allowed]
    if not all_sources:
        return _error_result(jd_id, jd_title, "No CVs available in the selected scope")

    candidate_sources, retrieval_scores, retrieval_hits, retrieval_method = _retrieve_candidate_sources(
        jd_schema,
        jd_content,
        top_k=top_k,
        source_whitelist=source_whitelist,
        search_similar_cv_chunks=search_similar_cv_chunks,
    )
    if not candidate_sources:
        candidate_sources = all_sources
        retrieval_method = f"{retrieval_method}+schema_fallback" if retrieval_method else "schema_fallback"
    else:
        candidate_sources = _dedupe_sources(candidate_sources + all_sources)

    results = []
    rejected = {}
    for source in candidate_sources:
        cv_chunks = get_chunks_by_source_for_matching(source)
        if not cv_chunks:
            continue
        score, evaluation, section_scores, evidence, cv_profile = _score_candidate(jd_schema, jd_content, cv_chunks)
        score, evaluation, section_scores, evidence = _apply_requirement_evidence_refinement(
            jd_schema,
            cv_chunks,
            evaluation,
            section_scores,
            evidence,
        )
        ok, reasons = _passes_hard_filters(jd_schema, evaluation, cv_profile)

        retrieval_score = float(retrieval_scores.get(source, 0.0))
        cross_score, cross_evidence = _cross_encoder_score(
            jd_content,
            cv_chunks,
            retrieval_hits=retrieval_hits.get(source, []),
        )
        rerank_score, rerank_method = _combine_rerank_score(score, retrieval_score, cross_score)
        if not ok:
            rejected[source] = reasons
            evaluation = evaluation.copy()
            evaluation["recommendation"] = "Not a fit"
            evaluation["summary"] = f"{evaluation.get('summary', '')} Filter flags: {'; '.join(reasons)}".strip()
            rerank_score = min(rerank_score, 44.0)
        results.append({
            "cv_source": source,
            "candidate_name": _candidate_name(source, cv_chunks, get_candidate_name),
            "jd_id": jd_id,
            "jd_title": jd_title,
            "similarity_score": round(score, 1),
            "dense_score": round(retrieval_score * 100, 1),
            "bm25_score": round(section_scores.get("required_skills", 0), 1),
            "cross_encoder_score": cross_score,
            "rerank_score": rerank_score,
            "rerank_method": rerank_method,
            "retrieval_method": retrieval_method,
            "section_scores": section_scores,
            "evaluation": evaluation,
            "cv_profile": cv_profile,
            "match_evidence": (evidence + cross_evidence + _retrieval_evidence(retrieval_hits.get(source, [])))[:8],
        })

    if not results:
        return _error_result(jd_id, jd_title, "No candidate CV chunks found")

    results.sort(key=lambda item: (item.get("rerank_score", 0), item["evaluation"].get("score", 0)), reverse=True)
    return results[:target_k]
