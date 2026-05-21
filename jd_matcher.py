"""
jd_matcher.py
=============
Match CV chunks với JD chunks bằng global vector search.

Luồng chính:
  1. Mỗi CV chunk được embed riêng → tìm JD chunks gần nhất trên toàn bộ collection
  2. Boost score theo section của JD chunk trả về → tổng hợp theo jd_id
  3. LLM đánh giá chi tiết JD có score cao nhất
"""

import json
import logging
import re
import unicodedata
from collections import defaultdict
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any, Dict, List

from langchain_core.prompts import PromptTemplate

from bedrock_utils import get_embedding
from jd_store import (
    count_indexed_jds,
    get_jd_chunks,
    search_similar_jds,
    search_similar_jd_skills,
)

logger = logging.getLogger(__name__)

LLM_MODEL = "qwen2.5:1.5b"

# Trọng số boost khi tổng hợp similarity score theo section của JD chunk.
SECTION_WEIGHTS = {
    "experience":   0.35,
    "skills":       0.30,
    "requirements": 0.20,
    "education":    0.075,
    "soft_skills":  0.075,
}
DEFAULT_SECTION_WEIGHT = 0.10


# =============================================================================
# TEXT HELPERS
# =============================================================================

def truncate_text(text: str, max_length: int = 1000) -> str:
    if len(text) <= max_length:
        return text
    truncated = text[:max_length]
    last_period = truncated.rfind(".")
    if last_period > max_length * 0.65:
        return truncated[: last_period + 1] + "..."
    return truncated + "..."


def extract_relevant_sections(cv_chunks: List[Dict]) -> str:
    """Lấy các CV chunks hữu ích để đưa vào LLM, không phụ thuộc tên section."""
    sections = []

    for ch in cv_chunks:
        if ch.get("skip_embed"):
            continue

        raw_section = str(ch.get("section", "")).strip().lower()
        label = raw_section.upper() or "CHUNK"
        skill_text = ch.get("skill_text") or ""
        content = "\n\n".join(
            part for part in [ch.get("text", ""), skill_text] if part
        )
        content = truncate_text(content, 1100 if "experience" in raw_section else 700)
        sections.append(f"[{label}]\n{content}")

    # Fallback nếu không nhận ra section nào
    if not sections and cv_chunks:
        for i, ch in enumerate(cv_chunks[:5]):
            sections.append(f"[CHUNK {i + 1}]\n{truncate_text(ch.get('text', ''), 600)}")

    return "\n\n".join(sections)


# =============================================================================
# CV PROFILE
# =============================================================================

def extract_cv_profile(cv_chunks: List[Dict]) -> Dict[str, Any]:
    full_text = " ".join(
        str(ch.get("text") or ch.get("embedding_text") or ch.get("content") or "")
        for ch in cv_chunks
    ).lower()
    found_skills = []
    for chunk in cv_chunks:
        info = chunk.get("extracted_info") if isinstance(chunk, dict) else {}
        if isinstance(info, dict):
            found_skills.extend(info.get("chunk_skills") or [])
            found_skills.extend(info.get("technical_skills") or [])
        skill_text = chunk.get("skill_text") if isinstance(chunk, dict) else ""
        if skill_text:
            found_skills.extend(
                line.strip("- ").strip()
                for line in str(skill_text).splitlines()
                if line.strip().startswith("-")
            )
    found_skills = list(dict.fromkeys(s for s in found_skills if s))
    stored_experience = _stored_cv_experience(cv_chunks)
    if stored_experience:
        experience_years = float(stored_experience.get("years") or 0)
        experience_duration = str(stored_experience.get("display") or _format_experience_duration(experience_years))
    else:
        experience_years = _calculate_experience_years(full_text)
        experience_duration = _format_experience_duration(experience_years)

    experience_level  = (
        "Senior"       if experience_years >= 5 else
        "Mid-level"    if experience_years >= 2 else
        "Junior/Intern"
    )

    return {
        "skills":          found_skills[:15],
        "total_skills":    len(found_skills),
        "experience_years": experience_years,
        "experience_duration": experience_duration,
        "experience_level": experience_level,
    }


def build_cv_summary(cv_chunks: List[Dict], cv_profile: Dict) -> str:
    relevant    = extract_relevant_sections(cv_chunks)
    profile_str = (
        f"[CV PRE-PROCESSED SUMMARY]\n"
        f"Experience Level : {cv_profile['experience_level']} ({cv_profile.get('experience_duration', '0 years')})\n"
        f"Technical Skills : {cv_profile['total_skills']} skills detected\n"
        f"Key Skills       : {', '.join(cv_profile['skills']) if cv_profile['skills'] else 'None'}"
    )
    return profile_str + "\n\n" + relevant


# =============================================================================
# GLOBAL VECTOR MATCHING
# =============================================================================

def _search_matching_jd_chunks(embedding: list, k_per_chunk: int) -> List[Dict]:
    """Search toàn bộ JD collection; section chỉ dùng ở bước boost score."""
    try:
        return search_similar_jds(embedding, k=k_per_chunk)
    except Exception as e:
        logger.warning("Global JD search failed: %s", e)
        return []

def _search_matching_jd_skill_chunks(skill_embedding: list, k_per_chunk: int) -> List[Dict]:
    try:
        return search_similar_jd_skills(skill_embedding, k=k_per_chunk)
    except Exception as e:
        logger.warning("JD skill-layer search failed, falling back to content layer only: %s", e)
        return []

def _merge_layer_matches(content_matches: List[Dict], skill_matches: List[Dict]) -> List[Dict]:
    merged: Dict[str, Dict] = {}
    for layer, matches in (("content", content_matches), ("skills", skill_matches)):
        for match in matches:
            key = match.get("chunk_id") or f"{match.get('jd_id')}:{match.get('section')}:{match.get('chunk_index')}"
            score = float(match.get("score", 0.0))
            if key not in merged or score > float(merged[key].get("score", 0.0)):
                row = match.copy()
                row["matched_layers"] = [layer]
                merged[key] = row
            elif layer not in merged[key].setdefault("matched_layers", []):
                merged[key]["matched_layers"].append(layer)
    return sorted(merged.values(), key=lambda item: float(item.get("score", 0.0)), reverse=True)


def _section_weight(section: str) -> float:
    return SECTION_WEIGHTS.get(str(section or "").strip().lower(), DEFAULT_SECTION_WEIGHT)


def _boost_score(score: float, section: str) -> float:
    return score * (1.0 + _section_weight(section))


def _aggregate_boosted_scores(scores: List[float]) -> float:
    if not scores:
        return 0.0
    best = max(scores)
    mean = sum(scores) / len(scores)
    return min(1.0, 0.7 * best + 0.3 * mean)


def _match_chunks_to_jds(cv_chunks: List[Dict], k_per_chunk: int = 5) -> Dict[str, Dict]:
    """
    Embed từng CV chunk → search toàn bộ JD collection → tổng hợp score theo jd_id.

    Section của CV không còn dùng làm filter cứng. Section của JD chunk trả về
    chỉ dùng để boost/rank sau khi vector search đã tìm candidate phù hợp.

    Returns:
        Dict[jd_id] = {
            "title":          str,
            "weighted_score": float,   # 0–1, tổng hợp có trọng số
            "section_scores": Dict[section, float],
            "matched_chunks": List[Dict],  # JD chunks tìm thấy
        }
    """
    jd_data: Dict[str, Dict] = defaultdict(lambda: {
        "title":         "",
        "section_scores": defaultdict(list),
        "boosted_scores": [],
        "matched_chunks": [],
    })

    for cv_chunk in cv_chunks:
        if cv_chunk.get("skip_embed"):
            continue

        raw_section = str(cv_chunk.get("section", "")).strip().lower()

        try:
            stored_embedding = cv_chunk.get("embedding")
            if isinstance(stored_embedding, list) and stored_embedding:
                embedding = stored_embedding
            else:
                embedding_source = cv_chunk.get("embedding_text") or cv_chunk.get("text") or ""
                embedding = get_embedding(embedding_source)
        except Exception as e:
            logger.warning("Embed thất bại chunk '%s': %s", raw_section, e)
            continue

        content_matches = _search_matching_jd_chunks(embedding, k_per_chunk)
        skill_matches: List[Dict] = []
        skill_text = cv_chunk.get("skill_text") or ""
        if skill_text.strip():
            try:
                skill_embedding = get_embedding(skill_text)
                skill_matches = _search_matching_jd_skill_chunks(skill_embedding, k_per_chunk)
            except Exception as e:
                logger.warning("CV skill embedding failed for chunk '%s': %s", raw_section, e)

        jd_chunks = _merge_layer_matches(content_matches, skill_matches)

        for jd_chunk in jd_chunks:
            jid   = jd_chunk.get("jd_id", "unknown")
            score = float(jd_chunk.get("score", 0.0))

            jd_data[jid]["title"] = jd_chunk.get("title", "Untitled")
            matched_section = jd_chunk.get("section") or "unknown"
            jd_data[jid]["section_scores"][matched_section].append(score)
            jd_data[jid]["boosted_scores"].append(_boost_score(score, matched_section))
            jd_data[jid]["matched_chunks"].append(jd_chunk)

    # Tổng hợp score sau khi boost theo section của JD chunk trả về.
    results = {}
    for jid, data in jd_data.items():
        weighted = _aggregate_boosted_scores(data["boosted_scores"])

        # Gộp content các JD chunks theo jd_id để đưa vào LLM
        seen_chunks = {}
        for ch in get_jd_chunks(jid) or data["matched_chunks"]:
            if ch["jd_id"] == jid:
                content = ch.get("content", "")
                skill_text = ch.get("skill_text") or ""
                seen_chunks[ch.get("section", "")] = "\n\n".join(
                    part for part in [content, skill_text] if part
                )

        jd_content = "\n\n".join(
            f"[{sec.upper()}]\n{txt}" for sec, txt in seen_chunks.items()
        )

        results[jid] = {
            "title":          data["title"],
            "weighted_score": round(weighted, 4),
            "section_scores": {
                sec: round(max(scores), 4)
                for sec, scores in data["section_scores"].items()
            },
            "jd_content":     jd_content,
        }

    return results


# =============================================================================
# LLM EVALUATION
# =============================================================================

MATCH_PROMPT = PromptTemplate.from_template(
    """You are a strict and experienced technical recruiter.

=== JOB DESCRIPTION ===
{jd_content}

=== CANDIDATE CV ===
{cv_summary}

Evaluate this candidate strictly against the Job Description.

Scoring Criteria:
- Technical Skills Match (40 points): How many required skills are present?
- Experience Relevance (30 points): Use the JD requirement. If the JD does not require experience, do not penalize lack of years.
- Education & Background (20 points): Use the JD requirement. If the JD does not mention a specific degree/background, give a neutral/pass score.
- Overall Fit & Communication (10 points): never exceed 10.

Rules:
- Never return a sub-score above its maximum.
- matched_skills and missing_skills must only contain skills explicitly present in the JD.
- Do not invent skills that are not in the JD.

Return ONLY a valid JSON object (no extra text, no markdown):

{{
  "score": <total 0-100>,
  "technical_score": <0-40>,
  "experience_score": <0-30>,
  "education_score": <0-20>,
  "fit_score": <0-10>,
  "matched_skills": ["list of matching skills"],
  "missing_skills": ["important missing skills"],
  "recommendation": "Strong fit" | "Good fit" | "Consider" | "Not a fit",
  "summary": "2-3 sentences brief and honest assessment"
}}

JSON:"""
)


def _evaluate_with_llm(jd_content: str, cv_summary: str, llm_chain) -> Dict:
    try:
        raw = llm_chain.invoke({
            "jd_content": truncate_text(jd_content, 2000),
            "cv_summary": cv_summary,
        })
        return _safe_json_parse(raw)
    except Exception as e:
        logger.error("LLM evaluation failed: %s", e)
        return _empty_evaluation(f"LLM processing error: {str(e)[:120]}")


# =============================================================================
# SCORING GUARDRAILS
# =============================================================================

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
        if not match_row and _skill_key(name) and re.search(rf"(?<![a-z0-9]){re.escape(_norm_text(name))}(?![a-z0-9])", _norm_text(cv_text)):
            match_row = {"name": name, "evidence": name}

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
        if any(_skills_match(skill, cv_skill) for cv_skill in cv_skills) or _skills_match(skill, cv_text):
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
    return "\n\n".join(parts), _first_schema(jd_chunks, "jd_schema")


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
        ok, reasons = _passes_hard_filters(jd_schema, evaluation, cv_profile)

        retrieval_score = float(retrieval_scores.get(source, 0.0))
        rerank_score = round((score * 0.85) + (retrieval_score * 100 * 0.15), 1)
        if not ok:
            rejected[source] = reasons
            evaluation = evaluation.copy()
            evaluation["recommendation"] = "Not a fit"
            evaluation["summary"] = f"{evaluation.get('summary', '')} Filter flags: {'; '.join(reasons)}".strip()
            rerank_score = min(rerank_score, 44.0)
        jd_data = {
            "title": jd_title,
            "weighted_score": score / 100,
            "section_scores": section_scores,
        }
        results.append({
            "cv_source": source,
            "candidate_name": _candidate_name(source, cv_chunks, get_candidate_name),
            "jd_id": jd_id,
            "jd_title": jd_title,
            "similarity_score": round(score, 1),
            "dense_score": round(retrieval_score * 100, 1),
            "bm25_score": round(section_scores.get("required_skills", 0), 1),
            "cross_encoder_score": 0,
            "rerank_score": rerank_score,
            "rerank_method": "schema_score+rag_tiebreak",
            "retrieval_method": retrieval_method,
            "section_scores":   jd_data["section_scores"],   # thêm mới: score từng section
            "evaluation":       evaluation,
            "cv_profile":       cv_profile,
        })

    # Bước 5: sort theo LLM score
    results.sort(key=lambda item: (item.get("rerank_score", 0), item["evaluation"].get("score", 0)), reverse=True)
    return results[:target_k]


# =============================================================================
# DATE / EXPERIENCE HELPERS (giữ nguyên)
# =============================================================================

def _stored_cv_experience(cv_chunks: List[Dict]) -> Dict[str, Any] | None:
    for chunk in cv_chunks:
        if not isinstance(chunk, dict):
            continue
        cv_experience = chunk.get("cv_experience")
        if isinstance(cv_experience, dict):
            return cv_experience
    return None


def _years_from_months(months: int) -> float:
    if months <= 0:
        return 0
    if months < 12:
        return round(months / 12, 2)
    return float(round(months / 12))


def _format_experience_duration(years: float) -> str:
    if years <= 0:
        return "0 years"
    months = round(years * 12)
    if months < 12:
        return f"{months} month" + ("" if months == 1 else "s")
    rounded_years = round(years)
    return f"{rounded_years} year" + ("" if rounded_years == 1 else "s")


def _calculate_experience_years(text: str) -> int:
    intervals = _extract_date_intervals(text)
    if intervals:
        total_months = _merge_and_sum_intervals(intervals)
        if total_months > 0:
            return min(max(round(total_months / 12), 0), 60)

    for pattern in [r"(\d+)\s*\+\s*years", r"(\d+)\s*years", r"over\s*(\d+)\s*years"]:
        m = re.search(pattern, text)
        if m:
            return int(m.group(1))
    return 0


def _extract_date_intervals(text: str) -> List[tuple[int, int]]:
    current       = datetime.now()
    current_index = _month_index(current.year, current.month)
    intervals: List[tuple[int, int]] = []

    month_names = (
        r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
        r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|"
        r"nov(?:ember)?|dec(?:ember)?"
    )
    separator  = r"(?:-|–|—|to|until|through)"
    end_token  = rf"(?:(?:{month_names})\s+\d{{4}}|\d{{1,2}}/\d{{4}}|\d{{4}}|present|current|now|ongoing)"
    patterns   = [
        rf"(?P<start_month>{month_names})\s+(?P<start_year>\d{{4}})\s*{separator}\s*(?P<end>{end_token})",
        rf"(?P<start_month_num>\d{{1,2}})/(?P<start_year_num>\d{{4}})\s*{separator}\s*(?P<end>{end_token})",
        rf"(?P<start_year_only>\d{{4}})\s*{separator}\s*(?P<end>{end_token})",
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            start_index = _parse_start_date(match)
            end_index   = _parse_end_date(match.group("end"), current_index)
            if start_index is None or end_index is None:
                continue
            if end_index < start_index:
                continue
            intervals.append((start_index, end_index))

    return intervals


def _parse_start_date(match: re.Match) -> int | None:
    g = match.groupdict()
    if g.get("start_month") and g.get("start_year"):
        return _month_index(int(match.group("start_year")), _month_name_to_number(match.group("start_month")))
    if g.get("start_month_num") and g.get("start_year_num"):
        month, year = int(match.group("start_month_num")), int(match.group("start_year_num"))
        return _month_index(year, month) if 1 <= month <= 12 else None
    if g.get("start_year_only"):
        return _month_index(int(match.group("start_year_only")), 1)
    return None


def _parse_end_date(value: str, current_index: int) -> int | None:
    value = value.strip().lower()
    if value in {"present", "current", "now", "ongoing"}:
        return current_index

    m = re.fullmatch(
        r"(?P<month>jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
        r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
        r"\s+(?P<year>\d{4})", value, re.IGNORECASE,
    )
    if m:
        return _month_index(int(m.group("year")), _month_name_to_number(m.group("month")))

    m = re.fullmatch(r"(?P<month>\d{1,2})/(?P<year>\d{4})", value)
    if m:
        month, year = int(m.group("month")), int(m.group("year"))
        return _month_index(year, month) if 1 <= month <= 12 else None

    if re.fullmatch(r"\d{4}", value):
        return _month_index(int(value), 12)

    return None


def _month_name_to_number(value: str) -> int:
    months = {
        "jan": 1, "january": 1, "feb": 2, "february": 2,
        "mar": 3, "march": 3, "apr": 4, "april": 4, "may": 5,
        "jun": 6, "june": 6, "jul": 7, "july": 7,
        "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
        "oct": 10, "october": 10, "nov": 11, "november": 11,
        "dec": 12, "december": 12,
    }
    return months[value.strip().lower()]


def _month_index(year: int, month: int) -> int:
    return year * 12 + month


def _merge_and_sum_intervals(intervals: List[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    intervals = sorted(intervals)
    merged    = [list(intervals[0])]
    for start, end in intervals[1:]:
        last = merged[-1]
        if start <= last[1] + 1:
            last[1] = max(last[1], end)
        else:
            merged.append([start, end])
    return sum(end - start + 1 for start, end in merged)


# =============================================================================
# JSON HELPERS
# =============================================================================

def _safe_json_parse(raw: str) -> Dict:
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*?\}", raw)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
        logger.warning("JSON parse failed, returning empty evaluation")
        return _empty_evaluation("Failed to parse LLM output")


def _empty_evaluation(summary: str = "Error") -> Dict:
    return {
        "score":            0,
        "technical_score":  0,
        "experience_score": 0,
        "education_score":  0,
        "fit_score":        0,
        "matched_skills":   [],
        "missing_skills":   [],
        "recommendation":   "Error",
        "summary":          summary,
    }
