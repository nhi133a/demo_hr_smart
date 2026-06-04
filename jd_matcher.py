import re
import logging
from typing import Any, Dict, List

from jd_candidate_retrieval import _retrieval_evidence, _retrieve_candidate_sources
from jd_cross_encoder import (
    _cross_encoder_chunk_text,
    _cross_encoder_probability,
    _cross_encoder_score,
    _get_cross_encoder_model,
    _truncate_for_cross_encoder,
)
from jd_llm_verifier import _llm_verify_requirement
from jd_matcher_config import (
    CROSS_ENCODER_ENABLED,
    CROSS_ENCODER_MAX_CHUNK_CHARS,
    CROSS_ENCODER_WEIGHT,
    LLM_REQUIREMENT_VERIFIER_ENABLED,
    LLM_REQUIREMENT_VERIFIER_MAX_REQUIREMENTS,
    LLM_REQUIREMENT_VERIFIER_MIN_CANDIDATE_SCORE,
    MATCH_CANDIDATE_POOL_LIMIT,
    MATCH_DEEP_RERANK_LIMIT,
    MATCH_DEEP_RERANK_MULTIPLIER,
    MATCH_MODE,
    MATCH_RETRIEVAL_MIN_CANDIDATES,
    MATCH_RETRIEVAL_MULTIPLIER,
    REQUIREMENT_EVIDENCE_ENABLED,
    REQUIREMENT_EVIDENCE_MAX_CHUNKS,
    REQUIREMENT_EVIDENCE_MAX_REQUIREMENTS,
    REQUIREMENT_EVIDENCE_THRESHOLD,
)
from jd_skill_matching import (
    _capability_match,
    _dedupe_keep_order,
    _match_rows,
    _row_name,
    _skill_key,
    _skill_text_match,
    _skills_match,
    _token_set,
)
from jd_store import _normalize_jd_schema, count_indexed_jds, get_jd_chunks

logger = logging.getLogger(__name__)

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


def _tag_rows(rows: List[Dict], category: str) -> List[Dict]:
    tagged = []
    for row in rows:
        item = dict(row)
        item.setdefault("category", category)
        tagged.append(item)
    return tagged


def _add_cv_capability(rows: List[Dict], item: Any, category: str, source: str, *, evidence: str = "") -> None:
    if isinstance(item, str):
        row = {"name": item}
    elif isinstance(item, dict):
        row = dict(item)
    else:
        return
    name = str(row.get("name") or row.get("level") or row.get("domain") or row.get("title") or "").strip()
    if not name:
        return
    row["name"] = name
    row.setdefault("category", category)
    row.setdefault("source", source)
    if evidence and not row.get("evidence"):
        row["evidence"] = evidence
    rows.append(row)


def _dedupe_rows(rows: List[Dict]) -> List[Dict]:
    seen = set()
    result = []
    for row in rows:
        key = (_skill_key(_row_name(row)), str(row.get("category") or ""))
        if key[0] and key not in seen:
            seen.add(key)
            result.append(row)
    return result


def _cv_skill_rows(chunks: List[Dict], cv_schema: Dict) -> List[Dict]:
    rows = []
    for row in _rows(cv_schema, "skills"):
        _add_cv_capability(rows, row, "skill", "cv_schema.skills")
    for row in _rows(cv_schema, "soft_skills"):
        _add_cv_capability(rows, row, "soft_skill", "cv_schema.soft_skills")
    for row in _rows(cv_schema, "certifications"):
        _add_cv_capability(rows, row, "certification", "cv_schema.certifications")
    for row in _rows(cv_schema, "languages"):
        _add_cv_capability(rows, row, "language", "cv_schema.languages")

    for project in cv_schema.get("projects", []) if isinstance(cv_schema, dict) else []:
        if not isinstance(project, dict):
            continue
        evidence = str(project.get("evidence") or project.get("description") or project.get("name") or "")
        for tech in project.get("technologies", []) if isinstance(project.get("technologies"), list) else []:
            _add_cv_capability(
                rows,
                {
                    "name": tech,
                    "evidence": evidence,
                    "source_chunk_ids": project.get("source_chunk_ids") or [],
                },
                "project_technology",
                "cv_schema.projects.technologies",
            )

    for exp in cv_schema.get("experience", []) if isinstance(cv_schema, dict) else []:
        if not isinstance(exp, dict):
            continue
        evidence = str(exp.get("evidence") or "")
        for key in ("domain", "title"):
            _add_cv_capability(
                rows,
                {
                    "name": exp.get(key),
                    "evidence": evidence,
                    "source_chunk_ids": exp.get("source_chunk_ids") or [],
                },
                "experience",
                f"cv_schema.experience.{key}",
            )

    if rows:
        return _dedupe_rows(rows)



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
    return [{"name": skill, "confidence": "medium", "category": "skill"} for skill in _dedupe_keep_order(fallback)]


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


def _requirement_units(jd_schema: Dict) -> List[Dict]:
    def is_label(value: str) -> bool:
        return bool(re.match(r"^\s*(?:job\s+title|title|position|role)\s*:", str(value or ""), re.I))

    def clean_evidence(value: str) -> str:
        text = re.sub(r"\s*\([^)]*(?:type|confidence|evidence|importance)\s*=.*?\)\s*$", "", str(value or ""), flags=re.I)
        return re.sub(r"\s+", " ", text).strip()

    def enrich_alternative_groups(items: List[Dict]) -> List[Dict]:
        alternative_by_name: Dict[tuple[str, str], str] = {}
        for key, source, _ in (
            ("required_skills", "required_skill", "required"),
            ("preferred_skills", "preferred_skill", "preferred"),
        ):
            for row in _rows(jd_schema, key):
                name = _row_name(row)
                evidence = clean_evidence(str(row.get("evidence") or ""))
                if name and evidence and re.search(r"\bor\b", evidence, re.I):
                    alternative_by_name[(source, _skill_key(name))] = _skill_key(evidence)
        for item in items:
            if item.get("alternative_group"):
                continue
            alt = alternative_by_name.get((str(item.get("source") or ""), _skill_key(str(item.get("name") or ""))))
            if alt:
                item["alternative_group"] = alt
        return items

    schema_units = jd_schema.get("requirement_units") if isinstance(jd_schema, dict) else []
    units: List[Dict] = []
    seen = set()
    for unit in schema_units if isinstance(schema_units, list) else []:
        if not isinstance(unit, dict):
            continue
        name = str(unit.get("name") or "").strip()
        evidence = clean_evidence(str(unit.get("evidence") or ""))
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
        dedupe_key = (source, _skill_key(name))
        if not name or is_label(name) or is_label(evidence) or dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        units.append(
            {
                "name": name,
                "source": source,
                "importance": importance or source,
                "evidence": evidence,
                "alternative_group": str(unit.get("alternative_group") or "").strip()
                or (_skill_key(evidence) if re.search(r"\bor\b", evidence, re.I) else ""),
                "row": unit,
            }
        )
        if len(units) >= REQUIREMENT_EVIDENCE_MAX_REQUIREMENTS:
            return units
    if units:
        return enrich_alternative_groups(units)

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
            evidence = clean_evidence(str(row.get("evidence") or ""))
            if not name or is_label(name) or is_label(evidence) or dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            units.append(
                {
                    "name": name,
                    "source": source,
                    "importance": str(row.get("importance") or importance),
                    "evidence": str(row.get("evidence") or ""),
                    "alternative_group": str(row.get("alternative_group") or "").strip()
                    or (_skill_key(evidence) if re.search(r"\bor\b", evidence, re.I) else ""),
                    "row": row,
                }
            )
            if len(units) >= REQUIREMENT_EVIDENCE_MAX_REQUIREMENTS:
                return units
    return enrich_alternative_groups(units)


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


def _apply_llm_requirement_verifier(
    best_by_requirement: Dict[tuple[str, str], Dict],
    units: List[Dict],
    jd_schema: Dict,
    cv_chunks: List[Dict],
    cv_rows: List[Dict],
    *,
    allow_llm: bool = False,
) -> None:
    if not allow_llm or MATCH_MODE != "deep" or not LLM_REQUIREMENT_VERIFIER_ENABLED:
        return
    verified = 0
    for unit in units:
        if verified >= LLM_REQUIREMENT_VERIFIER_MAX_REQUIREMENTS:
            break
        key = (str(unit.get("source") or ""), _skill_key(str(unit.get("name") or "")))
        current = best_by_requirement.get(key)
        if current and current.get("status") == "matched" and float(current.get("confidence") or 0) >= 0.90:
            continue
        chunks = _candidate_requirement_chunks(unit, cv_chunks, cv_rows)
        if not chunks or float(chunks[0].get("candidate_score") or 0.0) < LLM_REQUIREMENT_VERIFIER_MIN_CANDIDATE_SCORE:
            continue
        result = _llm_verify_requirement(unit, jd_schema, chunks)
        verified += 1
        if not result:
            continue
        if not current or float(result.get("confidence") or 0) >= float(current.get("confidence") or 0):
            best_by_requirement[key] = result


def _requirement_evidence_matches(
    jd_schema: Dict,
    cv_chunks: List[Dict],
    *,
    allow_llm: bool = False,
) -> tuple[List[Dict], List[Dict]]:
    if not REQUIREMENT_EVIDENCE_ENABLED:
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
        match_row = next(
            (
                row
                for row in cv_rows
                if _capability_match(
                    {"name": name, "category": str(unit.get("source") or "")},
                    row,
                )
            ),
            None,
        )
        if not match_row:
            match_row = _skill_text_match(name, cv_text)
        if match_row:
            key = (str(unit.get("source") or ""), _skill_key(name))
            best_by_requirement[key] = {
                "requirement": name,
                "source": unit.get("source", ""),
                "importance": unit.get("importance", ""),
                "alternative_group": unit.get("alternative_group", ""),
                "status": "matched",
                "confidence": 0.95,
                "evidence": str(match_row.get("evidence") or match_row.get("source") or name)[:500],
                "cv_section": str(match_row.get("section") or "schema"),
                "method": "schema_or_raw_evidence",
            }

    work_items = []
    pairs = []
    model = _get_cross_encoder_model() if _cross_encoder_allowed() else None
    if model is not None:
        for unit in units:
            requirement_text = _truncate_for_cross_encoder(_requirement_prompt(unit, jd_schema), 700)
            for chunk in _candidate_requirement_chunks(unit, cv_chunks, cv_rows):
                work_items.append((unit, chunk))
                pairs.append([requirement_text, _truncate_for_cross_encoder(chunk["text"], CROSS_ENCODER_MAX_CHUNK_CHARS)])

        if pairs:
            try:
                predictions = model.predict(pairs, show_progress_bar=False)
            except TypeError:
                predictions = model.predict(pairs)
            except Exception as exc:
                logger.warning("Requirement evidence scoring failed: %s", exc)
                predictions = []

            for (unit, chunk), raw_score in zip(work_items, list(predictions)):
                confidence = _cross_encoder_probability(raw_score)
                key = (str(unit.get("source") or ""), _skill_key(str(unit.get("name") or "")))
                current = best_by_requirement.get(key)
                if not current or confidence > current["confidence"]:
                    best_by_requirement[key] = {
                        "requirement": unit.get("name", ""),
                        "source": unit.get("source", ""),
                        "importance": unit.get("importance", ""),
                        "alternative_group": unit.get("alternative_group", ""),
                        "status": "matched" if confidence >= REQUIREMENT_EVIDENCE_THRESHOLD else "missing",
                        "confidence": round(confidence, 4),
                        "evidence": chunk["text"][:500] if confidence >= REQUIREMENT_EVIDENCE_THRESHOLD else "",
                        "cv_section": chunk.get("section", "unknown"),
                        "method": "cross_encoder_requirement_evidence",
                    }

    _apply_llm_requirement_verifier(best_by_requirement, units, jd_schema, cv_chunks, cv_rows, allow_llm=allow_llm)

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
                    "alternative_group": unit.get("alternative_group", ""),
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
    grouped: Dict[str, List[Dict]] = {}
    standalone: List[Dict] = []
    for item in selected:
        group = str(item.get("alternative_group") or "")
        if group:
            grouped.setdefault(group, []).append(item)
        else:
            standalone.append(item)

    matched: List[str] = []
    missing: List[str] = []
    total = len(standalone) + len(grouped)
    matched_count = 0

    for item in standalone:
        if item.get("status") == "matched":
            matched_count += 1
            matched.append(str(item.get("requirement") or ""))
        else:
            missing.append(str(item.get("requirement") or ""))

    for rows in grouped.values():
        matched_rows = [row for row in rows if row.get("status") == "matched"]
        if matched_rows:
            matched_count += 1
            matched.append(str(matched_rows[0].get("requirement") or ""))
        else:
            missing.append(" or ".join(str(row.get("requirement") or "") for row in rows if row.get("requirement")))

    return matched_count / max(1, total), _dedupe_keep_order(matched), _dedupe_keep_order(missing)


def _apply_requirement_evidence_refinement(
    jd_schema: Dict,
    cv_chunks: List[Dict],
    evaluation: Dict,
    section_scores: Dict,
    evidence: List[Dict],
    *,
    allow_llm: bool = False,
) -> tuple[float, Dict, Dict, List[Dict]]:
    details, requirement_evidence = _requirement_evidence_matches(jd_schema, cv_chunks, allow_llm=allow_llm)
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


def _dedupe_sources(sources: List[str]) -> List[str]:
    seen = set()
    result = []
    for source in sources:
        if source and source not in seen:
            seen.add(source)
            result.append(source)
    return result


def _candidate_pool_limit(total_sources: int, top_k: int) -> int:
    target = max(top_k * MATCH_RETRIEVAL_MULTIPLIER, top_k, MATCH_RETRIEVAL_MIN_CANDIDATES)
    return min(max(1, total_sources), target, MATCH_CANDIDATE_POOL_LIMIT)


def _deep_rerank_limit(total_results: int, top_k: int) -> int:
    if MATCH_MODE == "fast":
        return 0
    return min(max(1, total_results), max(top_k * MATCH_DEEP_RERANK_MULTIPLIER, 10), MATCH_DEEP_RERANK_LIMIT)


def _requirement_refinement_allowed() -> bool:
    return MATCH_MODE in {"balanced", "deep"} and REQUIREMENT_EVIDENCE_ENABLED


def _cross_encoder_allowed() -> bool:
    return MATCH_MODE == "deep" and CROSS_ENCODER_ENABLED


def _score_candidate(jd_schema: Dict, jd_content: str, cv_chunks: List[Dict]) -> tuple[float, Dict, Dict, List[Dict], Dict]:
    cv_schema = _first_schema(cv_chunks, "cv_schema")
    cv_rows = _cv_skill_rows(cv_chunks, cv_schema)
    cv_text = _cv_text(cv_chunks)

    required_rows = _tag_rows(_rows(jd_schema, "required_skills"), "skill")
    preferred_rows = _tag_rows(_rows(jd_schema, "preferred_skills"), "skill")
    competency_rows = _tag_rows(_rows(jd_schema, "competencies"), "competency") + _tag_rows(_rows(jd_schema, "soft_skills"), "soft_skill")

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


def _summary(score: float, matched_required: List[str], missing_required: List[str], cv_months: int, required_months: int) -> str:
    matched = ", ".join(matched_required[:5]) if matched_required else "no required skills"
    missing = ", ".join(missing_required[:5]) if missing_required else "no major required skill gaps"
    exp = ""
    if required_months:
        exp = f" Experience: {_format_months(cv_months)} against {_format_months(required_months)} required."
    return f"{_recommendation(score)} based on schema match: matched {matched}; missing {missing}.{exp}"


def _passes_hard_filters(
    jd_schema: Dict,
    evaluation: Dict,
    cv_profile: Dict,
    *,
    include_skill_filter: bool = True,
) -> tuple[bool, List[str]]:
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
    if include_skill_filter and filters["required_skills"] and threshold > 0 and required_total and required_ratio < threshold:
        reasons.append(f"required skill match rate {required_ratio:.0%} below threshold")
    return not reasons, reasons


def _prefilter_sources(
    jd_schema: Dict,
    jd_content: str,
    sources: List[str],
    get_chunks_by_source_for_matching,
) -> tuple[List[str], Dict[str, List[Dict]], Dict[str, List[str]]]:
    passed = []
    chunks_by_source: Dict[str, List[Dict]] = {}
    rejected: Dict[str, List[str]] = {}

    for source in sources:
        cv_chunks = get_chunks_by_source_for_matching(source)
        if not cv_chunks:
            rejected[source] = ["no CV chunks found"]
            continue
        chunks_by_source[source] = cv_chunks
        score, evaluation, _, _, cv_profile = _score_candidate(jd_schema, jd_content, cv_chunks)
        ok, reasons = _passes_hard_filters(jd_schema, evaluation, cv_profile, include_skill_filter=False)
        if ok:
            passed.append(source)
        else:
            rejected[source] = reasons
    return passed, chunks_by_source, rejected


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


def _candidate_sources_for_matching(
    jd_schema: Dict,
    jd_content: str,
    *,
    top_k: int,
    all_sources: List[str],
    source_whitelist: List[str] | None,
    search_similar_cv_chunks,
) -> tuple[List[str], Dict[str, float], Dict[str, List[Dict]], str]:
    candidate_sources, retrieval_scores, retrieval_hits, retrieval_method = _retrieve_candidate_sources(
        jd_schema,
        jd_content,
        top_k=top_k,
        source_whitelist=source_whitelist,
        search_similar_cv_chunks=search_similar_cv_chunks,
    )
    limit = _candidate_pool_limit(len(all_sources), top_k)
    if candidate_sources:
        return _dedupe_sources(candidate_sources)[:limit], retrieval_scores, retrieval_hits, retrieval_method
    method = f"{retrieval_method}+schema_fallback" if retrieval_method else "schema_fallback"
    return all_sources[:limit], {}, {}, method


def _fast_score_candidates(
    candidate_sources: List[str],
    *,
    jd_id: str,
    jd_title: str,
    jd_schema: Dict,
    jd_content: str,
    retrieval_scores: Dict[str, float],
    retrieval_hits: Dict[str, List[Dict]],
    retrieval_method: str,
    get_candidate_name,
    get_chunks_by_source_for_matching,
    chunks_by_source: Dict[str, List[Dict]] | None = None,
) -> tuple[List[Dict], Dict[str, List[Dict]]]:
    results = []
    chunks_by_source = dict(chunks_by_source or {})
    for source in candidate_sources:
        cv_chunks = chunks_by_source.get(source) or get_chunks_by_source_for_matching(source)
        if not cv_chunks:
            continue
        chunks_by_source[source] = cv_chunks
        score, evaluation, section_scores, evidence, cv_profile = _score_candidate(jd_schema, jd_content, cv_chunks)
        ok, reasons = _passes_hard_filters(jd_schema, evaluation, cv_profile)
        retrieval_score = float(retrieval_scores.get(source, 0.0))
        rerank_score, rerank_method = _combine_rerank_score(score, retrieval_score, 0.0)
        if not ok:
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
            "cross_encoder_score": 0.0,
            "rerank_score": rerank_score,
            "rerank_method": rerank_method,
            "retrieval_method": retrieval_method,
            "section_scores": section_scores,
            "evaluation": evaluation,
            "cv_profile": cv_profile,
            "match_evidence": (evidence + _retrieval_evidence(retrieval_hits.get(source, [])))[:8],
            "_schema_score": score,
            "_schema_evidence": evidence,
        })
    return results, chunks_by_source


def _deep_rerank_results(
    results: List[Dict],
    *,
    jd_schema: Dict,
    jd_content: str,
    top_k: int,
    allow_llm: bool,
    retrieval_scores: Dict[str, float],
    retrieval_hits: Dict[str, List[Dict]],
    chunks_by_source: Dict[str, List[Dict]],
) -> List[Dict]:
    limit = _deep_rerank_limit(len(results), top_k)
    if limit <= 0:
        return results

    ranked = sorted(results, key=lambda item: (item.get("rerank_score", 0), item["evaluation"].get("score", 0)), reverse=True)
    deep_sources = {item["cv_source"] for item in ranked[:limit]}
    updated = []
    for item in results:
        source = item["cv_source"]
        if source not in deep_sources:
            updated.append(item)
            continue

        cv_chunks = chunks_by_source.get(source) or []
        if not cv_chunks:
            updated.append(item)
            continue

        score = float(item.get("_schema_score", item.get("similarity_score", 0.0)))
        evaluation = item["evaluation"]
        section_scores = item["section_scores"]
        evidence = list(item.get("_schema_evidence") or [])
        if _requirement_refinement_allowed():
            score, evaluation, section_scores, evidence = _apply_requirement_evidence_refinement(
                jd_schema,
                cv_chunks,
                evaluation,
                section_scores,
                evidence,
                allow_llm=allow_llm,
            )

        retrieval_score = float(retrieval_scores.get(source, 0.0))
        cross_score = 0.0
        cross_evidence: List[Dict] = []
        if _cross_encoder_allowed():
            cross_score, cross_evidence = _cross_encoder_score(
                jd_content,
                cv_chunks,
                retrieval_hits=retrieval_hits.get(source, []),
            )

        rerank_score, rerank_method = _combine_rerank_score(score, retrieval_score, cross_score)
        ok, reasons = _passes_hard_filters(jd_schema, evaluation, item.get("cv_profile", {}))
        if not ok:
            evaluation = evaluation.copy()
            evaluation["recommendation"] = "Not a fit"
            evaluation["summary"] = f"{evaluation.get('summary', '')} Filter flags: {'; '.join(reasons)}".strip()
            rerank_score = min(rerank_score, 44.0)

        changed = item.copy()
        changed.update({
            "similarity_score": round(score, 1),
            "bm25_score": round(section_scores.get("required_skills", 0), 1),
            "cross_encoder_score": cross_score,
            "rerank_score": rerank_score,
            "rerank_method": rerank_method,
            "section_scores": section_scores,
            "evaluation": evaluation,
            "match_evidence": (evidence + cross_evidence + _retrieval_evidence(retrieval_hits.get(source, [])))[:8],
            "_schema_score": score,
            "_schema_evidence": evidence,
        })
        updated.append(changed)
    return updated


def _public_results(results: List[Dict], target_k: int) -> List[Dict]:
    results.sort(key=lambda item: (item.get("rerank_score", 0), item["evaluation"].get("score", 0)), reverse=True)
    public = []
    for item in results[:target_k]:
        clean = item.copy()
        clean.pop("_schema_score", None)
        clean.pop("_schema_evidence", None)
        public.append(clean)
    return public


def match_jd_to_cvs(
    jd_id: str,
    top_k: int = 5,
    *,
    use_llm: bool = False,
    source_whitelist: List[str] | None = None,
    applicants_only: bool = False,
) -> List[Dict]:
    from mongo_utils import (
        get_candidate_name,
        get_candidate_profile,
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
    if applicants_only:
        scoped_sources = []
        for source in all_sources:
            profile = get_candidate_profile(source)
            applied_ids = profile.get("applied_jd_ids")
            if isinstance(applied_ids, str):
                applied_ids = [applied_ids]
            if profile.get("applied_jd_id") == jd_id or jd_id in (applied_ids or []):
                scoped_sources.append(source)
        all_sources = scoped_sources
    if not all_sources:
        message = "No CVs applied to this JD" if applicants_only else "No CVs available in the selected scope"
        return _error_result(jd_id, jd_title, message)

    filtered_sources, preloaded_chunks, rejected_sources = _prefilter_sources(
        jd_schema,
        jd_content,
        all_sources,
        get_chunks_by_source_for_matching,
    )
    if not filtered_sources:
        return _error_result(jd_id, jd_title, "No CVs passed the hard filters", method="hard_filter")

    candidate_sources, retrieval_scores, retrieval_hits, retrieval_method = _candidate_sources_for_matching(
        jd_schema,
        jd_content,
        top_k=target_k,
        all_sources=filtered_sources,
        source_whitelist=filtered_sources,
        search_similar_cv_chunks=search_similar_cv_chunks,
    )

    results, chunks_by_source = _fast_score_candidates(
        candidate_sources,
        jd_id=jd_id,
        jd_title=jd_title,
        jd_schema=jd_schema,
        jd_content=jd_content,
        retrieval_scores=retrieval_scores,
        retrieval_hits=retrieval_hits,
        retrieval_method=retrieval_method,
        get_candidate_name=get_candidate_name,
        get_chunks_by_source_for_matching=get_chunks_by_source_for_matching,
        chunks_by_source=preloaded_chunks,
    )

    if not results:
        return _error_result(jd_id, jd_title, "No candidate CV chunks found")

    results = _deep_rerank_results(
        results,
        jd_schema=jd_schema,
        jd_content=jd_content,
        top_k=target_k,
        allow_llm=use_llm,
        retrieval_scores=retrieval_scores,
        retrieval_hits=retrieval_hits,
        chunks_by_source=chunks_by_source,
    )
    return _public_results(results, target_k)
