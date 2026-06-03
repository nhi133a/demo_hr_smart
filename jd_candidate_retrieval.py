import logging
from typing import Any, Dict, List

from bedrock_utils import get_embedding
from jd_matcher_config import MATCH_RETRIEVAL_MIN_CANDIDATES, MATCH_RETRIEVAL_MULTIPLIER
from jd_skill_matching import _row_name

logger = logging.getLogger(__name__)


def _rows(schema: Any, key: str) -> List[Any]:
    value = schema.get(key) if isinstance(schema, dict) else getattr(schema, key, [])
    return [item for item in value if _row_name(item)]


def _retrieval_query_text(jd_schema: Any, jd_content: str) -> str:
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

    exp = jd_schema.get("experience") if isinstance(jd_schema, dict) else getattr(jd_schema, "experience", None)
    min_months = exp.get("min_months") if isinstance(exp, dict) else getattr(exp, "min_months", 0)
    if min_months:
        parts.append(f"[EXPERIENCE]\nminimum_months: {min_months}")

    context = jd_schema.get("work_context") if isinstance(jd_schema, dict) else getattr(jd_schema, "work_context", None)
    context_lines = []
    for key in ("role", "seniority", "industry", "company_type"):
        value = context.get(key) if isinstance(context, dict) else getattr(context, key, None)
        if value:
            context_lines.append(f"{key}: {value}")
    for key in ("domains", "platforms", "tools", "processes"):
        values = context.get(key) if isinstance(context, dict) else getattr(context, key, [])
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
    jd_schema: Any,
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
    limit = max(top_k * MATCH_RETRIEVAL_MULTIPLIER, top_k, MATCH_RETRIEVAL_MIN_CANDIDATES)
    return ranked_sources[:limit], scores, hits_by_source, "rag_vector"
