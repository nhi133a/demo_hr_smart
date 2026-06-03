import logging
import math
import re
from typing import Any, Dict, List

from jd_matcher_config import (
    CROSS_ENCODER_ENABLED,
    CROSS_ENCODER_LOCAL_FILES_ONLY,
    CROSS_ENCODER_MAX_CHUNK_CHARS,
    CROSS_ENCODER_MAX_CHUNKS,
    CROSS_ENCODER_MAX_JD_CHARS,
    CROSS_ENCODER_MODEL,
    CROSS_ENCODER_TOP_AVG,
)

logger = logging.getLogger(__name__)

_cross_encoder_model = None
_cross_encoder_load_attempted = False


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
