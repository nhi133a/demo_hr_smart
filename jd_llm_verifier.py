import concurrent.futures
import json
import logging
import re
from typing import Any, Dict, List

from jd_cross_encoder import _truncate_for_cross_encoder
from jd_matcher_config import (
    LLM_REQUIREMENT_VERIFIER_ENABLED,
    LLM_REQUIREMENT_VERIFIER_MAX_CHARS,
    LLM_REQUIREMENT_VERIFIER_THRESHOLD,
    LLM_REQUIREMENT_VERIFIER_TIMEOUT_SECONDS,
)

logger = logging.getLogger(__name__)


def _extract_json_object(raw: str) -> Dict:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", str(raw or "").strip(), flags=re.IGNORECASE)
    candidates = [text]
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        candidates.insert(0, match.group(0))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            continue
    return {}


def _llm_requirement_prompt(unit: Dict, jd_schema: Any, chunks: List[Dict]) -> str:
    title = str(jd_schema.get("job_title") if isinstance(jd_schema, dict) else getattr(jd_schema, "job_title", "") or "").strip()
    evidence_parts = []
    total = 0
    for idx, chunk in enumerate(chunks, 1):
        text = _truncate_for_cross_encoder(chunk.get("text", ""), 900)
        if not text:
            continue
        item = f"[CV_EVIDENCE_{idx} | section={chunk.get('section', 'unknown')}]\n{text}"
        if total + len(item) > LLM_REQUIREMENT_VERIFIER_MAX_CHARS:
            break
        evidence_parts.append(item)
        total += len(item)

    return f"""You are verifying whether a CV provides practical evidence for a job requirement.

Job title: {title or "unknown"}
Requirement: {unit.get("name", "")}
Requirement type: {unit.get("source", "")}
JD evidence: {unit.get("evidence", "")}

CV evidence snippets:
{chr(10).join(evidence_parts) if evidence_parts else "(no useful CV evidence)"}

Rules:
- Match only when CV evidence shows practical capability through work, projects, study, certification, concrete usage, responsibility, or result.
- Do not match from vague interest, willingness, keyword-only mentions, or unsupported claims.
- Do not invent evidence outside the snippets.
- Distinguish basic exposure from deeper capability:
  - Mentioning a tool/technology proves only exposure or usage.
  - Match design/ownership/leadership/analysis/strategy only when the CV shows related decisions, responsibility, outputs, or outcomes.
  - For tool-specific requirements, require the exact tool or a direct equivalent.
- Semantic matching is allowed when the evidence demonstrates the same practical capability using different wording.
- For intern/fresher roles, academic or personal projects can be valid evidence.
- For higher-level roles, require stronger evidence such as ownership, repeated use, impact, scale, or professional responsibility.
- Set confidence from 0.0 to 1.0:
  - High confidence for direct and explicit evidence.
  - Medium confidence for practical but indirect evidence.
  - Low confidence for vague, weak, or keyword-only evidence.
- Return matched=true only when confidence meets the threshold.

Return JSON only:
{{
  "matched": true,
  "confidence": 0.85,
  "reason": "",
  "evidence_quote": ""
}}"""


def _generate_answer_with_timeout(prompt: str, timeout_seconds: int) -> str:
    try:
        from llm_provider import generate_answer
    except Exception:
        return ""

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(generate_answer, prompt)
    try:
        return str(future.result(timeout=timeout_seconds) or "")
    except concurrent.futures.TimeoutError:
        future.cancel()
        return ""
    except Exception:
        return ""
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def _llm_verify_requirement(unit: Dict, jd_schema: Any, chunks: List[Dict]) -> Dict | None:
    if not LLM_REQUIREMENT_VERIFIER_ENABLED or not chunks:
        return None
    try:
        raw = _generate_answer_with_timeout(
            _llm_requirement_prompt(unit, jd_schema, chunks),
            LLM_REQUIREMENT_VERIFIER_TIMEOUT_SECONDS,
        )
        if not raw:
            return None
        data = _extract_json_object(raw)
    except Exception as exc:
        logger.warning("LLM requirement verifier unavailable: %s", exc)
        return None

    try:
        confidence = max(0.0, min(1.0, float(data.get("confidence", 0.0))))
    except Exception:
        confidence = 0.0
    matched = bool(data.get("matched")) and confidence >= LLM_REQUIREMENT_VERIFIER_THRESHOLD
    evidence_quote = re.sub(r"\s+", " ", str(data.get("evidence_quote") or "")).strip()
    reason = re.sub(r"\s+", " ", str(data.get("reason") or "")).strip()
    return {
        "requirement": unit.get("name", ""),
        "source": unit.get("source", ""),
        "importance": unit.get("importance", ""),
        "alternative_group": unit.get("alternative_group", ""),
        "status": "matched" if matched else "missing",
        "confidence": round(confidence, 4),
        "evidence": evidence_quote[:500] if matched else "",
        "cv_section": "llm_verified",
        "method": "llm_requirement_verifier",
        "reason": reason[:300],
    }
