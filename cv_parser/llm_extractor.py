from __future__ import annotations
import json
import os
import re
from typing import Any, Dict, List, Optional
from .models import CVStructuredSchema, simple_clean, normalize_text, NormalizedText, _clip
from .chunking import normalize_section_label, cv_markdown_to_chunks

CV_LLM_PARSE_MAX_CHARS = max(4000, int(os.getenv("CV_LLM_PARSE_MAX_CHARS", "32000") or 32000))

CV_SCHEMA_PROMPT = """
Extract the CV into ONE valid JSON object matching the schema.

RULES:
- Output JSON only. No markdown, no explanations.
- Missing data -> null. Do not hallucinate.

TRACEABILITY (CRITICAL):
- Every extracted item (skill, experience, education, project) MUST contain `source_chunk_ids` and exact `evidence` from the text. 
- If no supporting text exists, omit the item entirely.

EXTRACTION STRATEGY FOR ALL CV FORMATS:
1. IGNORE STRICT HEADINGS: Candidates often misplace data (e.g., University details inside "Personal Info" or "Summary"). Scan the entire document holistically for Education, Experience, and Contact details regardless of where they are located.
2. HYPER-SPECIFIC SKILLS: Thoroughly extract ALL explicit Tools, Frameworks, Programming Languages, and specific methodologies (e.g., "Selenium", "Pytest", "Bruno", "React"). DO NOT generalize them into broad categories like "Automation Testing" or "Web Development". Keep the specific names.
3. HIDDEN SKILLS: Extract implicit skills hidden inside "Work Experience" or "Projects" descriptions. Consolidate ALL found skills (both explicit and implicit) into the top-level "skills" array.
4. NON-STANDARD EXPERIENCE: Capture work experience even if it lacks explicit dates or standard company names (e.g., "Freelance", "Scientific Research", "Open Source"). Group them logically by title and available context.
5. PRESERVE SCHEMA: Infer skill types only when directly supported by context. Preserve the JSON schema structure exactly.

CV:
{cv_text}

SCHEMA:
{schema_definition}
"""

def _extract_json_object(text: str) -> Dict[str, Any]:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", simple_clean(text), flags=re.I)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
        raise ValueError("LLM JSON response must be an object")
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            raise ValueError("LLM did not return a valid JSON object")
        try:
            parsed = json.loads(m.group(0))
            if isinstance(parsed, dict):
                return parsed
            raise ValueError("LLM JSON response must be an object")
        except json.JSONDecodeError:
            raise ValueError("LLM did not return a valid JSON object")


def _llm_call(prompt: str) -> str:
    try:
        from llm_provider import generate_answer
    except Exception as exc:
        raise RuntimeError("LLM provider is not available for CV parsing") from exc
    answer = str(generate_answer(prompt) or "")
    if not answer.strip():
        raise ValueError("LLM returned an empty CV parser response")
    return answer


def _normalize_cv_schema(data: Dict[str, Any]) -> Dict[str, Any]:
    raw = data if isinstance(data, dict) else {}
    return CVStructuredSchema.model_validate(raw).model_dump(by_alias=True)

def _build_llm_input_from_chunks(chunks: List[Dict[str, Any]], fallback_text: str | NormalizedText) -> str:
    """Build a compact, section-aware LLM input from existing chunks."""
    normalized = normalize_text(fallback_text)
    if not chunks:
        return _clip(normalized, CV_LLM_PARSE_MAX_CHARS)

    priority = {"profile": 0, "summary": 0, "skills": 1, "experience": 2,
                "projects": 3, "education": 4, "certifications": 5, "languages": 6, "unknown": 7}
    section_limits = {"profile": 1200, "summary": 1200, "skills": 2200, "experience": 4200,
                      "projects": 2800, "education": 1600, "certifications": 1000, "languages": 800, "unknown": 900}

    used: Dict[str, int] = {}
    parts: List[str] = []
    total = 0

    for chunk in sorted(chunks, key=lambda c: (priority.get(str(c.get("section") or "unknown"), 8), int(c.get("chunk_index", 0) or 0))):
        section = normalize_section_label(chunk.get("section", "unknown"))
        text    = simple_clean(chunk.get("text", ""))
        if not text:
            continue
        remaining = section_limits.get(section, 900) - used.get(section, 0)
        if remaining <= 0:
            continue
        text = _clip(text, min(remaining, 1800))
        raw_heading = simple_clean(chunk.get("raw_heading") or "")
        chunk_id = simple_clean(chunk.get("chunk_id") or f"chunk_{chunk.get('chunk_index', len(parts))}")
        header = f"[chunk_id={chunk_id} | section={section}]"
        if raw_heading and raw_heading.lower() != section:
            header = f"[chunk_id={chunk_id} | section={section} | heading={raw_heading[:80]}]"
        part = simple_clean(f"{header}\n{text}")
        if total + len(part) + 2 > CV_LLM_PARSE_MAX_CHARS:
            break
        parts.append(part)
        total += len(part) + 2
        used[section] = used.get(section, 0) + len(text)

    compact = simple_clean("\n\n".join(parts))
    return compact or _clip(normalized, CV_LLM_PARSE_MAX_CHARS)


def _build_llm_input(cv_text: str | NormalizedText) -> str:
    """Build a compact, section-aware LLM input from raw CV text."""
    normalized = normalize_text(cv_text)
    chunks = cv_markdown_to_chunks(normalized, {}, min_chars=40, max_chars=1800)
    return _build_llm_input_from_chunks(chunks, normalized)


def extract_cv_schema(cv_text: str | NormalizedText, *, chunks: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
    """Parse CV text → structured schema dict via LLM + Pydantic normalization."""
    normalized = normalize_text(cv_text)
    llm_input = _build_llm_input_from_chunks(chunks, normalized) if chunks is not None else _build_llm_input(normalized)
    schema_definition = json.dumps(CVStructuredSchema.model_json_schema(by_alias=True), ensure_ascii=False)
    raw_json = _extract_json_object(
        _llm_call(CV_SCHEMA_PROMPT.replace("{cv_text}", llm_input).replace("{schema_definition}", schema_definition))
    )
    return _normalize_cv_schema(raw_json)