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
from typing import Any, Dict, List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

from bedrock_utils import get_embedding
from jd_store import (
    count_indexed_jds,
    get_jd_chunks,
    search_similar_jd_skills,
    search_similar_jds,
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
    experience_years  = _calculate_experience_years(full_text)
    experience_level  = (
        "Senior"       if experience_years >= 5 else
        "Mid-level"    if experience_years >= 2 else
        "Junior/Intern"
    )

    return {
        "skills":          found_skills[:15],
        "total_skills":    len(found_skills),
        "experience_years": experience_years,
        "experience_level": experience_level,
    }


def build_cv_summary(cv_chunks: List[Dict], cv_profile: Dict) -> str:
    relevant    = extract_relevant_sections(cv_chunks)
    profile_str = (
        f"[CV PRE-PROCESSED SUMMARY]\n"
        f"Experience Level : {cv_profile['experience_level']} ({cv_profile['experience_years']} years)\n"
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

NO_EXPERIENCE_REQUIRED_PATTERNS = [
    r"\bchua\s+can\s+kinh\s+nghiem\b",
    r"\bchua\s+yeu\s+cau\s+kinh\s+nghiem\b",
    r"\bkhong\s+(?:yeu\s+cau|can|doi\s+hoi)\s+kinh\s+nghiem\b",
    r"no\s+experience\s+(required|needed)",
    r"no\s+(professional\s+|work\s+)?experience\s+(required|needed)",
    r"entry\s*level",
    r"fresher",
    r"intern(ship)?",
]

EXPERIENCE_YEAR_PATTERNS = [
    r"(?:at\s+least|min(?:imum)?|toi\s+thieu|tu)\s*(\d+)\s*\+?\s*(?:years?|yrs?|nam)",
    r"(\d+)\s*\+?\s*(?:years?|yrs?|nam)\s+(?:of\s+)?(?:experience|exp|kinh\s+nghiem)",
    r"(?:experience|exp|kinh\s+nghiem)\s*(?::|toi\s+thieu|tu)?\s*(\d+)\s*\+?\s*(?:years?|yrs?|nam)",
]

EDUCATION_REQUIRED_PATTERNS = [
    r"\b(bachelor|master|phd|degree|diploma)\b",
    r"\b(computer science|information technology|software engineering)\b",
    r"\b(university|college)\b",
    r"\b(dai\s+hoc|cao\s+dang|bang|tot\s+nghiep|cu\s+nhan|thac\s+si)\b",
]

NO_EDUCATION_REQUIRED_PATTERNS = [
    r"\bno\s+(specific\s+)?(?:degree|education|background)\s+(?:required|needed)\b",
    r"\b(?:degree|education|background)\s+(?:not\s+required|optional)\b",
    r"\bkhong\s+(?:yeu\s+cau|can|doi\s+hoi)\s+(?:bang|bang\s+cap|dai\s+hoc|hoc\s+van)\b",
    r"\bkhong\s+yeu\s+cau\s+bang\s+cap\s+cu\s+the\b",
]


def _to_number(value: Any, default: int = 0) -> int:
    try:
        return int(round(float(value)))
    except Exception:
        return default


def _clamp_score(value: Any, maximum: int) -> int:
    return max(0, min(maximum, _to_number(value)))


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


def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        text = re.sub(r"\s+", " ", str(item or "")).strip(" -,*.;:")
        key = _skill_key(text)
        if text and key and key not in seen:
            seen.add(key)
            result.append(text)
    return result


def _extract_jd_required_skills(jd_content: str) -> List[str]:
    skills: List[str] = []
    in_skill_block = False

    for raw_line in str(jd_content or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if re.match(r"^\[(SKILLS|JD_REQUIRED_SKILLS)\]$", line, re.IGNORECASE):
            in_skill_block = True
            continue
        if line.startswith("[") and line.endswith("]"):
            in_skill_block = False
            continue

        if in_skill_block and line.startswith("-"):
            skills.append(line.strip("- ").strip())
            continue

        if in_skill_block:
            parts = re.split(r",|/|\bor\b|\bhoặc\b|\bhoac\b", line, flags=re.IGNORECASE)
            skills.extend(part.strip() for part in parts)

    return _dedupe_keep_order(skills)


def _cv_text_and_skills(cv_chunks: List[Dict], cv_profile: Dict) -> tuple[str, List[str]]:
    texts = []
    skills = list(cv_profile.get("skills") or [])
    for chunk in cv_chunks:
        texts.append(str(chunk.get("text") or chunk.get("embedding_text") or chunk.get("content") or ""))
        info = chunk.get("extracted_info") if isinstance(chunk, dict) else {}
        if isinstance(info, dict):
            skills.extend(info.get("chunk_skills") or [])
            skills.extend(info.get("technical_skills") or [])
        skill_text = chunk.get("skill_text") if isinstance(chunk, dict) else ""
        for line in str(skill_text or "").splitlines():
            if line.strip().startswith("-"):
                skills.append(line.strip("- ").strip())
    return "\n".join(texts), _dedupe_keep_order(skills)


def _contains_skill(haystack: str, skill: str) -> bool:
    normalized_haystack = _norm_text(haystack)
    normalized_skill = _norm_text(skill)
    if not normalized_skill:
        return False
    return bool(re.search(r"(?<![a-z0-9])" + re.escape(normalized_skill) + r"(?![a-z0-9])", normalized_haystack))


def _required_experience_years(jd_content: str) -> int | None:
    text = _norm_text(jd_content)
    for pattern in NO_EXPERIENCE_REQUIRED_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return 0
    for pattern in EXPERIENCE_YEAR_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def _score_experience_for_jd(jd_content: str, cv_years: int) -> int | None:
    required_years = _required_experience_years(jd_content)
    if required_years is None:
        return None
    if required_years <= 0:
        return 30
    if cv_years >= required_years:
        return 30
    if cv_years <= 0:
        return 8
    return max(10, min(29, round(30 * (cv_years / required_years))))


def _jd_requires_education(jd_content: str) -> bool:
    education_blocks = re.findall(
        r"\[EDUCATION\]([\s\S]*?)(?=\n\[[A-Z_]+\]|\Z)",
        str(jd_content or ""),
        flags=re.IGNORECASE,
    )
    text = " ".join(education_blocks) if education_blocks else str(jd_content or "")
    normalized_text = _norm_text(text)
    if any(re.search(pattern, normalized_text, re.IGNORECASE) for pattern in NO_EDUCATION_REQUIRED_PATTERNS):
        return False
    return any(re.search(pattern, normalized_text, re.IGNORECASE) for pattern in EDUCATION_REQUIRED_PATTERNS)


def _score_education_for_jd(jd_content: str, cv_text: str) -> int | None:
    if not _jd_requires_education(jd_content):
        return 20
    cv_norm = _norm_text(cv_text)
    has_education_signal = bool(re.search(
        r"\b(bachelor|master|phd|degree|university|college|computer science|information technology|"
        r"software engineering|education|gpa)\b",
        cv_norm,
        re.IGNORECASE,
    ))
    return 20 if has_education_signal else 8


def _postprocess_evaluation(
    evaluation: Dict,
    jd_content: str,
    cv_chunks: List[Dict],
    cv_profile: Dict,
) -> Dict:
    fixed = evaluation.copy() if isinstance(evaluation, dict) else _empty_evaluation("Invalid LLM evaluation")

    for key, maximum in SCORE_LIMITS.items():
        fixed[key] = _clamp_score(fixed.get(key, 0), maximum)

    cv_text, cv_skills = _cv_text_and_skills(cv_chunks, cv_profile)
    jd_required_skills = _extract_jd_required_skills(jd_content)
    jd_skill_by_key = {_skill_key(skill): skill for skill in jd_required_skills if _skill_key(skill)}

    llm_matched = fixed.get("matched_skills") if isinstance(fixed.get("matched_skills"), list) else []
    llm_missing = fixed.get("missing_skills") if isinstance(fixed.get("missing_skills"), list) else []

    matched_keys = set()
    for skill in llm_matched:
        key = _skill_key(skill)
        if key in jd_skill_by_key and (_contains_skill(cv_text, skill) or any(_skill_key(s) == key for s in cv_skills)):
            matched_keys.add(key)

    # Add deterministic matches that the small LLM may miss.
    for skill in jd_required_skills:
        key = _skill_key(skill)
        if key and (_contains_skill(cv_text, skill) or any(_skill_key(s) == key for s in cv_skills)):
            matched_keys.add(key)

    missing_keys = set()
    for skill in llm_missing:
        key = _skill_key(skill)
        if key in jd_skill_by_key and key not in matched_keys:
            missing_keys.add(key)
    for skill in jd_required_skills:
        key = _skill_key(skill)
        if key and key not in matched_keys:
            missing_keys.add(key)

    fixed["matched_skills"] = [jd_skill_by_key[key] for key in jd_skill_by_key if key in matched_keys]
    fixed["missing_skills"] = [jd_skill_by_key[key] for key in jd_skill_by_key if key in missing_keys]

    if jd_required_skills:
        fixed["technical_score"] = round(40 * len(fixed["matched_skills"]) / len(jd_required_skills))
    else:
        fixed["technical_score"] = 40
    fixed["technical_score"] = _clamp_score(fixed.get("technical_score", 0), 40)

    experience_score = _score_experience_for_jd(jd_content, int(cv_profile.get("experience_years") or 0))
    if experience_score is not None:
        fixed["experience_score"] = experience_score

    education_score = _score_education_for_jd(jd_content, cv_text)
    if education_score is not None:
        fixed["education_score"] = education_score

    fixed["fit_score"] = _clamp_score(fixed.get("fit_score", 0), 10)
    fixed["score"] = sum(fixed[key] for key in SCORE_LIMITS)

    if fixed["score"] >= 80:
        fixed["recommendation"] = "Strong fit"
    elif fixed["score"] >= 65:
        fixed["recommendation"] = "Good fit"
    elif fixed["score"] >= 45:
        fixed["recommendation"] = "Consider"
    else:
        fixed["recommendation"] = "Not a fit"

    summary = str(fixed.get("summary") or "").strip()
    if not summary:
        fixed["summary"] = "Evaluation adjusted using JD-aware scoring rules."
    elif fixed["score"] != _to_number(evaluation.get("score", fixed["score"])):
        fixed["summary"] = summary + " Scores were normalized against explicit JD requirements."

    return fixed


# =============================================================================
# PUBLIC API
# =============================================================================

def match_cv_to_jds(cv_chunks: List[Dict], top_k: int = 1) -> List[Dict]:
    """
    Match CV với JD bằng global vector search trên JD chunks.

    Luồng:
      1. Mỗi CV chunk → embed → tìm JD chunks gần nhất trên toàn bộ collection
      2. Boost theo section của JD chunk trả về rồi tổng hợp score theo jd_id
      3. Lấy top_k JD có score cao nhất
      4. LLM đánh giá chi tiết từng JD
      5. Sắp xếp theo LLM score, trả về kết quả

    Args:
        cv_chunks: List[Dict] từ pdf_utils.cv_pdf_to_chunks()
        top_k:     Số JD trả về sau khi rank

    Returns:
        List[Dict] mỗi phần tử gồm:
          jd_id, jd_title, similarity_score, section_scores,
          evaluation (LLM), cv_profile
    """
    if count_indexed_jds() == 0:
        return [{
            "jd_id":            "error",
            "jd_title":         "JDs not indexed yet",
            "similarity_score": 0,
            "section_scores":   {},
            "evaluation":       {"match": "Please index JDs first using the Index JDs button"},
            "cv_profile":       "N/A",
        }]

    cv_profile = extract_cv_profile(cv_chunks)
    cv_summary = build_cv_summary(cv_chunks, cv_profile)

    # Bước 1 & 2: global vector search + section boost
    try:
        jd_scores = _match_chunks_to_jds(cv_chunks, k_per_chunk=5)
    except Exception as e:
        return [{
            "jd_id":            "error",
            "jd_title":         "Vector Search Error",
            "similarity_score": 0,
            "section_scores":   {},
            "evaluation":       {"match": f"Error: {str(e)[:100]}"},
            "cv_profile":       cv_profile,
        }]

    if not jd_scores:
        return [{
            "jd_id":            "error",
            "jd_title":         "No matching JDs found",
            "similarity_score": 0,
            "section_scores":   {},
            "evaluation":       _empty_evaluation("Vector search returned no results."),
            "cv_profile":       cv_profile,
        }]

    # Bước 3: lấy top_k JD theo weighted_score
    ranked = sorted(jd_scores.items(), key=lambda x: x[1]["weighted_score"], reverse=True)
    top_jds = ranked[:top_k]

    # Bước 4: LLM đánh giá từng JD
    llm   = OllamaLLM(model=LLM_MODEL, temperature=0.0)
    chain = MATCH_PROMPT | llm | StrOutputParser()

    results = []
    for jid, jd_data in top_jds:
        evaluation = _evaluate_with_llm(jd_data["jd_content"], cv_summary, chain)
        evaluation = _postprocess_evaluation(
            evaluation,
            jd_data["jd_content"],
            cv_chunks,
            cv_profile,
        )

        results.append({
            "jd_id":            jid,
            "jd_title":         jd_data["title"],
            "similarity_score": round(min(1.0, jd_data["weighted_score"]) * 100, 1),
            "section_scores":   jd_data["section_scores"],   # thêm mới: score từng section
            "evaluation":       evaluation,
            "cv_profile":       cv_profile,
        })

    # Bước 5: sort theo LLM score
    results.sort(key=lambda x: x["evaluation"].get("score", 0), reverse=True)
    return results


# =============================================================================
# DATE / EXPERIENCE HELPERS (giữ nguyên)
# =============================================================================

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
    end_token  = rf"(?:{month_names}\s+\d{{4}}|\d{{1,2}}/\d{{4}}|\d{{4}}|present|current|now|ongoing)"
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
