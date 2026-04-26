import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

from bedrock_utils import get_embedding
from jd_store import count_indexed_jds, search_similar_jds

logger = logging.getLogger(__name__)

LLM_MODEL = "qwen2.5:1.5b"


def truncate_text(text: str, max_length: int = 1000) -> str:
    if len(text) <= max_length:
        return text
    truncated = text[:max_length]
    last_period = truncated.rfind(".")
    if last_period > max_length * 0.65:
        return truncated[: last_period + 1] + "..."
    return truncated + "..."


def extract_relevant_sections(cv_chunks: List[Dict]) -> str:
    section_aliases = {
        "skills_tools": "skills",
        "skill": "skills",
        "technical_skills": "skills",
        "passion": "summary",
        "profile": "summary",
        "about": "summary",
        "work_experience": "experience",
        "employment": "experience",
    }
    priority = ["experience", "skills", "summary", "projects", "education", "certifications"]

    sections = []
    for ch in cv_chunks:
        raw_section = str(ch.get("section", "")).strip().lower()
        normalized_section = section_aliases.get(raw_section, raw_section)
        if normalized_section in priority:
            section_name = normalized_section.upper()
            content = truncate_text(ch["text"], 900 if normalized_section == "experience" else 550)
            sections.append(f"[{section_name}]\n{content}")

    if not sections and cv_chunks:
        for i, ch in enumerate(cv_chunks[:5]):
            sections.append(f"[CHUNK {i + 1}]\n{truncate_text(ch['text'], 600)}")

    return "\n\n".join(sections)


def extract_cv_profile(cv_chunks: List[Dict]) -> Dict[str, Any]:
    full_text = " ".join(ch["text"] for ch in cv_chunks).lower()

    skill_keywords = [
        "python",
        "java",
        "javascript",
        "typescript",
        "react",
        "node.js",
        "fastapi",
        "aws",
        "azure",
        "docker",
        "kubernetes",
        "sql",
        "mongodb",
        "git",
        "selenium",
        "manual testing",
        "jira",
        "postman",
        "api testing",
        "test case",
        "bug reporting",
        "qa",
    ]

    found_skills = [s.title() for s in skill_keywords if s in full_text]
    experience_years = _calculate_experience_years(full_text)
    experience_level = "Senior" if experience_years >= 5 else "Mid-level" if experience_years >= 2 else "Junior/Intern"

    return {
        "skills": found_skills[:15],
        "total_skills": len(found_skills),
        "experience_years": experience_years,
        "experience_level": experience_level,
    }


def _calculate_experience_years(text: str) -> int:
    intervals = _extract_date_intervals(text)
    if intervals:
        total_months = _merge_and_sum_intervals(intervals)
        if total_months > 0:
            return min(max(round(total_months / 12), 0), 60)

    year_patterns = [
        r"(\d+)\s*\+\s*years",
        r"(\d+)\s*years",
        r"over\s*(\d+)\s*years",
    ]

    for pattern in year_patterns:
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))

    return 0


def _extract_date_intervals(text: str) -> List[tuple[int, int]]:
    current = datetime.now()
    current_index = _month_index(current.year, current.month)
    intervals: List[tuple[int, int]] = []

    month_names = (
        r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
        r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|"
        r"nov(?:ember)?|dec(?:ember)?"
    )
    separator = r"(?:-|–|—|to|until|through)"
    end_token = rf"(?:{month_names}\s+\d{{4}}|\d{{1,2}}/\d{{4}}|\d{{4}}|present|current|now|ongoing)"

    patterns = [
        rf"(?P<start_month>{month_names})\s+(?P<start_year>\d{{4}})\s*{separator}\s*(?P<end>{end_token})",
        rf"(?P<start_month_num>\d{{1,2}})/(?P<start_year_num>\d{{4}})\s*{separator}\s*(?P<end>{end_token})",
        rf"(?P<start_year_only>\d{{4}})\s*{separator}\s*(?P<end>{end_token})",
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            start_index = _parse_start_date(match)
            end_index = _parse_end_date(match.group("end"), current_index)
            if start_index is None or end_index is None:
                continue
            if end_index < start_index:
                continue
            intervals.append((start_index, end_index))

    return intervals


def _parse_start_date(match: re.Match) -> int | None:
    if match.groupdict().get("start_month") and match.groupdict().get("start_year"):
        return _month_index(
            int(match.group("start_year")),
            _month_name_to_number(match.group("start_month")),
        )

    if match.groupdict().get("start_month_num") and match.groupdict().get("start_year_num"):
        month = int(match.group("start_month_num"))
        year = int(match.group("start_year_num"))
        if 1 <= month <= 12:
            return _month_index(year, month)
        return None

    if match.groupdict().get("start_year_only"):
        return _month_index(int(match.group("start_year_only")), 1)

    return None


def _parse_end_date(value: str, current_index: int) -> int | None:
    value = value.strip().lower()

    if value in {"present", "current", "now", "ongoing"}:
        return current_index

    month_year_match = re.fullmatch(
        r"(?P<month>jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
        r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+(?P<year>\d{4})",
        value,
        re.IGNORECASE,
    )
    if month_year_match:
        return _month_index(
            int(month_year_match.group("year")),
            _month_name_to_number(month_year_match.group("month")),
        )

    numeric_match = re.fullmatch(r"(?P<month>\d{1,2})/(?P<year>\d{4})", value)
    if numeric_match:
        month = int(numeric_match.group("month"))
        year = int(numeric_match.group("year"))
        if 1 <= month <= 12:
            return _month_index(year, month)
        return None

    year_match = re.fullmatch(r"\d{4}", value)
    if year_match:
        return _month_index(int(value), 12)

    return None


def _month_name_to_number(value: str) -> int:
    months = {
        "jan": 1,
        "january": 1,
        "feb": 2,
        "february": 2,
        "mar": 3,
        "march": 3,
        "apr": 4,
        "april": 4,
        "may": 5,
        "jun": 6,
        "june": 6,
        "jul": 7,
        "july": 7,
        "aug": 8,
        "august": 8,
        "sep": 9,
        "sept": 9,
        "september": 9,
        "oct": 10,
        "october": 10,
        "nov": 11,
        "november": 11,
        "dec": 12,
        "december": 12,
    }
    return months[value.strip().lower()]


def _month_index(year: int, month: int) -> int:
    return year * 12 + month


def _merge_and_sum_intervals(intervals: List[tuple[int, int]]) -> int:
    if not intervals:
        return 0

    intervals = sorted(intervals)
    merged = [list(intervals[0])]

    for start, end in intervals[1:]:
        last = merged[-1]
        if start <= last[1] + 1:
            last[1] = max(last[1], end)
        else:
            merged.append([start, end])

    return sum(end - start + 1 for start, end in merged)


def build_cv_summary(cv_chunks: List[Dict], cv_profile: Dict) -> str:
    relevant = extract_relevant_sections(cv_chunks)

    profile_str = f"""
[CV PRE-PROCESSED SUMMARY]
Experience Level : {cv_profile['experience_level']} ({cv_profile['experience_years']} years)
Technical Skills : {cv_profile['total_skills']} skills detected
Key Skills       : {', '.join(cv_profile['skills']) if cv_profile['skills'] else 'None'}
"""

    return profile_str.strip() + "\n\n" + relevant


MATCH_PROMPT = PromptTemplate.from_template(
    """
You are a strict and experienced technical recruiter.

=== JOB DESCRIPTION ===
{jd_content}

=== CANDIDATE CV ===
{cv_summary}

Evaluate this candidate strictly against the Job Description.

Scoring Criteria:
- Technical Skills Match (40 points): How many required skills are present?
- Experience Relevance (30 points): Years and relevance of experience.
- Education & Background (20 points)
- Overall Fit & Communication (10 points)

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


def match_cv_to_jds(cv_chunks: List[Dict], top_k: int = 5) -> List[Dict]:
    if count_indexed_jds() == 0:
        return [{
            "jd_id": "error",
            "jd_title": "JDs not indexed yet",
            "similarity_score": 0,
            "evaluation": {"match": "Please index JDs first using the Index JDs button"},
            "cv_profile": "N/A",
        }]

    cv_profile = extract_cv_profile(cv_chunks)
    cv_summary = build_cv_summary(cv_chunks, cv_profile)

    try:
        query_embedding = get_embedding(cv_summary)
        matched_jds = search_similar_jds(query_embedding, k=top_k)
    except Exception as e:
        return [{
            "jd_id": "error",
            "jd_title": "MongoDB Vector Search Error",
            "similarity_score": 0,
            "evaluation": {"match": f"Error loading JDs: {str(e)[:100]}"},
            "cv_profile": cv_profile,
        }]

    if not matched_jds:
        return [{
            "jd_id": "error",
            "jd_title": "No matching JD candidates found",
            "similarity_score": 0,
            "evaluation": _empty_evaluation("Vector search returned no JD results."),
            "cv_profile": cv_profile,
        }]

    llm = OllamaLLM(model=LLM_MODEL, temperature=0.0)
    chain = MATCH_PROMPT | llm | StrOutputParser()

    results = []
    for jd_doc in matched_jds:
        similarity_pct = round(max(0.0, float(jd_doc.get("score", 0.0))) * 100, 1)

        try:
            raw = chain.invoke({
                "jd_content": str(jd_doc.get("content", ""))[:2000],
                "cv_summary": cv_summary,
            })
            evaluation = _safe_json_parse(raw)
        except Exception as e:
            logger.error("LLM evaluation failed: %s", e)
            evaluation = _empty_evaluation(f"LLM processing error: {str(e)[:120]}")

        results.append({
            "jd_id": jd_doc.get("jd_id", "unknown"),
            "jd_title": jd_doc.get("title", "Untitled"),
            "similarity_score": similarity_pct,
            "evaluation": evaluation,
            "cv_profile": cv_profile,
        })

    results.sort(key=lambda x: x["evaluation"].get("score", 0), reverse=True)
    return results[:1] if results else results


def _safe_json_parse(raw: str) -> Dict:
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*?\}", raw)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
        logger.warning("JSON parse failed, returning empty evaluation")
        return _empty_evaluation("Failed to parse LLM output")


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
