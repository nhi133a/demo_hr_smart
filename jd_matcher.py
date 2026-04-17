import re
import json
import logging
from typing import Dict, Any, List

from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from jd_store import get_embeddings, CHROMA_JD_DIR

logger = logging.getLogger(__name__)

LLM_MODEL = "qwen2.5:1.5b"   # Khuyến nghị nâng lên 7b nếu có thể

# =====================================================
# CẢI THIỆN: TRÍCH XUẤT & XÂY DỰNG RELEVANT SECTIONS
# =====================================================

def truncate_text(text: str, max_length: int = 1000) -> str:
    """Cải thiện truncate thông minh hơn (từ rank_cvs)"""
    if len(text) <= max_length:
        return text
    truncated = text[:max_length]
    last_period = truncated.rfind('.')
    if last_period > max_length * 0.65:
        return truncated[:last_period + 1] + '...'
    return truncated + '...'


def extract_relevant_sections(cv_chunks: List[Dict]) -> str:
    """Lấy các section quan trọng theo thứ tự ưu tiên - lấy cảm hứng từ rank_cvs"""
    priority = ["experience", "skills", "summary", "projects", "education", "certifications"]

    sections = []
    for ch in cv_chunks:
        if ch.get("section") in priority:
            section_name = ch["section"].upper()
            content = truncate_text(ch["text"], 900 if ch["section"] == "experience" else 550)
            sections.append(f"[{section_name}]\n{content}")

    # Fallback nếu không có section nào
    if not sections and cv_chunks:
        for i, ch in enumerate(cv_chunks[:5]):
            sections.append(f"[CHUNK {i+1}]\n{truncate_text(ch['text'], 600)}")

    return "\n\n".join(sections)


def extract_cv_profile(cv_chunks: List[Dict]) -> Dict[str, Any]:
    """Giữ logic cũ nhưng tinh gọn và chính xác hơn"""
    full_text = " ".join(ch["text"] for ch in cv_chunks).lower()

    skill_keywords = ["python", "java", "javascript", "typescript", "react", "node.js", "fastapi",
                      "aws", "azure", "docker", "kubernetes", "sql", "mongodb", "git", "selenium"]

    found_skills = [s.title() for s in skill_keywords if s in full_text]

    # Extract experience years
    year_match = re.search(r'(\d+)\s*\+?\s*years?', full_text)
    experience_years = int(year_match.group(1)) if year_match else 0

    experience_level = "Senior" if experience_years >= 5 else "Mid-level" if experience_years >= 2 else "Junior/Intern"

    return {
        "skills": found_skills[:15],
        "total_skills": len(found_skills),
        "experience_years": experience_years,
        "experience_level": experience_level,
    }


def build_cv_summary(cv_chunks: List[Dict], cv_profile: Dict) -> str:
    """Kết hợp profile + relevant sections"""
    relevant = extract_relevant_sections(cv_chunks)

    profile_str = f"""
[CV PRE-PROCESSED SUMMARY]
Experience Level : {cv_profile['experience_level']} ({cv_profile['experience_years']} years)
Technical Skills : {cv_profile['total_skills']} skills detected
Key Skills       : {', '.join(cv_profile['skills']) if cv_profile['skills'] else 'None'}
"""

    return profile_str.strip() + "\n\n" + relevant


# =====================================================
# PROMPT ĐƯỢC CẢI THIỆN THEO KIỂU RECRUITER
# =====================================================

MATCH_PROMPT = PromptTemplate.from_template("""
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

JSON:""")


# =====================================================
# MAIN MATCHING FUNCTION - ĐÃ CẢI THIỆN
# =====================================================

def match_cv_to_jds(cv_chunks: List[Dict], top_k: int = 5) -> List[Dict]:
    """
    Pipeline cải thiện:
    - Extract profile
    - Build rich summary với relevant sections
    - Vector search trên Chroma
    - LLM scoring với prompt chuyên nghiệp
    """
    logger.info("Starting CV to JD matching...")

    # Bước 1: Extract profile
    cv_profile = extract_cv_profile(cv_chunks)

    # Bước 2: Build summary (có relevant sections)
    cv_summary = build_cv_summary(cv_chunks, cv_profile)

    # Bước 3: Tìm JD phù hợp qua vector search
    embeddings = get_embeddings()
    jd_store = Chroma(persist_directory=CHROMA_JD_DIR, embedding_function=embeddings)

    matched_jds = jd_store.similarity_search_with_score(cv_summary, k=top_k)

    # Bước 4: Scoring bằng LLM
    llm = OllamaLLM(model=LLM_MODEL, temperature=0.0)
    chain = MATCH_PROMPT | llm | StrOutputParser()

    results = []
    for jd_doc, distance in matched_jds:
        similarity_pct = round(max(0.0, 1.0 - distance) * 100, 1)

        try:
            raw = chain.invoke({
                "jd_content": jd_doc.page_content[:2500],   # giới hạn độ dài
                "cv_summary": cv_summary
            })

            evaluation = _safe_json_parse(raw)

        except Exception as e:
            logger.error(f"LLM evaluation failed: {e}")
            evaluation = _empty_evaluation("LLM processing error")

        results.append({
            "jd_id":            jd_doc.metadata.get("jd_id", "unknown"),
            "jd_title":         jd_doc.metadata.get("title", "Untitled"),
            "similarity_score": similarity_pct,
            "evaluation":       evaluation,
            "cv_profile":       cv_profile
        })

    # Sort theo điểm LLM
    results.sort(key=lambda x: x["evaluation"].get("score", 0), reverse=True)

    logger.info(f"Matching completed. Best score: {results[0]['evaluation'].get('score', 0) if results else 0}")
    return results


# =====================================================
# HELPER FUNCTIONS
# =====================================================

def _safe_json_parse(raw: str) -> Dict:
    """Parse JSON robust hơn"""
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        match = re.search(r'\{[\s\S]*?\}', raw)
        if match:
            try:
                return json.loads(match.group())
            except:
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
        "summary": summary
    }