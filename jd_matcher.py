import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from jd_store import get_embeddings, CHROMA_JD_DIR
import json
import re
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

LLM_MODEL = "qwen2.5:1.5b"

# =====================================================
# TÍCH HỢP TỪ resume_processor.py
# =====================================================

def categorize_skills(skills: List[str]) -> Dict[str, int]:
    """
    Lấy từ resume_processor.py — phân loại kỹ năng theo domain
    Dùng để làm CV summary chi tiết hơn trước khi matching
    """
    categories = {
        "Programming": ["python", "java", "javascript", "typescript", "c++", "c#", "go", "rust"],
        "Frontend":    ["react", "angular", "vue", "html", "css", "bootstrap", "tailwind"],
        "Backend":     ["node.js", "express", "django", "flask", "spring", "fastapi", "laravel"],
        "Database":    ["sql", "mysql", "postgresql", "mongodb", "redis", "elasticsearch"],
        "Cloud":       ["aws", "azure", "gcp", "docker", "kubernetes", "terraform"],
        "AI_ML":       ["machine learning", "ai", "tensorflow", "pytorch", "data science"],
        "DevOps":      ["git", "jenkins", "ci/cd", "devops", "docker", "kubernetes"],
        "QA_Testing":  ["manual testing", "test case", "bug reporting", "jira", "postman",
                        "selenium", "api testing", "automation testing"],
        "Mobile":      ["ios", "android", "react native", "flutter", "swift", "kotlin"]
    }

    breakdown = {}
    for category, keywords in categories.items():
        count = sum(1 for skill in skills if skill.lower() in [k.lower() for k in keywords])
        if count > 0:
            breakdown[category] = count

    return breakdown


def extract_cv_profile(cv_chunks: List[Dict]) -> Dict[str, Any]:
    """
    Lấy từ fallback logic của resume_processor.py
    Extract thông tin cấu trúc từ CV chunks trước khi matching
    Giúp LLM chấm điểm chính xác hơn vì có thông tin pre-processed
    """
    # Gộp toàn bộ text
    full_text = " ".join(ch["text"] for ch in cv_chunks).lower()

    # --- Extract skills ---
    skill_keywords = [
        "python", "java", "javascript", "typescript", "react", "angular", "vue",
        "node.js", "express", "django", "flask", "fastapi", "spring",
        "aws", "azure", "gcp", "docker", "kubernetes", "terraform",
        "sql", "mysql", "postgresql", "mongodb", "redis",
        "git", "linux", "html", "css",
        "manual testing", "test case", "bug reporting", "jira", "postman",
        "selenium", "api testing", "automation testing",
        "machine learning", "ai", "tensorflow", "pytorch"
    ]
    found_skills = [s.title() for s in skill_keywords if s in full_text]

    # --- Extract experience years ---
    year_patterns = [
        r'(\d+)\s*\+?\s*years?\s*(?:of\s*)?(?:experience|exp)',
        r'experience\s*(?:of\s*)?(\d+)\s*\+?\s*years?'
    ]
    experience_years = 0
    for pattern in year_patterns:
        matches = re.findall(pattern, full_text)
        if matches:
            try:
                experience_years = max(int(m) for m in matches)
                break
            except ValueError:
                continue

    # Fallback: đếm số năm từ khoảng thời gian làm việc
    if experience_years == 0:
        year_ranges = re.findall(r'20(\d{2})\s*[-–]\s*(?:20(\d{2})|present|now)', full_text)
        if year_ranges:
            total = sum(
                (int(end) if end else 25) - int(start)
                for start, end in year_ranges
            )
            experience_years = max(0, total)

    # --- Experience level ---
    if experience_years >= 5:
        experience_level = "Senior"
    elif experience_years >= 2:
        experience_level = "Mid-level"
    else:
        experience_level = "Junior / Intern"

    # --- Skill breakdown theo domain ---
    skill_breakdown = categorize_skills(found_skills)

    # --- Overall score sơ bộ ---
    overall_score = min(100, 20 + len(found_skills) * 4 + experience_years * 3)

    return {
        "skills": found_skills,
        "total_skills": len(found_skills),
        "skill_breakdown": skill_breakdown,
        "experience_years": experience_years,
        "experience_level": experience_level,
        "overall_score": overall_score,
        "fit_assessment": "High" if overall_score >= 70 else "Medium" if overall_score >= 50 else "Low"
    }


def build_cv_summary(cv_chunks: List[Dict], cv_profile: Dict[str, Any]) -> str:
    """
    Kết hợp CV chunks + profile đã extract
    Tạo summary đầy đủ nhất để đưa vào LLM chấm điểm
    """
    priority_sections = [
        "skills", "experience", "summary", "projects",
        "education", "certifications", "skills_tools",
        "title", "passion"
    ]

    # Lấy text theo section quan trọng
    section_text = "\n\n".join(
        f"[{ch['section'].upper()}]\n{ch['text']}"
        for ch in cv_chunks
        if ch["section"] in priority_sections
    )

    if not section_text.strip():
        section_text = "\n\n".join(ch["text"] for ch in cv_chunks[:8])

    # Thêm profile đã pre-process
    skill_breakdown_str = ", ".join(
        f"{cat}: {count}"
        for cat, count in cv_profile["skill_breakdown"].items()
    )

    profile_summary = f"""
[PRE-PROCESSED PROFILE]
Experience: {cv_profile['experience_years']} years ({cv_profile['experience_level']})
Total skills detected: {cv_profile['total_skills']}
Skill domains: {skill_breakdown_str if skill_breakdown_str else 'Not detected'}
Key skills: {', '.join(cv_profile['skills'][:15]) if cv_profile['skills'] else 'Not detected'}
"""

    return profile_summary + "\n\n" + section_text


# =====================================================
# PROMPT CHẤM ĐIỂM
# =====================================================

MATCH_PROMPT = PromptTemplate.from_template("""
You are a strict professional recruiter. Evaluate if the candidate's CV matches the Job Description.

=== JOB DESCRIPTION ===
{jd_content}

=== CANDIDATE CV ===
{cv_summary}

IMPORTANT RULES:
- Only give high scores if skills DIRECTLY match JD requirements.
- If JD is for QA/Testing: Python/Node.js skills do NOT count as matched.
- If JD is for Backend: Manual testing skills do NOT count as matched.
- Use the PRE-PROCESSED PROFILE section to verify experience years and skill domains.
- Be strict. Different positions must receive different scores.

Score using ONLY these 4 criteria:

1. Technical skills match (40pts):
   - All required skills present: 40pts
   - More than half present: 25pts
   - Less than half present: 10pts
   - No relevant skills: 0pts

2. Years of experience (30pts):
   - Meets or exceeds requirement: 30pts
   - Slightly under (less than 1yr gap): 15pts
   - No experience: 0pts

3. Education relevance (20pts):
   - Directly related field: 20pts
   - Somewhat related: 10pts
   - Unrelated or not mentioned: 0pts

4. Soft skills (10pts):
   - Clearly mentioned: 10pts
   - Not mentioned: 0pts

Return ONLY valid JSON, no markdown, no extra text:
{{
  "score": <sum of 4 criteria, 0-100>,
  "technical_score": <0-40>,
  "experience_score": <0-30>,
  "education_score": <0-20>,
  "softskill_score": <0-10>,
  "matched_skills": ["skills that DIRECTLY match JD requirements"],
  "missing_skills": ["skills JD requires but absent in CV"],
  "recommendation": "Strong fit / Consider / Not a fit",
  "summary": "2-3 sentences explaining the score"
}}
""")


# =====================================================
# MAIN MATCHING FUNCTION
# =====================================================

def match_cv_to_jds(cv_chunks: List[Dict], top_k: int = 3) -> List[Dict]:
    """
    Pipeline chính:
    1. Extract CV profile (từ resume_processor logic)
    2. Build CV summary đầy đủ
    3. Tìm JD phù hợp qua vector search
    4. Chấm điểm từng JD bằng LLM
    """

    # Bước 1 — Extract profile từ CV chunks
    cv_profile = extract_cv_profile(cv_chunks)
    logger.info(f"CV Profile: {cv_profile['experience_level']}, "
                f"{cv_profile['total_skills']} skills, "
                f"domains: {list(cv_profile['skill_breakdown'].keys())}")

    # Bước 2 — Build CV summary đầy đủ
    cv_summary = build_cv_summary(cv_chunks, cv_profile)

    # Bước 3 — Tìm JD phù hợp qua ChromaDB
    embeddings = get_embeddings()
    jd_store = Chroma(
        persist_directory=CHROMA_JD_DIR,
        embedding_function=embeddings
    )
    matched_jds = jd_store.similarity_search_with_score(cv_summary, k=top_k)

    # Bước 4 — Chấm điểm từng JD bằng LLM
    llm = OllamaLLM(model=LLM_MODEL, temperature=0)
    chain = MATCH_PROMPT | llm | StrOutputParser()

    results = []
    for jd_doc, distance in matched_jds:

        # Fix similarity: ChromaDB trả về distance, không phải similarity
        similarity_pct = round(max(0.0, 1.0 - distance) * 100, 1)

        raw = chain.invoke({
            "jd_content": jd_doc.page_content,
            "cv_summary": cv_summary
        })

        # Parse JSON an toàn
        try:
            evaluation = json.loads(raw)
        except json.JSONDecodeError:
            json_match = re.search(r'\{.*\}', raw, re.DOTALL)
            if json_match:
                try:
                    evaluation = json.loads(json_match.group())
                except json.JSONDecodeError:
                    evaluation = _empty_evaluation(raw[:200])
            else:
                evaluation = _empty_evaluation("Could not parse LLM response")

        results.append({
            "jd_id":            jd_doc.metadata.get("jd_id"),
            "jd_title":         jd_doc.metadata.get("title"),
            "similarity_score": similarity_pct,
            "evaluation":       evaluation,
            # Thêm CV profile vào kết quả để UI hiển thị
            "cv_profile":       cv_profile
        })

    # Sort theo LLM score cao nhất
    results.sort(key=lambda x: x["evaluation"].get("score", 0), reverse=True)
    return results


def _empty_evaluation(summary: str) -> Dict:
    return {
        "score": 0,
        "technical_score": 0,
        "experience_score": 0,
        "education_score": 0,
        "softskill_score": 0,
        "matched_skills": [],
        "missing_skills": [],
        "recommendation": "Error",
        "summary": summary
    }