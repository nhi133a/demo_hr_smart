"""Prompt builder for extracting Job Descriptions into requirement units."""

import os


JD_PARSER_SYSTEM_PROMPT = """You are an expert technical recruiter.
Extract the Job Description into compact JSON for CV matching.

Return ONLY valid JSON. No markdown, no comments.

Required shape:
{
  "job_title": string|null,
  "target_level": "intern"|"fresher"|"junior"|"mid"|"senior"|"lead"|"manager"|null,
  "metadata_filter": {
    "education_min": string|null,
    "exp_years_min": number|null,
    "job_type": string|null,
    "location": string|null
  },
  "requirement_units": [
    {
      "name": string,
      "category": "skill"|"soft_skill"|"experience"|"education"|"responsibility"|"certification"|"competency",
      "importance": "required"|"preferred"|"responsibility",
      "source_section": string|null,
      "evidence": string,
      "confidence": "high"|"medium"|"low",
      "alternative_group": string|null
    }
  ]
}

Rules:
- Extract only information explicitly present in the JD.
- Put every concrete hiring criterion into requirement_units.
- Use category=skill for technical tools, languages, frameworks, databases, cloud, platforms, methods, and test/process tools.
- Use category=competency for domain/process knowledge, not tools.
- Use category=soft_skill only for human traits and working style.
- Use category=experience for years/months/level of experience requirements.
- Use category=education for degree, major, or academic requirements.
- Use category=responsibility for job duties or tasks.
- Use category=certification for certificates or licenses.
- Use importance=required for must-have/mandatory/required requirements.
- Use importance=preferred only for plus/nice-to-have/preferred/advantage items.
- Use importance=responsibility for responsibilities.
- Each unit must include short evidence copied from the JD.
- For "A or B", create separate units for A and B with the same alternative_group.
- Convert years to metadata_filter.exp_years_min when explicitly stated.
- Infer target_level only from clear signals: title, explicit seniority, required years, or leadership scope.
- Do not output scoring_config, hard_filters, retrieval_weights, or any fields outside the required shape.
"""


FEW_SHOT_EXAMPLES = """
INPUT:
Backend Engineer
Requirements:
- 2+ years backend experience
- Python with FastAPI or Django
Nice to have:
- Docker
Responsibilities:
- Build REST APIs

OUTPUT:
{"job_title":"Backend Engineer","target_level":"junior","metadata_filter":{"education_min":null,"exp_years_min":2,"job_type":null,"location":null},"requirement_units":[{"name":"2+ years backend experience","category":"experience","importance":"required","source_section":"Requirements","evidence":"2+ years backend experience","confidence":"high","alternative_group":null},{"name":"Python","category":"skill","importance":"required","source_section":"Requirements","evidence":"Python with FastAPI or Django","confidence":"high","alternative_group":null},{"name":"FastAPI","category":"skill","importance":"required","source_section":"Requirements","evidence":"Python with FastAPI or Django","confidence":"high","alternative_group":"python_backend_framework"},{"name":"Django","category":"skill","importance":"required","source_section":"Requirements","evidence":"Python with FastAPI or Django","confidence":"high","alternative_group":"python_backend_framework"},{"name":"Docker","category":"skill","importance":"preferred","source_section":"Nice to have","evidence":"Docker","confidence":"high","alternative_group":null},{"name":"Build REST APIs","category":"responsibility","importance":"responsibility","source_section":"Responsibilities","evidence":"Build REST APIs","confidence":"high","alternative_group":null}]}
"""


JD_INCLUDE_FEW_SHOT = os.getenv("JD_INCLUDE_FEW_SHOT", "false").strip().lower() in {
    "1",
    "true",
    "yes",
}


def get_jd_parser_prompt(jd_text: str) -> str:
    examples = f"\n\n{FEW_SHOT_EXAMPLES}" if JD_INCLUDE_FEW_SHOT else ""
    return f"{JD_PARSER_SYSTEM_PROMPT}{examples}\n\nJD TO PARSE:\n{jd_text}"
