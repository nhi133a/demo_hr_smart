"""Prompt builder for extracting Job Descriptions into a CV-comparable schema."""

import os


JD_PARSER_SYSTEM_PROMPT = """You are an expert technical recruiter and data analyst.
Parse the Job Description into structured JSON for high-precision CV matching.

Rules:
- Return JSON only. No markdown.
- Extract only information explicitly present in the JD.
- Do not infer missing requirements, seniority, education, experience, tools, filters, or scoring logic.
- Every extracted skill, competency, soft skill, and requirement unit must include short evidence from the JD.
- Use null for missing scalar values, [] for missing lists, false for missing booleans, and 0 for missing min_months.
- Do not enable hard filters. All hard_filters values must be false and required_skill_min_match_rate must be 0.

Skill taxonomy:
- required_skills: mandatory technical tools, languages, frameworks, databases, cloud services, certifications, concepts, and processes.
- preferred_skills: bonus, nice-to-have, plus, advantage, preferred technical items.
- competencies: domain knowledge or technical/process competencies.
- soft_skills: human traits and working style.
- responsibilities: duties/tasks, unless the sentence explicitly states a concrete required skill.

Experience:
- Convert years to months. Example: 2 years = 24.
- For ranges, use lower bound as min_months and upper bound as max_months.
- Set exclude_internship=true only when the JD explicitly excludes internship experience.

Required JSON shape:
{
  "source_type": "jd",
  "job_title": string|null,
  "metadata_filter": {
    "education_min": string|null,
    "exp_years_min": number|null,
    "job_type": string|null,
    "location": string|null,
    "target_level": string|null
  },
  "scoring_config": {
    "weights": {
      "required_skills": number,
      "preferred_skills": number,
      "competencies": number,
      "experience": number,
      "education": number,
      "quality": number,
      "context": number
    },
    "required_skill_min_match_rate": 0,
    "hard_filters": {
      "metadata": false,
      "required_skills": false,
      "experience": false,
      "education": false
    },
    "retrieval_weights": {
      "dense": number,
      "lexical": number,
      "schema": number,
      "cross_encoder": number,
      "llm_judge": number
    }
  },
  "required_skills": [
    {
      "name": string,
      "type": string|null,
      "confidence": "high|medium|low",
      "importance": "required",
      "evidence": string,
      "alternative_group": string|null
    }
  ],
  "preferred_skills": [
    {
      "name": string,
      "type": string|null,
      "confidence": "high|medium|low",
      "importance": "preferred",
      "evidence": string,
      "alternative_group": string|null
    }
  ],
  "competencies": [
    {
      "name": string,
      "category": "technical|process|domain"|null,
      "confidence": "high|medium|low",
      "importance": "required|preferred",
      "evidence": string
    }
  ],
  "responsibilities": [string],
  "soft_skills": [
    {"name": string, "confidence": "high|medium|low", "evidence": string}
  ],
  "experience": {
    "min_months": number,
    "max_months": number|null,
    "required": boolean,
    "exclude_internship": boolean,
    "evidence": string
  },
  "education": {
    "level": string|null,
    "major": string|null,
    "required": boolean,
    "evidence": string|null
  },
  "work_context": {
    "role": string|null,
    "seniority": string|null,
    "team_size": string|null,
    "report_to": string|null,
    "industry": string|null,
    "company_type": string|null,
    "domains": [string],
    "platforms": [string],
    "tools": [string],
    "processes": [string]
  },
  "requirement_units": [
    {
      "name": string,
      "category": "skill|soft_skill|experience|education|responsibility|certification"|null,
      "importance": "required|preferred|responsibility"|null,
      "source_section": string|null,
      "evidence": string,
      "confidence": "high|medium|low",
      "alternative_group": string|null
    }
  ],
  "recruiter_expectations": [string],
  "_confidence": {
    "job_title": "high|medium|low"|null,
    "metadata_filter": "high|medium|low"|null,
    "required_skills": "high|medium|low"|null,
    "preferred_skills": "high|medium|low"|null,
    "responsibilities": "high|medium|low"|null,
    "soft_skills": "high|medium|low"|null,
    "work_context": "high|medium|low"|null
  }
}
"""


FEW_SHOT_EXAMPLES = """
INPUT:
Requirements:
- 2+ years backend experience
- Python with FastAPI or Django
Nice to have:
- Docker

OUTPUT:
{"source_type":"jd","job_title":null,"metadata_filter":{"education_min":null,"exp_years_min":2,"job_type":null,"location":null,"target_level":null},"scoring_config":{"weights":{"required_skills":0.45,"preferred_skills":0.1,"competencies":0.1,"experience":0.15,"education":0.05,"quality":0.05,"context":0.1},"required_skill_min_match_rate":0,"hard_filters":{"metadata":false,"required_skills":false,"experience":false,"education":false},"retrieval_weights":{"dense":0.7,"lexical":0.2,"schema":0.1,"cross_encoder":0,"llm_judge":0}},"required_skills":[{"name":"Python","type":"language","confidence":"high","importance":"required","evidence":"Python with FastAPI or Django","alternative_group":null},{"name":"FastAPI","type":"framework","confidence":"high","importance":"required","evidence":"FastAPI or Django","alternative_group":"python_backend_framework"},{"name":"Django","type":"framework","confidence":"high","importance":"required","evidence":"FastAPI or Django","alternative_group":"python_backend_framework"}],"preferred_skills":[{"name":"Docker","type":"tool","confidence":"high","importance":"preferred","evidence":"Docker","alternative_group":null}],"competencies":[],"responsibilities":[],"soft_skills":[],"experience":{"min_months":24,"max_months":null,"required":true,"exclude_internship":false,"evidence":"2+ years backend experience"},"education":{"level":null,"major":null,"required":false,"evidence":null},"work_context":{"role":null,"seniority":null,"team_size":null,"report_to":null,"industry":null,"company_type":null,"domains":[],"platforms":[],"tools":["Python","FastAPI","Django","Docker"],"processes":[]},"requirement_units":[{"name":"Python","category":"skill","importance":"required","source_section":"requirements","evidence":"Python with FastAPI or Django","confidence":"high","alternative_group":null},{"name":"FastAPI","category":"skill","importance":"required","source_section":"requirements","evidence":"FastAPI or Django","confidence":"high","alternative_group":"python_backend_framework"},{"name":"Django","category":"skill","importance":"required","source_section":"requirements","evidence":"FastAPI or Django","confidence":"high","alternative_group":"python_backend_framework"},{"name":"Docker","category":"skill","importance":"preferred","source_section":"preferred_skills","evidence":"Docker","confidence":"high","alternative_group":null}],"recruiter_expectations":[],"_confidence":{"job_title":"low","metadata_filter":"high","required_skills":"high","preferred_skills":"high","responsibilities":"low","soft_skills":"low","work_context":"medium"}}
"""


JD_INCLUDE_FEW_SHOT = os.getenv("JD_INCLUDE_FEW_SHOT", "false").strip().lower() in {
    "1",
    "true",
    "yes",
}


def get_jd_parser_prompt(jd_text: str) -> str:
    """Build the JD parser prompt."""
    examples = f"\n\n{FEW_SHOT_EXAMPLES}" if JD_INCLUDE_FEW_SHOT else ""
    return f"""{JD_PARSER_SYSTEM_PROMPT}
{examples}

JD TO PARSE:
{jd_text}
"""