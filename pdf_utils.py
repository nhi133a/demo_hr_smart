import json
import re
import html
import tempfile
import os
from pathlib import Path
from typing import Any, List, Dict, Optional
from datetime import datetime

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling.chunking import HybridChunker

def build_converter(*, ocr: bool = False, table_structure: bool = True) -> DocumentConverter:
    opts = PdfPipelineOptions(do_ocr=ocr, do_table_structure=table_structure)
    return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})

def fix_spaced_letters(text: str) -> str:
    return re.sub(r'\b(?:[A-Z]\s){3,}[A-Z]\b', lambda m: m.group(0).replace(" ", ""), text)

def clean_text(text: str) -> str:
    text = html.unescape(text)
    text = fix_spaced_letters(text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def _load_document(pdf_source: str | Path | bytes, converter: DocumentConverter):
    if isinstance(pdf_source, bytes):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf_source)
            tmp_path = tmp.name
        try:
            return converter.convert(tmp_path).document
        finally:
            os.unlink(tmp_path)
    return converter.convert(str(pdf_source)).document

def _merge_small_chunks(chunks: List[Dict], min_chars: int = 80, max_chars: int = 1200) -> List[Dict]:
    if not chunks:
        return chunks
    merged = [chunks[0].copy()]
    for curr in chunks[1:]:
        prev = merged[-1]
        if (len(curr.get("text", "")) < min_chars
                and curr.get("headings") == prev.get("headings")
                and len(prev.get("text", "")) + len(curr.get("text", "")) <= max_chars):
            prev["text"] += "\n" + curr["text"]
        else:
            merged.append(curr.copy())
    return merged

SKIP_EMBED_SECTIONS: set = {
    "contact information", "contact   information",
    "personal information", "personal   information",
    "references", "referee",
    "thông tin cá nhân", "liên hệ", "địa chỉ",
}

_RE_PII = re.compile(
    r"\b(contact|personal\s+info|references?|referee|liên\s+hệ|địa\s+chỉ|thông\s+tin\s+cá\s+nhân)\b",
    re.IGNORECASE,
)

_RE_VALUABLE_EDU = re.compile(
    r"\b(degree|gpa|bachelor|master|phd|d\.sc|b\.sc|m\.sc|specialization|major|faculty|"
    r"tá»‘t\s+nghiá»‡p|chuyÃªn\s+ngÃ nh|khoa|báº±ng\s+(?:cá»­\s+nhÃ¢n|tháº¡c\s+sÄ©|tiáº¿n\s+sÄ©))\b",
    re.IGNORECASE,
)
_RE_VALUABLE_OTHER = re.compile(
    r"\b(skill|experience|worked|intern|project|years?\s+of\s+experience|"
    r"ká»¹\s+nÄƒng|kinh\s+nghiá»‡m|dá»±\s+Ã¡n)\b",
    re.IGNORECASE,
)

def _is_pii_section(chunk: Dict) -> bool:
    label = " ".join([str(chunk.get("section", "")), *chunk.get("headings", [])]).lower()
    return bool(_RE_PII.search(label))

def _should_embed(chunk: Dict) -> bool:
    if not _is_pii_section(chunk):
        return True
    text = chunk.get("text", "")
    return bool(_RE_VALUABLE_EDU.search(text) or _RE_VALUABLE_OTHER.search(text))


EMPTY_CV_EXTRACTION: Dict[str, Any] = {
    "technical_skills": [], "soft_skills": [], "companies": [],
    "total_experience_years": None, "total_experience_months": 0,
    "experience_duration": "0 years", "projects": [], "education": [],
}

def _as_list(v: Any) -> List[Any]:
    return v if isinstance(v, list) else []

def _clean_string_list(value: Any, limit: int = 40) -> List[str]:
    return [s for s in (clean_text(i) for i in _as_list(value) if isinstance(i, str)) if s][:limit]

def _clean_dict_list(value: Any, allowed_keys: List[str], limit: int = 20) -> List[Dict[str, Any]]:
    cleaned = []
    for item in _as_list(value):
        if not isinstance(item, dict):
            continue
        row = {}
        for key in allowed_keys:
            cell = item.get(key)
            if isinstance(cell, str):    cell = clean_text(cell)
            elif isinstance(cell, list): cell = _clean_string_list(cell, limit=20)
            elif cell is not None and not isinstance(cell, (int, float)): cell = str(cell)
            if cell not in (None, "", []): row[key] = cell
        if row:
            cleaned.append(row)
    return cleaned[:limit]

def _extract_json_object(text: str) -> Dict[str, Any]:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m: return {}
        try: parsed = json.loads(m.group(0))
        except json.JSONDecodeError: return {}
    return parsed if isinstance(parsed, dict) else {}

def _normalize_extraction(data: Dict[str, Any]) -> Dict[str, Any]:
    n = EMPTY_CV_EXTRACTION.copy()
    n["technical_skills"] = _clean_string_list(data.get("technical_skills"))
    n["soft_skills"]       = _clean_string_list(data.get("soft_skills"))
    n["companies"]  = _clean_dict_list(data.get("companies"),  ["name","role","duration","years","skills"])
    n["projects"]   = _clean_dict_list(data.get("projects"),   ["name","role","duration","technologies","summary"])
    n["education"]  = _clean_dict_list(data.get("education"),  ["school","degree","major","gpa","duration"])
    years = data.get("total_experience_years")
    if isinstance(years, (int, float)):
        n["total_experience_years"] = years
    elif isinstance(years, str):
        m = re.search(r"\d+(?:\.\d+)?", years)
        n["total_experience_years"] = float(m.group(0)) if m else None
    return n


def _experience_display(total_months: int) -> str:
    if total_months <= 0:
        return "0 years"
    if total_months < 12:
        return f"{total_months} month" + ("" if total_months == 1 else "s")
    years = round(total_months / 12)
    return f"{years} year" + ("" if years == 1 else "s")


def _experience_years_for_matching(total_months: int) -> float:
    if total_months <= 0:
        return 0
    if total_months < 12:
        return round(total_months / 12, 2)
    return float(round(total_months / 12))


def _extract_total_experience_months(text: str) -> int:
    intervals = _extract_date_intervals(text)
    if intervals:
        months = _merge_and_sum_intervals(intervals)
        if months > 0:
            return min(months, 60 * 12)

    explicit_months = _extract_explicit_duration_months(text)
    if explicit_months > 0:
        return min(explicit_months, 60 * 12)

    return 0


def _extract_explicit_duration_months(text: str) -> int:
    text = _normalize_duration_words(text)
    total = 0
    seen_spans = []

    duration_pattern = re.compile(
        r"(?<!\d)(\d+(?:\.\d+)?)\s*\+?\s*"
        r"(years?|yrs?|yoe|nam|năm|months?|mos?|thang|tháng)\b",
        re.IGNORECASE,
    )
    for match in duration_pattern.finditer(text):
        number = float(match.group(1))
        unit = match.group(2).lower()
        months = round(number * 12) if unit.startswith(("year", "yr", "yoe", "nam", "năm")) else round(number)
        total += months
        seen_spans.append(match.span())

    if total:
        return total

    summary_patterns = [
        r"(?:over|more\s+than|almost|about|around|approximately)?\s*(\d+(?:\.\d+)?)\s*\+?\s*years?\s+of\s+experience",
        r"experience\s*(?:of|:)?\s*(\d+(?:\.\d+)?)\s*\+?\s*years?",
    ]
    for pattern in summary_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return round(float(match.group(1)) * 12)

    return 0

def _normalize_duration_words(text: str) -> str:
    numbers = {
        "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
        "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
        "eleven": "11", "twelve": "12",
    }
    for word, number in numbers.items():
        text = re.sub(rf"\b{word}\b", number, text, flags=re.IGNORECASE)
    return text


def _finalize_experience_extraction(extraction: Dict[str, Any], cv_text: str) -> Dict[str, Any]:
    finalized = extraction.copy()
    months = _extract_total_experience_months(cv_text)

    if months <= 0 and isinstance(finalized.get("total_experience_years"), (int, float)):
        months = round(float(finalized["total_experience_years"]) * 12)

    months = max(months, 0)
    finalized["total_experience_months"] = months
    finalized["total_experience_years"] = _experience_years_for_matching(months)
    finalized["experience_duration"] = _experience_display(months)
    return finalized


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
    end_token = rf"(?:(?:{month_names})\s+\d{{4}}|\d{{1,2}}/\d{{4}}|\d{{4}}|present|current|now|ongoing)"
    patterns = [
        rf"(?P<start_month>{month_names})\s+(?P<start_year>\d{{4}})\s*{separator}\s*(?P<end>{end_token})",
        rf"(?P<start_month_num>\d{{1,2}})/(?P<start_year_num>\d{{4}})\s*{separator}\s*(?P<end>{end_token})",
        rf"(?P<start_year_only>\d{{4}})\s*{separator}\s*(?P<end>{end_token})",
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            start_index = _parse_start_date(match)
            end_index = _parse_end_date(match.group("end"), current_index)
            if start_index is None or end_index is None or end_index < start_index:
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
        r"\s+(?P<year>\d{4})",
        value,
        re.IGNORECASE,
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
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        last = merged[-1]
        if start <= last[1] + 1:
            last[1] = max(last[1], end)
        else:
            merged.append([start, end])
    return sum(end - start + 1 for start, end in merged)

def _dedupe_strings(items: List[str], limit: int = 40) -> List[str]:
    seen, result = set(), []
    for item in items:
        item = clean_text(item)
        if item and item.lower() not in seen:
            result.append(item); seen.add(item.lower())
    return result[:limit]

def _normalize_company_name(name: str) -> str:
    name = re.sub(
        r'\b(inc|llc|ltd|limited|corp|corporation|co|company|group|holdings?|'
        r'joint\s+stock|jsc|tnhh|cp|cty|cÃ´ng\s+ty)\b\.?',
        '', name.lower(), flags=re.IGNORECASE,
    )
    return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', ' ', name)).strip()

def _company_name_matches(company_name: str, haystack: str) -> bool:
    if not company_name: return False
    nn, nh = _normalize_company_name(company_name), _normalize_company_name(haystack)
    if nn in nh: return True
    tokens = [t for t in nn.split() if len(t) > 3]
    if not tokens: return company_name.lower() in haystack.lower()
    return sum(1 for t in tokens if t in nh) / len(tokens) >= 0.75

_RE_SOFT_SKILL_LABEL = re.compile(
    r"\b(strengths?|soft\s+skills?|personal\s+skills?|attributes?|qualities)\b",
    re.IGNORECASE,
)
_RE_MEANINGFUL_SKILL_LABEL = re.compile(
    r"\b(skills?|technical|tools?|technology|competenc|experience|employment|work|"
    r"career|intern|internship|project|portfolio|objective|summary)\b",
    re.IGNORECASE,
)

def _label_for_chunk(chunk: Dict) -> str:
    return " ".join([str(chunk.get("section", "")), *chunk.get("headings", [])])

def _is_soft_skill_section_label(label: str) -> bool:
    return bool(_RE_SOFT_SKILL_LABEL.search(label))

def _extract_soft_skills_from_soft_sections(chunks: List[Dict]) -> List[str]:
    skills = []
    for chunk in chunks:
        label = _label_for_chunk(chunk).lower()
        if not _is_soft_skill_section_label(label):
            continue
        for line in chunk.get("text", "").splitlines():
            line = clean_text(re.sub(r"^[\-\u2022*]\s*", "", line))
            if line and 2 <= len(line) <= 120:
                skills.append(line)
    return _dedupe_strings(skills, limit=30)

def _merge_extracted_skills(extraction: Dict[str, Any], chunks: List[Dict]) -> Dict[str, Any]:
    merged = extraction.copy()
    merged["technical_skills"] = _dedupe_strings(
        extraction.get("technical_skills", []),
        limit=80,
    )
    merged["soft_skills"] = _dedupe_strings(
        extraction.get("soft_skills", []),
        limit=30,
    )
    return merged

def _contains_skill(text_lower: str, skill: str) -> bool:
    return bool(re.search(r"(?<![a-z0-9])" + re.escape(skill.lower()) + r"(?![a-z0-9])", text_lower))

def _validate_extraction_against_text(extraction: Dict[str, Any], cv_text: str) -> Dict[str, Any]:
    tl = cv_text.lower()
    v  = extraction.copy()
    v["technical_skills"] = [s for s in extraction.get("technical_skills", []) if _contains_skill(tl, s)]
    v["soft_skills"]       = [s for s in extraction.get("soft_skills", [])      if _contains_skill(tl, s)]
    v["companies"]         = [c for c in extraction.get("companies", [])
                               if c.get("name") and _company_name_matches(c["name"], cv_text)]
    if re.search(r"\b(fresher|no professional experience|no work experience)\b", tl):
        v["total_experience_years"] = 0
    elif not v["companies"] and not re.search(r"\b\d+(?:\.\d+)?\s*\+?\s*(?:years?|yrs?)\b", tl):
        v["total_experience_years"] = None
    return v



def _build_cv_text_full(chunks: List[Dict], max_chars: int = 16000) -> str:
    _RE_PRI = re.compile(r"\b(experience|education|skill|project|employment|work|intern|academic)\b", re.IGNORECASE)
    priority, rest = [], []
    for ch in chunks:
        label = " ".join([str(ch.get("section", "")), *ch.get("headings", [])]).lower()
        (priority if _RE_PRI.search(label) else rest).append(ch)
    lines, total = [], 0
    for ch in priority + rest:
        block = f"[SECTION: {ch.get('section','unknown')}]\n{ch.get('text','')}"
        if total + len(block) > max_chars:
            remaining = max_chars - total
            if remaining > 200: lines.append(block[:remaining])
            break
        lines.append(block); total += len(block)
    return "\n\n".join(lines)

def _llm_call(prompt: str) -> str:
    try:
        from local_llm import generate_answer
        return generate_answer(prompt)
    except Exception:
        try:
            from local_llm import generate_answer
            return generate_answer(prompt)
        except Exception:
            return ""

def _parse_array_response(raw: str, allowed_keys: List[str]) -> List[Dict]:
    obj = _extract_json_object(f'{{"_": {raw.strip()}}}')
    result = _clean_dict_list(obj.get("_", []), allowed_keys)
    if not result:
        try:
            arr = json.loads(raw.strip())
            if isinstance(arr, list):
                result = _clean_dict_list(arr, allowed_keys)
        except Exception:
            pass
    return result

def _parse_string_array_response(raw: str) -> List[str]:
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.IGNORECASE)
    obj = _extract_json_object(raw)
    if obj:
        return _clean_string_list(obj.get("technical_skills") or obj.get("skills") or obj.get("_"))
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    candidates = [m.group(0)] if m else []
    candidates.append(raw)
    for candidate in candidates:
        try:
            arr = json.loads(candidate)
            if isinstance(arr, list):
                return _clean_string_list(arr, limit=80)
        except Exception:
            pass
    return _clean_string_list([line.strip("-* \t") for line in raw.splitlines()], limit=80)

_PROMPT_FULL = """Extract structured facts from this CV. Return ONLY valid JSON, no markdown.

RULES: Only extract info explicitly written. Do NOT hallucinate. Empty field -> [] or null.
technical_skills: ALL tools/languages/frameworks/testing methods/QA tools found anywhere in the CV including Experience and Projects sections.
soft_skills: only human traits (responsibility, teamwork, communication, curiosity, etc.).
total_experience_years: sum explicit durations only, else null.
companies: only explicitly named orgs, NOT from Summary/Objective.

Schema:
{{"technical_skills":[],"soft_skills":[],"companies":[{{"name":"","role":"","duration":"","years":0,"skills":[]}}],"total_experience_years":null,"projects":[{{"name":"","role":"","technologies":[],"summary":""}}],"education":[{{"school":"","degree":"","major":"","gpa":"","duration":""}}]}}

CV:
{cv_text}"""

_PROMPT_EDU = """Previous extraction missed education. Extract ONLY education array.
Education may appear inside "Personal Information" or similar sections.
Return ONLY a JSON array: [{{"school":"","degree":"","major":"","gpa":"","duration":""}}]
CV:\n{cv_text}"""

_PROMPT_EXP = """Previous extraction missed work experience. Extract ONLY companies array.
Return ONLY a JSON array: [{{"name":"","role":"","duration":"","years":0,"skills":[]}}]
CV:\n{cv_text}"""

_PROMPT_SKILLS = """Extract ALL technical skills from this CV.
Include: tools, technologies, languages, frameworks, databases, platforms, QA/testing methods.
Look in ALL sections especially Experience, Projects, Objective.
Do NOT include soft skills or personality traits.
Return ONLY a JSON array of strings.
CV:\n{cv_text}"""

def _local_llm_extract_cv_facts(chunks: List[Dict], max_chars: int = 16000) -> Dict[str, Any]:
    if not chunks:
        return EMPTY_CV_EXTRACTION.copy()
    cv_text = _build_cv_text_full(chunks, max_chars=max_chars)

    # Pass 1
    extraction = _normalize_extraction(_extract_json_object(_llm_call(_PROMPT_FULL.format(cv_text=cv_text))))
    extraction = _validate_extraction_against_text(extraction, cv_text)

    # Pass 2: targeted
    if not extraction.get("education"):
        edu = _parse_array_response(_llm_call(_PROMPT_EDU.format(cv_text=cv_text)),
                                    ["school","degree","major","gpa","duration"])
        if edu: extraction["education"] = edu

    if not extraction.get("companies"):
        exp = _parse_array_response(_llm_call(_PROMPT_EXP.format(cv_text=cv_text)),
                                    ["name","role","duration","years","skills"])
        if exp:
            validated = [c for c in exp if c.get("name") and _company_name_matches(c["name"], cv_text)]
            extraction["companies"] = validated or exp

    targeted_skills = _parse_string_array_response(_llm_call(_PROMPT_SKILLS.format(cv_text=cv_text)))
    if targeted_skills:
        extraction["technical_skills"] = _dedupe_strings(
            extraction.get("technical_skills", [])
            + [s for s in targeted_skills if _contains_skill(cv_text.lower(), s)],
            limit=80,
        )

    extraction = _finalize_experience_extraction(extraction, cv_text)
    return _merge_extracted_skills(extraction, chunks)

def _llm_extract_chunk_skills(chunk: Dict) -> List[str]:
    try:
        from local_llm import extract_normalized_skills

        return _dedupe_strings(
            extract_normalized_skills(
                chunk.get("text", ""),
                source_type="CV",
                section=str(chunk.get("section") or "unknown"),
            ),
            limit=40,
        )
    except Exception:
        return []

def _build_skill_text(skills: List[str], label: str = "CV_SKILLS") -> str:
    try:
        from local_llm import build_skill_text

        return build_skill_text(skills, label=label)
    except Exception:
        return "\n".join(f"- {skill}" for skill in skills if skill)


def _normalized_cv_section(chunk: Dict) -> str:
    text = _label_for_chunk(chunk).lower()
    if re.search(r"\b(education|academic|university|college|school|gpa|degree)\b", text): return "education"
    if re.search(r"\b(project|portfolio)\b", text): return "projects"
    if re.search(r"\b(objective|summary)\b", text): return "general"
    if re.search(r"\b(experience|employment|work|career|intern|internship|company|qa|qc|tester|engineer|developer)\b", text): return "experience"
    if _is_soft_skill_section_label(text): return "soft_skills"
    if re.search(r"\b(skills?|technical|tools?|technology|competenc|strength)\b", text): return "skills"
    return "general"


def _matching_projects_for_chunk(chunk: Dict, extraction: Dict[str, Any]) -> List[Dict]:
    haystack = " ".join([chunk.get("section",""), *chunk.get("headings",[]), chunk.get("text","")[:200]]).lower()
    return [p for p in extraction.get("projects", [])
            if (name := str(p.get("name","")).lower()) and name in haystack]

def _chunk_extracted_skills(chunk: Dict, extraction: Dict[str, Any]) -> List[str]:
    text    = chunk.get("text", "")
    section = _normalized_cv_section(chunk)

    skills = _clean_string_list(chunk.get("llm_skills"), limit=40)

    for skill in extraction.get("technical_skills", []):
        if _contains_skill(text.lower(), skill):
            skills.append(skill)

    chunk_label_full = " ".join([chunk.get("section",""), *chunk.get("headings",[]), text[:300]])
    for company in extraction.get("companies", []):
        if company.get("name") and _company_name_matches(company["name"], chunk_label_full):
            skills.extend(_clean_string_list(company.get("skills"), limit=20))

    if section == "projects":
        for project in (_matching_projects_for_chunk(chunk, extraction) or extraction.get("projects", [])):
            skills.extend(_clean_string_list(project.get("technologies"), limit=20))

    return _dedupe_strings(skills, limit=40)

def _chunk_relevant_extraction(chunk: Dict, extraction: Dict[str, Any]) -> Dict[str, Any]:
    section  = _normalized_cv_section(chunk)
    relevant = EMPTY_CV_EXTRACTION.copy()

    chunk_skills = _chunk_extracted_skills(chunk, extraction)
    if chunk_skills:
        relevant["chunk_skills"] = chunk_skills   # key ngoÃ i schema, chá»‰ dÃ¹ng trong pipeline

    if section == "skills":
        relevant["technical_skills"] = extraction.get("technical_skills", [])
        relevant["soft_skills"]       = extraction.get("soft_skills", [])

    elif section == "soft_skills":
        relevant["soft_skills"] = extraction.get("soft_skills", [])

    elif section == "experience":
        chunk_label = " ".join([chunk.get("section",""), *chunk.get("headings",[]), chunk.get("text","")[:300]])
        matched = [c for c in extraction.get("companies", [])
                   if c.get("name") and _company_name_matches(c["name"], chunk_label)]
        relevant["companies"] = matched
        relevant["total_experience_years"] = extraction.get("total_experience_years")
        relevant["total_experience_months"] = extraction.get("total_experience_months", 0)
        relevant["experience_duration"] = extraction.get("experience_duration", "0 years")

    elif section == "projects":
        matched = _matching_projects_for_chunk(chunk, extraction)
        relevant["projects"] = matched or extraction.get("projects", [])

    elif section == "education":
        relevant["education"] = extraction.get("education", [])

    elif section == "general":
        if re.search(r"\b(university|college|school|degree|bachelor|master|phd|trÆ°á»ng|Ä‘áº¡i\s+há»c)\b",
                     chunk.get("text", ""), re.IGNORECASE):
            relevant["education"] = extraction.get("education", [])

    return relevant

def _format_items(title: str, items: List[Any]) -> List[str]:
    if not items: return []
    lines = []
    for item in items:
        if isinstance(item, dict):
            parts = [f"{k}: {', '.join(map(str,v)) if isinstance(v,list) else v}"
                     for k, v in item.items() if v not in (None,"",[])]
            if parts: lines.append("- " + "; ".join(parts))
        elif item:
            lines.append(f"- {item}")
    return [f"[{title}]\n" + "\n".join(lines)] if lines else []

def _build_embedding_text(chunk: Dict, extraction: Dict[str, Any]) -> str:
    section = _normalized_cv_section(chunk)
    ce      = _chunk_relevant_extraction(chunk, extraction)
    if section == "general" and ce.get("projects"): section = "projects"

    lines = [chunk.get("text", "")]
    chunk_skills = ce.pop("chunk_skills", [])  # tÃ¡ch riÃªng, khÃ´ng format cÃ¹ng global fields

    if section == "skills":
        lines += _format_items("EXTRACTED SKILLS", ce.get("technical_skills", []))
        lines += _format_items("EXTRACTED SOFT SKILLS", ce.get("soft_skills", []))
        lines += _format_items("EXTRACTED SKILLS FROM THIS SECTION", chunk_skills)
    elif section == "soft_skills":
        lines += _format_items("EXTRACTED SOFT SKILLS", ce.get("soft_skills", []))
    elif section == "experience":
        lines += _format_items("EXTRACTED SKILLS FROM THIS EXPERIENCE", chunk_skills)
        if ce.get("total_experience_years") is not None:
            lines.append(f"[EXTRACTED TOTAL EXPERIENCE]\n- {ce.get('experience_duration', '0 years')}")
        lines += _format_items("EXTRACTED COMPANIES AND WORK EXPERIENCE", ce.get("companies", []))
    elif section == "projects":
        lines += _format_items("EXTRACTED SKILLS FROM THIS PROJECT", chunk_skills)
        lines += _format_items("EXTRACTED PROJECTS", ce.get("projects", []))
    elif section == "education":
        lines += _format_items("EXTRACTED EDUCATION", ce.get("education", []))
    elif section == "general" and ce.get("education"):
        lines += _format_items("EXTRACTED EDUCATION", ce["education"])
    elif chunk_skills:
        lines += _format_items("EXTRACTED SKILLS FROM THIS SECTION", chunk_skills)

    return clean_text("\n\n".join(l for l in lines if l))

def enrich_chunks_for_embedding(chunks: List[Dict], *, use_llm: bool = True) -> List[Dict]:
    extraction = (
        _local_llm_extract_cv_facts(chunks)
        if use_llm
        else _finalize_experience_extraction(
            _merge_extracted_skills(EMPTY_CV_EXTRACTION.copy(), chunks),
            _build_cv_text_full(chunks),
        )
    )
    cv_experience = {
        "months": extraction.get("total_experience_months", 0),
        "years": extraction.get("total_experience_years", 0),
        "display": extraction.get("experience_duration", "0 years"),
    }
    enriched = []
    for chunk in chunks:
        item = chunk.copy()
        item["cv_experience"] = cv_experience
        item["total_experience_months"] = cv_experience["months"]
        item["total_experience_years"] = cv_experience["years"]
        item["experience_duration"] = cv_experience["display"]
        if not _should_embed(chunk):
            item["skip_embed"] = True
        else:
            if use_llm:
                item["llm_skills"] = _llm_extract_chunk_skills(item)
            ce = _chunk_relevant_extraction(item, extraction)
            compact = {k: v for k, v in ce.items() if v not in (None, "", []) and k != "chunk_skills"}
            chunk_skills = ce.get("chunk_skills", [])
            if chunk_skills: compact["chunk_skills"] = chunk_skills
            if chunk_skills:
                item["skill_text"] = _build_skill_text(chunk_skills, label="CV_SKILLS")
            if compact: item["extracted_info"] = compact
            item["embedding_text"] = _build_embedding_text(item, extraction)
        enriched.append(item)
    return enriched

def cv_pdf_to_chunks(
    pdf_source: str | Path | bytes,
    *,
    max_tokens: int = 512,
    min_chars: int = 80,
    max_chars: int = 1200,
    converter: Optional[DocumentConverter] = None,
    ocr: bool = False,
    enrich_with_llm: bool = True,
) -> List[Dict]:
    if converter is None:
        converter = build_converter(ocr=ocr)
    doc     = _load_document(pdf_source, converter)
    chunker = HybridChunker(max_tokens=max_tokens, merge_peers=True)
    results: List[Dict] = []

    for raw in chunker.chunk(doc):
        chunk_text = (getattr(raw, "text", "") or "").strip()
        if not chunk_text and hasattr(raw, "export_to_text"):
            chunk_text = raw.export_to_text().strip()
        chunk_text = clean_text(chunk_text)
        if len(chunk_text) < 20:
            continue
        headings = [fix_spaced_letters(h).strip()
                    for h in (getattr(getattr(raw, "meta", None), "headings", None) or [])]
        results.append({
            "chunk_index": len(results),
            "section":     headings[-1] if headings else "unknown",
            "text":        chunk_text,
            "headings":    headings,
        })

    results = _merge_small_chunks(results, min_chars=min_chars, max_chars=max_chars)
    results = enrich_chunks_for_embedding(results, use_llm=enrich_with_llm)
    for i, r in enumerate(results):
        r["chunk_index"] = i
    return results

cv_pdf_to_semantic_chunks = cv_pdf_to_chunks

def cv_chunk_text(chunk: Dict[str, Any] | Any, *, prefer_embedding: bool = False) -> str:
    if not isinstance(chunk, dict):
        return str(chunk or "")
    if prefer_embedding:
        return str(chunk.get("embedding_text") or chunk.get("text") or chunk.get("content") or "")
    return str(chunk.get("text") or chunk.get("original_text") or chunk.get("content") or chunk.get("embedding_text") or "")


def cv_chunk_embedding_text(chunk: Dict[str, Any] | Any) -> str:
    return cv_chunk_text(chunk, prefer_embedding=True)


def cv_chunk_skills(chunk: Dict[str, Any] | Any) -> List[str]:
    if not isinstance(chunk, dict):
        return []

    skills: List[str] = []
    info = chunk.get("extracted_info")
    if isinstance(info, dict):
        skills.extend(_clean_string_list(info.get("chunk_skills"), limit=80))
        skills.extend(_clean_string_list(info.get("technical_skills"), limit=80))

    skills.extend(_clean_string_list(chunk.get("llm_skills"), limit=80))

    for line in str(chunk.get("skill_text") or "").splitlines():
        line = line.strip()
        if line.startswith("-"):
            skills.append(line.strip("- ").strip())

    return _dedupe_strings(skills, limit=80)


def normalize_cv_chunks_for_matching(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for index, chunk in enumerate(chunks or []):
        if not isinstance(chunk, dict):
            chunk = {"text": str(chunk or "")}

        item = dict(chunk)
        item["section"] = str(item.get("section") or "unknown")
        item["text"] = cv_chunk_text(item)
        item["embedding_text"] = cv_chunk_embedding_text(item)
        item["chunk_index"] = item.get("chunk_index", index)
        normalized.append(item)
    return normalized


def pdf_to_markdown(
    pdf_source: str | Path | bytes,
    *,
    include_toc: bool = False,
    ocr: bool = False,
    converter: Optional[DocumentConverter] = None,
) -> str:
    if converter is None:
        converter = build_converter(ocr=ocr)
    md = clean_text(_load_document(pdf_source, converter).export_to_markdown())
    if include_toc:
        headings = re.findall(r'^(#{1,3}) (.+)', md, re.MULTILINE)
        if headings:
            toc = ["## Table of Contents\n"] + [
                "  " * (len(h) - 1) + f"- [{t}](#{re.sub(r'[^a-z0-9]+', '-', t.lower()).strip('-')})"
                for h, t in headings
            ]
            md = "\n".join(toc) + "\n\n---\n\n" + md
    return md


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="pdf_utils - Docling CV semantic chunker")
    p.add_argument("pdf")
    p.add_argument("-o", "--output")
    p.add_argument("--chunk",         action="store_true")
    p.add_argument("--json",          action="store_true")
    p.add_argument("--ocr",           action="store_true")
    p.add_argument("--no-llm-enrich", action="store_true")
    p.add_argument("--toc",           action="store_true")
    p.add_argument("--max-tokens",    type=int, default=512)
    p.add_argument("--min-chars",     type=int, default=80)
    args = p.parse_args()
    conv = build_converter(ocr=args.ocr)

    if args.chunk:
        chunks = cv_pdf_to_chunks(args.pdf, max_tokens=args.max_tokens, min_chars=args.min_chars,
                                  converter=conv, enrich_with_llm=not args.no_llm_enrich)
        if args.json:
            out = json.dumps(chunks, ensure_ascii=False, indent=2)
            if args.output:
                Path(args.output).write_text(out, encoding="utf-8")
                print(f"Saved {len(chunks)} chunks -> {args.output}")
            else:
                print(out)
        else:
            print(f"\nTotal chunks: {len(chunks)}\n")
            for c in chunks:
                hdg = " > ".join(c["headings"]) or "(no heading)"
                print(f"{'â”€'*60}\n#{c['chunk_index']}  section={c['section']}  [{hdg}]")
                print(c["text"][:400] + ("..." if len(c["text"]) > 400 else ""))
    else:
        md = pdf_to_markdown(args.pdf, include_toc=args.toc, converter=conv)
        if args.output:
            Path(args.output).write_text(md, encoding="utf-8")
            print(f"Saved: {args.output}")
        else:
            print(md)