import json
import re
import html
import tempfile
import os
from pathlib import Path
from typing import Any, List, Dict, Optional

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling.chunking import HybridChunker


# ── Converter ─────────────────────────────────────────────────────────────────

def build_converter(*, ocr: bool = False, table_structure: bool = True) -> DocumentConverter:
    opts = PdfPipelineOptions(do_ocr=ocr, do_table_structure=table_structure)
    return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})


# ── Text helpers ──────────────────────────────────────────────────────────────

def fix_spaced_letters(text: str) -> str:
    return re.sub(r'\b(?:[A-Z]\s){3,}[A-Z]\b', lambda m: m.group(0).replace(" ", ""), text)

def clean_text(text: str) -> str:
    text = html.unescape(text)
    text = fix_spaced_letters(text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ── Document loading ──────────────────────────────────────────────────────────

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


# ── Chunk helpers ─────────────────────────────────────────────────────────────

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


# ── PII / embed filtering ─────────────────────────────────────────────────────
# FIX 1: mỗi section là phần tử set riêng biệt (code cũ là 1 string khổng lồ)
# FIX 2: không skip nếu PII block chứa education/skills

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
_RE_VALUABLE = re.compile(
    r"\b(university|college|school|degree|gpa|bachelor|master|phd|intern|project|"
    r"skill|experience|worked|years?|trường|đại\s+học|kỹ\s+năng|kinh\s+nghiệm|dự\s+án)\b",
    re.IGNORECASE,
)

def _is_pii_section(chunk: Dict) -> bool:
    label = " ".join([str(chunk.get("section", "")), *chunk.get("headings", [])]).lower()
    return bool(_RE_PII.search(label))

def _should_embed(chunk: Dict) -> bool:
    if not _is_pii_section(chunk):
        return True
    return bool(_RE_VALUABLE.search(chunk.get("text", "")))


# ── Extraction schema & helpers ───────────────────────────────────────────────

EMPTY_CV_EXTRACTION: Dict[str, Any] = {
    "technical_skills": [], "soft_skills": [], "companies": [],
    "total_experience_years": None, "projects": [], "education": [],
    "chunk_skills": [],
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

def _dedupe_strings(items: List[str], limit: int = 40) -> List[str]:
    seen, result = set(), []
    for item in items:
        item = clean_text(item)
        if item and item.lower() not in seen:
            result.append(item); seen.add(item.lower())
    return result[:limit]


# ── FIX 3: Company fuzzy matching ─────────────────────────────────────────────
# Cũ: exact substring → miss alias, legal suffix, dấu câu

def _normalize_company_name(name: str) -> str:
    name = re.sub(
        r'\b(inc|llc|ltd|limited|corp|corporation|co|company|group|holdings?|'
        r'joint\s+stock|jsc|tnhh|cp|cty|công\s+ty)\b\.?',
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


# ── Skill extraction helpers ──────────────────────────────────────────────────
# Skills are extracted by the LLM. Local code only maps extracted skills back to
# the chunks where they appear, so embedding_text stays explicit.

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

def _merge_extracted_skills(extraction: Dict[str, Any], chunks: List[Dict]) -> Dict[str, Any]:
    merged = extraction.copy()
    merged["technical_skills"] = _dedupe_strings(extraction.get("technical_skills", []), limit=80)
    merged["soft_skills"] = _dedupe_strings(extraction.get("soft_skills", []), limit=30)
    return merged


# ── Validation ────────────────────────────────────────────────────────────────

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


# ── FIX 5: LLM multi-pass extraction ─────────────────────────────────────────
# Cũ: single-pass 12k chars, lọc PII trước LLM → bỏ sót education trong personal block
# Mới: ưu tiên section quan trọng, 16k chars, pass 2 targeted cho field còn thiếu

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
    candidates = [raw]
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if m:
        candidates.append(m.group(0))
    obj = _extract_json_object(raw)
    if obj:
        return _clean_string_list(obj.get("technical_skills") or obj.get("skills") or obj.get("_"))
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
technical_skills: word-for-word tools/languages/frameworks/testing methods/QA tools mentioned anywhere, including Experience and Projects.
soft_skills: only human traits such as responsibility, teamwork, communication, curiosity.
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

_PROMPT_SKILLS = """Extract ONLY technical skills from this CV.
Include tools, technologies, programming languages, frameworks, databases, platforms, QA/testing methods, and skills found in Experience/Projects/Objective sections.
Do NOT include soft skills/personality traits.
Return ONLY a JSON array of strings.
CV:\n{cv_text}"""

def _local_llm_extract_cv_facts(chunks: List[Dict], max_chars: int = 16000) -> Dict[str, Any]:
    if not chunks:
        return EMPTY_CV_EXTRACTION.copy()
    cv_text = _build_cv_text_full(chunks, max_chars=max_chars)

    # Pass 1: full extraction
    extraction = _normalize_extraction(_extract_json_object(_llm_call(_PROMPT_FULL.format(cv_text=cv_text))))
    extraction = _validate_extraction_against_text(extraction, cv_text)

    # Pass 2: targeted re-extraction cho field còn thiếu
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

    return _merge_extracted_skills(extraction, chunks)


# ── Section detection & chunk enrichment ─────────────────────────────────────

def _normalized_cv_section(chunk: Dict) -> str:
    text = _label_for_chunk(chunk).lower()
    if re.search(r"\b(education|academic|university|college|school|gpa|degree)\b", text): return "education"
    if re.search(r"\b(project|portfolio)\b", text): return "projects"
    if re.search(r"\b(objective|summary)\b", text): return "general"
    if re.search(r"\b(experience|employment|work|career|intern|internship|company|qa|qc|tester)\b", text): return "experience"
    if _is_soft_skill_section_label(text): return "soft_skills"
    if re.search(r"\b(skills?|technical|tools?|technology|competenc|strength)\b", text): return "skills"
    return "general"

def _matching_projects_for_chunk(chunk: Dict, extraction: Dict[str, Any]) -> List[Dict]:
    haystack = " ".join([chunk.get("section",""), *chunk.get("headings",[]), chunk.get("text","")[:200]]).lower()
    return [p for p in extraction.get("projects", [])
            if (name := str(p.get("name","")).lower()) and name in haystack]

def _chunk_extracted_skills(chunk: Dict, extraction: Dict[str, Any]) -> List[str]:
    label = _label_for_chunk(chunk).lower()
    text = chunk.get("text", "")
    section = _normalized_cv_section(chunk)
    skills: List[str] = []

    if section in ("skills", "experience", "projects") or _RE_MEANINGFUL_SKILL_LABEL.search(label):
        for skill in extraction.get("technical_skills", []):
            if _contains_skill(text.lower(), skill):
                skills.append(skill)

    if section == "projects":
        for project in _matching_projects_for_chunk(chunk, extraction) or extraction.get("projects", []):
            skills.extend(_clean_string_list(project.get("technologies"), limit=20))

    chunk_label = " ".join([chunk.get("section",""), *chunk.get("headings",[]), text[:300]])
    for company in extraction.get("companies", []):
        if company.get("name") and _company_name_matches(company["name"], chunk_label):
            skills.extend(_clean_string_list(company.get("skills"), limit=20))

    return _dedupe_strings(skills, limit=40)

def _chunk_relevant_extraction(chunk: Dict, extraction: Dict[str, Any]) -> Dict[str, Any]:
    section  = _normalized_cv_section(chunk)
    relevant = EMPTY_CV_EXTRACTION.copy()
    chunk_skills = _chunk_extracted_skills(chunk, extraction)
    if chunk_skills:
        relevant["chunk_skills"] = chunk_skills

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
        if matched: relevant["total_experience_years"] = extraction.get("total_experience_years")

    elif section == "projects":
        matched = _matching_projects_for_chunk(chunk, extraction)
        relevant["projects"] = matched or extraction.get("projects", [])

    elif section == "education":
        relevant["education"] = extraction.get("education", [])

    elif section == "general":
        # FIX 2 extension: education ẩn trong general/personal block
        if re.search(r"\b(university|college|school|degree|bachelor|master|phd|trường|đại\s+học)\b",
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
    if section == "skills":
        lines += _format_items("EXTRACTED SKILLS", ce.get("technical_skills", []))
        lines += _format_items("EXTRACTED SOFT SKILLS", ce.get("soft_skills", []))
        lines += _format_items("EXTRACTED SKILLS FROM THIS SECTION", ce.get("chunk_skills", []))
    elif section == "soft_skills":
        lines += _format_items("EXTRACTED SOFT SKILLS", ce.get("soft_skills", []))
    elif section == "experience":
        lines += _format_items("EXTRACTED SKILLS FROM THIS EXPERIENCE", ce.get("chunk_skills", []))
        if ce.get("total_experience_years") is not None:
            lines.append(f"[EXTRACTED TOTAL EXPERIENCE YEARS]\n- {ce['total_experience_years']}")
        lines += _format_items("EXTRACTED COMPANIES AND WORK EXPERIENCE", ce.get("companies", []))
    elif section == "projects":
        lines += _format_items("EXTRACTED SKILLS FROM THIS PROJECT", ce.get("chunk_skills", []))
        lines += _format_items("EXTRACTED PROJECTS", ce.get("projects", []))
    elif section == "education":
        lines += _format_items(f"EXTRACTED {section.upper()}", ce.get(section, []))
    elif section == "general" and ce.get("education"):
        lines += _format_items("EXTRACTED EDUCATION", ce["education"])
    elif ce.get("chunk_skills"):
        lines += _format_items("EXTRACTED SKILLS FROM THIS SECTION", ce["chunk_skills"])

    return clean_text("\n\n".join(l for l in lines if l))

def enrich_chunks_for_embedding(chunks: List[Dict], *, use_llm: bool = True) -> List[Dict]:
    extraction = (
        _local_llm_extract_cv_facts(chunks)
        if use_llm
        else _merge_extracted_skills(EMPTY_CV_EXTRACTION.copy(), chunks)
    )
    enriched   = []
    for chunk in chunks:
        item = chunk.copy()
        if not _should_embed(chunk):
            item["skip_embed"] = True
        else:
            ce = _chunk_relevant_extraction(item, extraction)
            compact = {k: v for k, v in ce.items() if v not in (None, "", [])}
            if compact: item["extracted_info"] = compact
            item["embedding_text"] = _build_embedding_text(item, extraction)
        enriched.append(item)
    return enriched


# ── Public API ────────────────────────────────────────────────────────────────

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

cv_pdf_to_semantic_chunks = cv_pdf_to_chunks  # alias

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


# ── CLI ───────────────────────────────────────────────────────────────────────

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
                print(f"{'─'*60}\n#{c['chunk_index']}  section={c['section']}  [{hdg}]")
                print(c["text"][:400] + ("..." if len(c["text"]) > 400 else ""))
    else:
        md = pdf_to_markdown(args.pdf, include_toc=args.toc, converter=conv)
        if args.output:
            Path(args.output).write_text(md, encoding="utf-8")
            print(f"Saved: {args.output}")
        else:
            print(md)
