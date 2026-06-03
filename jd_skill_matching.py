import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any, Dict, List


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


def _dedupe_keep_order(items: List[Any]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        text = re.sub(r"\s+", " ", str(item or "")).strip(" -,*.;:")
        key = _skill_key(text)
        if text and key and key not in seen:
            seen.add(key)
            result.append(text)
    return result


def _row_name(row: Dict) -> str:
    if isinstance(row, dict):
        return str(row.get("name") or "").strip()
    return str(getattr(row, "name", "") or "").strip()


def _row_value(row: Any, key: str, default: Any = "") -> Any:
    return row.get(key, default) if isinstance(row, dict) else getattr(row, key, default)


NOISE_REQUIREMENT_KEYS = {
    "softskills",
    "softskill",
    "workcontext",
    "competencies",
    "competency",
    "requirements",
    "requiredskills",
    "preferredskills",
    "skills",
    "tools",
    "technologies",
    "programminglanguages",
    "databases",
    "responsibilities",
    "jobrequirements",
    "jobresponsibilities",
}


def _is_noise_requirement(value: str) -> bool:
    key = _skill_key(str(value or "").strip("[](){}: "))
    return key in NOISE_REQUIREMENT_KEYS


def _row_weight(row: Dict) -> float:
    importance = str(_row_value(row, "importance") or "").lower()
    confidence = str(_row_value(row, "confidence") or "").lower()
    weight = 1.25 if importance in {"must", "required", "high"} else 1.0
    if confidence == "low":
        weight *= 0.75
    elif confidence == "high":
        weight *= 1.10
    return weight


def _token_set(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9+#]+", _norm_text(value)))


SKILL_TEXT_EVIDENCE = {
    "restapidesign": (
        r"\brest(?:ful)?\s+apis?\b",
        r"\brest\s+endpoints?\b",
        r"\bapi\s+(?:design|development|endpoints?)\b",
        r"\bapi\s+nodes?\b",
        r"\bhttp\s+json\s+endpoints?\b",
        r"\bfastapi\b",
    ),
    "databasemodeling": (
        r"\bdatabase\s+(?:model(?:ing|s)?|design|layouts?|schemas?)\b",
        r"\b(?:relational|non relational)\s+(?:data\s+)?model(?:ing|s)?\b",
        r"\b(?:entity|table)\s+(?:relationships?|schemas?)\b",
    ),
    "serversidelogic": (
        r"\bserver\s+side\s+(?:logic|components?|services?|development)\b",
        r"\bbackend\s+(?:logic|components?|services?|development)\b",
    ),
    "teamwork": (
        r"\bteam(?:work| collaboration)\b",
        r"\bcollaborat(?:e|ed|ing|ion)\b",
        r"\bworked\s+(?:with|in)\s+(?:a\s+)?team\b",
    ),
    "selflearning": (
        r"\bself\s+(?:learning|study|taught)\b",
        r"\beager(?:ness)?\s+to\s+learn\b",
        r"\blearn(?:ed|ing)?\s+new\s+(?:technologies|tools|skills)\b",
    ),
    "communication": (
        r"\bcommunicat(?:e|ed|ing|ion)\b",
        r"\bclear\s+(?:presentation|reporting|documentation)\b",
    ),
    "goodcommunication": (
        r"\bcommunicat(?:e|ed|ing|ion)\b",
        r"\bclear\s+(?:presentation|reporting|documentation)\b",
    ),
    "manualtesting": (r"\bmanual\s+test(?:ing| cases?)\b",),
    "manualtestingprocesses": (r"\bmanual\s+test(?:ing| cases?)\b",),
    "functionaltesting": (
        r"\bfunctional\s+testing\b",
        r"\bfunctional\b(?:\s+[a-z0-9]+){0,5}\s+testing\b",
    ),
    "defectreporting": (
        r"\b(?:report(?:ed|ing)?|document(?:ed|ation)?|track(?:ed|ing)?)\b(?:\s+[a-z0-9]+){0,6}\s+defects?\b",
        r"\bdefects?\b(?:\s+[a-z0-9]+){0,6}\s+(?:report(?:ed|ing)?|document(?:ed|ation)?|track(?:ed|ing)?)\b",
    ),
    "bugtracking": (
        r"\b(?:track(?:ed|ing)?|reproduc(?:e|ed|ing))\b(?:\s+[a-z0-9]+){0,6}\s+(?:bugs?|defects?)\b",
        r"\b(?:bugs?|defects?)\b(?:\s+[a-z0-9]+){0,6}\s+track(?:ed|ing)?\b",
    ),
    "proactiveness": (r"\bproactiv(?:e|ely|eness)\b",),
    "dataanalysis": (
        r"\bdata\s+(?:analysis|analytics|processing|preprocessing|visuali[sz]ation)\b",
        r"\banaly[sz](?:e|ed|ing)\s+data\b",
        r"\b(?:pandas|numpy)\b",
    ),
    "reporting": (
        r"\breports?\b",
        r"\breporting\b",
        r"\bdashboards?\b",
        r"\b(?:power\s*bi|tableau)\b",
    ),
    "dashboards": (
        r"\bdashboards?\b",
        r"\bdata\s+visuali[sz]ation\b",
        r"\b(?:power\s*bi|tableau)\b",
    ),
    "datavisualization": (
        r"\bdata\s+visuali[sz]ation\b",
        r"\b(?:matplotlib|seaborn|power\s*bi|tableau)\b",
    ),
}


def _skills_match(jd_skill: str, cv_skill: str) -> bool:
    left_key = _skill_key(jd_skill)
    right_key = _skill_key(cv_skill)
    if not left_key or not right_key:
        return False
    if left_key == right_key:
        return True
    left_tokens = _token_set(jd_skill)
    right_tokens = _token_set(cv_skill)
    if left_tokens and right_tokens and left_tokens <= right_tokens:
        return True
    return SequenceMatcher(None, left_key, right_key).ratio() >= 0.90


def _snippet_for_pattern(text: str, pattern: str) -> str:
    for snippet in re.split(r"(?:\r?\n)+|(?<=[.!?])\s+", str(text or "")):
        if re.search(pattern, _norm_text(snippet), re.I):
            return re.sub(r"\s+", " ", snippet).strip()[:500]
    return ""


def _skill_text_match(jd_skill: str, cv_text: str) -> Dict[str, str] | None:
    normalized_cv_text = _norm_text(cv_text)
    normalized_skill = _norm_text(jd_skill)
    if not normalized_cv_text or not normalized_skill:
        return None

    exact_pattern = rf"(?<![a-z0-9]){re.escape(normalized_skill)}(?![a-z0-9])"
    if re.search(exact_pattern, normalized_cv_text):
        return {
            "name": jd_skill,
            "evidence": _snippet_for_pattern(cv_text, exact_pattern) or jd_skill,
            "section": "raw_text",
        }

    for pattern in SKILL_TEXT_EVIDENCE.get(_skill_key(jd_skill), ()):
        if re.search(pattern, normalized_cv_text, re.I):
            return {
                "name": jd_skill,
                "evidence": _snippet_for_pattern(cv_text, pattern) or jd_skill,
                "section": "raw_text",
            }
    return None


def _match_rows(jd_rows: List[Dict], cv_rows: List[Dict], cv_text: str) -> tuple[float, List[str], List[str], Dict[str, Dict]]:
    if not jd_rows:
        return 1.0, [], [], {}

    matched = []
    missing = []
    evidence_by_skill = {}
    total_weight = 0.0
    matched_weight = 0.0
    processed_alternative_keys = set()

    def row_match(row: Dict) -> Dict[str, str] | Dict | None:
        name = _row_name(row)
        match = next((cv_row for cv_row in cv_rows if _skills_match(name, _row_name(cv_row))), None)
        return match or _skill_text_match(name, cv_text)

    for jd_row in jd_rows:
        name = _row_name(jd_row)
        if not name or _is_noise_requirement(name):
            continue
        evidence = str(_row_value(jd_row, "evidence") or "")
        if _is_noise_requirement(evidence):
            continue
        alternative_key = _skill_key(evidence) if re.search(r"\bor\b", evidence, re.I) else ""
        alternative_rows = []
        if alternative_key:
            if alternative_key in processed_alternative_keys:
                continue
            processed_alternative_keys.add(alternative_key)
            alternative_rows = [
                row
                for row in jd_rows
                if not _is_noise_requirement(_row_name(row))
                and _skill_key(str(_row_value(row, "evidence") or "")) == alternative_key
            ]

        if alternative_rows:
            total_weight += max(_row_weight(row) for row in alternative_rows)
            alternatives = []
            for row in alternative_rows:
                alt_name = _row_name(row)
                match_row = row_match(row)
                if match_row:
                    alternatives.append((alt_name, row, match_row))
            if alternatives:
                alt_name, _, match_row = alternatives[0]
                matched.append(alt_name)
                matched_weight += max(_row_weight(row) for row in alternative_rows)
                evidence_by_skill[alt_name] = match_row
            else:
                missing.append(evidence or " / ".join(_row_name(row) for row in alternative_rows if _row_name(row)))
            continue

        total_weight += _row_weight(jd_row)
        match_row = row_match(jd_row)
        if match_row:
            matched.append(name)
            matched_weight += _row_weight(jd_row)
            evidence_by_skill[name] = match_row
        else:
            missing.append(name)

    matched = [item for item in _dedupe_keep_order(matched) if not _is_noise_requirement(item)]
    missing = [item for item in _dedupe_keep_order(missing) if not _is_noise_requirement(item)]
    return matched_weight / max(total_weight, 1.0), matched, missing, evidence_by_skill
