import json
import re
from typing import List


def generate_answer(prompt):
    try:
        import ollama
    except ImportError as e:
        raise RuntimeError(
            "Missing Python package 'ollama'. Install it with: pip install ollama"
        ) from e

    response = ollama.chat(
        model="qwen2.5:1.5b",
        options={"temperature": 0},
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]


def _parse_string_array(raw: str) -> List[str]:
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", (raw or "").strip(), flags=re.IGNORECASE)
    candidates = [raw]
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        candidates.insert(0, match.group(0))

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
            if isinstance(parsed, dict):
                value = parsed.get("skills") or parsed.get("required_skills") or parsed.get("demonstrated_skills")
                if isinstance(value, list):
                    return [str(item).strip() for item in value if str(item).strip()]
        except Exception:
            pass

    return [line.strip("-* \t") for line in raw.splitlines() if line.strip("-* \t")]


def extract_normalized_skills(text: str, *, source_type: str, section: str = "") -> List[str]:
    prompt = f"""Extract normalized skills from this {source_type} chunk.

Rules:
- Infer skills from context, not only explicit keyword lists.
- Normalize equivalent wording into a concise canonical skill name.
- Include technical skills, tools, methods, domain practices, platforms, languages, frameworks, and role-relevant competencies.
- Exclude personal traits unless the chunk is specifically about soft skills.
- Do not invent skills unsupported by the text.
- Return ONLY a JSON array of strings.

Section: {section or "unknown"}
Text:
{text[:3000]}"""

    return _parse_string_array(generate_answer(prompt))[:40]


def build_skill_text(skills: List[str], *, label: str) -> str:
    unique = []
    seen = set()
    for skill in skills:
        value = re.sub(r"\s+", " ", str(skill)).strip()
        key = value.lower()
        if value and key not in seen:
            unique.append(value)
            seen.add(key)
    if not unique:
        return ""
    return f"[{label}]\n" + "\n".join(f"- {skill}" for skill in unique)
