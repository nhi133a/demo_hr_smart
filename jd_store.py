import certifi
import os
import re
from datetime import datetime, timezone
from typing import List, Dict

from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne

from bedrock_utils import get_embedding

load_dotenv()

MONGO_CLIENT_OPTS = {
    "maxPoolSize": 5,
    "minPoolSize": 1,
    "maxIdleTimeMS": 45000,
    "retryWrites": False,
}

JD_COLLECTION_NAME = "job_descriptions"
VECTOR_INDEX_NAME  = "jd_vector_index"

JD_SECTION_ALIASES = {
    "requirements": "requirements",
    "yêu cầu": "requirements",
    "mô tả công việc": "requirements",
    "trách nhiệm": "requirements",
    "requirement": "requirements",
    "responsibilities": "requirements",
    "responsibility": "requirements",
    "job description": "requirements",
    "description": "requirements",
    "summary": "requirements",
    "objective": "requirements",
    "skills": "skills",
    "kỹ năng": "skills",
    "kĩ năng": "skills",
    "công nghệ": "skills",
    "công cụ": "skills",
    "skill": "skills",
    "technical skills": "skills",
    "technologies": "skills",
    "tools": "skills",
    "experience": "experience",
    "kinh nghiệm": "experience",
    "kinh nghiệm làm việc": "experience",
    "work experience": "experience",
    "qualification": "experience",
    "qualifications": "experience",
    "education": "education",
    "học vấn": "education",
    "bằng cấp": "education",
    "degree": "education",
    "soft skills": "soft_skills",
    "kỹ năng mềm": "soft_skills",
    "kĩ năng mềm": "soft_skills",
    "soft skill": "soft_skills",
    "personal skills": "soft_skills",
    "benefits": "benefits",
    "benefit": "benefits",
}
SAMPLE_JDS = [
    {
        "id":    "jd_001",
        "title": "Tester Intern / QA Intern",
        "sections": {
            "requirements": "Manual testing, test case design, bug reporting, API testing.",
            "experience":   "Chưa cần kinh nghiệm, ưu tiên có dự án thực tế hoặc internship.",
            "skills":       "Jira, Postman, Selenium cơ bản, đọc hiểu tài liệu kỹ thuật.",
            "soft_skills":  "Tỉ mỉ, cẩn thận, tiếng Anh cơ bản (đọc tài liệu).",
        },
    },
    {
        "id":    "jd_002",
        "title": "Backend Developer Intern",
        "sections": {
            "requirements": "REST API design, database modeling, server-side logic.",
            "experience":   "Có project cá nhân là lợi thế, chưa cần kinh nghiệm thực tế.",
            "skills":       "Python hoặc Node.js, SQL, Git, hiểu HTTP và JSON.",
            "soft_skills":  "Teamwork, chủ động học hỏi, giao tiếp tốt.",
        },
    },
    {
        "id":    "jd_003",
        "title": "Frontend Developer Intern",
        "sections": {
            "requirements": "Xây dựng giao diện web responsive, tích hợp API.",
            "experience":   "Chưa cần kinh nghiệm, ưu tiên có portfolio hoặc dự án thực tế.",
            "skills":       "HTML, CSS, JavaScript, React hoặc Vue.",
            "soft_skills":  "Sáng tạo, chú ý đến UI/UX, giao tiếp tốt.",
        },
    },
    {
        "id":    "jd_004",
        "title": "Data Analyst Intern",
        "sections": {
            "requirements": "Phân tích dữ liệu, xây dựng báo cáo và dashboard.",
            "experience":   "Chưa cần kinh nghiệm, ưu tiên có project phân tích dữ liệu.",
            "skills":       "Excel, SQL, Python (Pandas, Matplotlib), Power BI hoặc Tableau.",
            "soft_skills":  "Tư duy logic, chính xác, trình bày kết quả rõ ràng.",
        },
    },
]


# =============================================================================
# MONGODB CONNECTION
# =============================================================================

_client = None

def get_mongo_client():
    global _client
    if _client is None:
        _client = MongoClient(
            os.getenv("MONGO_URI"),
            tlsCAFile=certifi.where(),
            **MONGO_CLIENT_OPTS,
        )
    return _client

def get_jd_store():
    return get_mongo_client()["aws_rag_db"][JD_COLLECTION_NAME]

def count_indexed_jds() -> int:
    try:
        return get_jd_store().count_documents({})
    except Exception:
        return 0


def _normalize_jd_section(section_name: str) -> str:
    key = re.sub(r"[^\w\s_&/-]+", " ", str(section_name).lower(), flags=re.UNICODE)
    key = re.sub(r"[_&/-]+", " ", key)
    key = re.sub(r"\s+", " ", key).strip()
    return JD_SECTION_ALIASES.get(key, key.replace(" ", "_") or "requirements")


def _split_jd_text_by_headers(text: str) -> Dict[str, str]:
    """
    Split raw JD text into section chunks from common markdown/plain-text headers.
    If no header is found, the whole JD becomes a requirements chunk.
    """
    chunks: Dict[str, List[str]] = {}
    current = "requirements"
    saw_header = False

    for raw_line in str(text).splitlines():
        line = raw_line.strip()
        if not line:
            continue

        inline_match = re.match(r"^([^:]{2,60})\s*:\s*(.+)$", line, re.UNICODE)
        if inline_match:
            candidate = _normalize_jd_section(inline_match.group(1))
            if candidate in set(JD_SECTION_ALIASES.values()) or candidate in {"requirements", "skills", "experience", "education", "soft_skills", "benefits"}:
                current = candidate
                saw_header = True
                chunks.setdefault(current, []).append(inline_match.group(2).strip())
                continue

        header_match = re.match(r"^(?:#{1,4}\s*)?([^:]{2,60})\s*:?\s*$", line, re.UNICODE)
        if header_match:
            candidate = _normalize_jd_section(header_match.group(1))
            if candidate in set(JD_SECTION_ALIASES.values()) or candidate in {"requirements", "skills", "experience", "education", "soft_skills", "benefits"}:
                current = candidate
                saw_header = True
                chunks.setdefault(current, [])
                continue

        chunks.setdefault(current, []).append(line)

    if not saw_header:
        return {"requirements": str(text).strip()} if str(text).strip() else {}

    return {
        section: "\n".join(lines).strip()
        for section, lines in chunks.items()
        if "\n".join(lines).strip()
    }


def jd_to_section_chunks(jd: Dict) -> List[Dict]:
    """
    Normalize any supported JD shape into independent section chunks.
    Supported inputs:
      - {"id", "title", "sections": {"skills": "..."}}
      - {"id", "title", "text": "Requirements:\n..."}
      - {"id", "title", "description": "..."}
    """
    jd_id = str(jd.get("id") or jd.get("jd_id") or jd.get("title") or "jd").strip()
    title = str(jd.get("title") or jd_id).strip()

    if isinstance(jd.get("sections"), dict):
        sections = {
            _normalize_jd_section(name): str(text).strip()
            for name, text in jd["sections"].items()
            if str(text).strip()
        }
    else:
        sections = _split_jd_text_by_headers(jd.get("text") or jd.get("description") or jd.get("content") or "")

    chunks = []
    for idx, (section_name, section_text) in enumerate(sections.items()):
        chunk_id = f"{jd_id}_{idx:02d}_{section_name}"
        embed_text = f"[JD: {title}]\n[SECTION: {section_name}]\n{section_text}"
        chunks.append({
            "chunk_id": chunk_id,
            "jd_id": jd_id,
            "source": jd_id,
            "title": title,
            "section": section_name,
            "section_original": section_name,
            "chunk_index": idx,
            "content": section_text,
            "embedding_text": embed_text,
        })
    return chunks


# =============================================================================
# INGEST — tách từng section thành chunk riêng
# =============================================================================

def _legacy_ingest_jds() -> int:
    """
    Lưu từng section của JD thành một document riêng trong MongoDB.
    Mỗi document có: jd_id, section, title, content, embedding.

    Ví dụ jd_001 có 4 sections → 4 documents trong MongoDB.
    """
    collection  = get_jd_store()
    operations  = []
    chunk_count = 0

    for jd in SAMPLE_JDS:
        for section_name, section_text in jd["sections"].items():
            section_text = section_text.strip()
            if not section_text:
                continue

            # Thêm tiêu đề JD vào context trước khi embed
            # giúp vector hiểu đây là yêu cầu của vị trí nào
            embed_text = f"{jd['title']}\n[{section_name.upper()}]\n{section_text}"
            embedding  = get_embedding(embed_text)

            chunk_id = f"{jd['id']}_{section_name}"   # vd: jd_001_skills

            operations.append(
                UpdateOne(
                    {"chunk_id": chunk_id},
                    {
                        "$set": {
                            "chunk_id":  chunk_id,
                            "jd_id":     jd["id"],
                            "title":     jd["title"],
                            "section":   section_name,
                            "content":   section_text,
                            "embedding": embedding,
                        }
                    },
                    upsert=True,
                )
            )
            chunk_count += 1

    if operations:
        collection.bulk_write(operations, ordered=False)

    # Xoá chunk_id không còn trong danh sách hiện tại
    valid_chunk_ids = [
        f"{jd['id']}_{sec}"
        for jd in SAMPLE_JDS
        for sec in jd["sections"]
    ]
    collection.delete_many({"chunk_id": {"$nin": valid_chunk_ids}})

    return chunk_count


# =============================================================================
# SEARCH — tìm JD chunks gần với CV query nhất
# =============================================================================

def search_similar_jds(query_embedding: list, k: int = 5) -> List[Dict]:
    """
    Tìm k JD chunks gần nhất với query_embedding.

    Trả về list dict gồm: jd_id, title, section, content, score.
    Nhiều chunk của cùng một JD có thể xuất hiện nếu nhiều section đều phù hợp.
    """
    collection = get_jd_store()
    pipeline = [
        {
            "$vectorSearch": {
                "index":          VECTOR_INDEX_NAME,
                "path":           "embedding",
                "queryVector":    query_embedding,
                "numCandidates":  min(max(k * 10, 50), 200),
                "limit":          k,
            }
        },
        {
            "$project": {
                "_id":     0,
                "chunk_id": 1,
                "jd_id":   1,
                "title":   1,
                "source": 1,
                "section": 1,
                "chunk_index": 1,
                "content": 1,
                "embedding_text": 1,
                "score":   {"$meta": "vectorSearchScore"},
            }
        },
    ]
    return list(collection.aggregate(pipeline))


def search_jds_by_section(query_embedding: list, section: str, k: int = 3) -> List[Dict]:
    collection = get_jd_store()
    section = _normalize_jd_section(section)
    pipeline = [
        {
            "$vectorSearch": {
                "index":         VECTOR_INDEX_NAME,
                "path":          "embedding",
                "queryVector":   query_embedding,
                "numCandidates": min(max(k * 10, 50), 200),
                "limit":         k * 3,   # lấy dư để filter section
                "filter":        {"section": section},
            }
        },
        {"$match": {"section": section}},
        {"$limit": k},
        {
            "$project": {
                "_id":     0,
                "chunk_id": 1,
                "jd_id":   1,
                "title":   1,
                "source": 1,
                "section": 1,
                "chunk_index": 1,
                "content": 1,
                "embedding_text": 1,
                "score":   {"$meta": "vectorSearchScore"},
            }
        },
    ]
    return list(collection.aggregate(pipeline))


def search_similar_jd_skills(query_embedding: list, k: int = 5) -> List[Dict]:
    """
    Search JD chunks that are most useful for skill matching.

    The matcher uses this as a second pass with CV skill-only text. Prefer the
    skills section, but keep requirements/experience in the search because many
    JDs describe required skills outside a dedicated Skills header.
    """
    collection = get_jd_store()
    skill_sections = {"skills", "requirements", "experience"}
    pipeline = [
        {
            "$vectorSearch": {
                "index":          VECTOR_INDEX_NAME,
                "path":           "embedding",
                "queryVector":    query_embedding,
                "numCandidates":  min(max(k * 12, 60), 200),
                "limit":          k * 4,
            }
        },
        {
            "$project": {
                "_id": 0,
                "chunk_id": 1,
                "jd_id": 1,
                "title": 1,
                "source": 1,
                "section": 1,
                "chunk_index": 1,
                "content": 1,
                "embedding_text": 1,
                "llm_skills": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]
    hits = list(collection.aggregate(pipeline))
    preferred = [
        hit for hit in hits
        if str(hit.get("section") or "").strip().lower() in skill_sections
    ]
    return (preferred or hits)[:k]


def get_top_jd_ids(query_embedding: list, k: int = 3) -> List[str]:
    """
    Trả về jd_id của top k JD phù hợp nhất (không trùng lặp).
    Dùng khi chỉ cần biết JD nào phù hợp, không cần chi tiết từng chunk.
    """
    chunks   = search_similar_jds(query_embedding, k=k * 4)
    seen     = set()
    jd_ids   = []

    for chunk in chunks:
        jid = chunk["jd_id"]
        if jid not in seen:
            seen.add(jid)
            jd_ids.append(jid)
        if len(jd_ids) >= k:
            break

    return jd_ids


def get_jd_chunks(jd_id: str) -> List[Dict]:
    collection = get_jd_store()
    return list(collection.find(
        {"jd_id": jd_id},
        {
            "_id": 0,
            "chunk_id": 1,
            "jd_id": 1,
            "title": 1,
            "source": 1,
            "section": 1,
            "chunk_index": 1,
            "content": 1,
            "embedding_text": 1,
            "llm_skills": 1,
        },
    ).sort("chunk_index", 1))


# Final public ingestion implementation. Kept at the end so it overrides the
# older sample-only implementation above.
def ingest_jds(jds: List[Dict] | None = None, *, prune_missing: bool = True) -> int:
    """Store each JD section as a separate vector-searchable chunk."""
    jds = jds or SAMPLE_JDS
    collection = get_jd_store()
    operations = []
    valid_chunk_ids = []
    chunk_count = 0

    for jd in jds:
        for chunk in jd_to_section_chunks(jd):
            embedding = get_embedding(chunk["embedding_text"])
            valid_chunk_ids.append(chunk["chunk_id"])
            operations.append(
                UpdateOne(
                    {"chunk_id": chunk["chunk_id"]},
                    {
                        "$set": {
                            **chunk,
                            "embedding": embedding,
                            "updated_at": datetime.now(timezone.utc),
                        }
                    },
                    upsert=True,
                )
            )
            chunk_count += 1

    if operations:
        collection.bulk_write(operations, ordered=False)

    if prune_missing and valid_chunk_ids:
        collection.delete_many({"chunk_id": {"$nin": valid_chunk_ids}})

    return chunk_count
