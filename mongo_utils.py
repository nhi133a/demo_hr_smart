import os
import logging
from datetime import datetime, timezone
import certifi
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()
log = logging.getLogger(__name__)

# Connection pooling for better performance
MONGO_CLIENT_OPTS = {
    "maxPoolSize": 5,
    "minPoolSize": 1,
    "maxIdleTimeMS": 45000,
    "retryWrites": False  # Disable for better performance
}

_client = None
_collection = None
_profiles_collection = None

DB_NAME = "aws_rag_db"
CV_CHUNKS_COLLECTION_NAME = "cv_chunks"
CANDIDATE_PROFILES_COLLECTION_NAME = "candidates_profile"
CV_VECTOR_INDEX_NAME = "vector_index"
CV_VECTOR_FIELD = "embedding"


def get_collection():
    global _client, _collection
    if _collection is None:
        _client = MongoClient(os.getenv("MONGO_URI"), tlsCAFile=certifi.where(), **MONGO_CLIENT_OPTS)
        _collection = _client[DB_NAME][CV_CHUNKS_COLLECTION_NAME]
    return _collection


def get_profiles_collection():
    global _client, _profiles_collection
    if _profiles_collection is None:
        if _client is None:
            _client = MongoClient(os.getenv("MONGO_URI"), tlsCAFile=certifi.where(), **MONGO_CLIENT_OPTS)
        _profiles_collection = _client[DB_NAME][CANDIDATE_PROFILES_COLLECTION_NAME]
    return _profiles_collection


def _profile_skill_names(schema: dict) -> list[str]:
    seen = set()
    names = []
    keys = ("skills",) if isinstance(schema, dict) and schema.get("skills") else ("required_skills", "preferred_skills")
    for key in keys:
        for row in schema.get(key, []) if isinstance(schema, dict) else []:
            name = str(row.get("name") if isinstance(row, dict) else row or "").strip()
            lower = name.lower()
            if name and lower not in seen:
                names.append(name)
                seen.add(lower)
    return names


def upsert_candidate_profile(
    source_name: str,
    schema: dict,
    *,
    file_hash: str = None,
    original_filename: str = None,
    extra_metadata: dict = None,
) -> None:
    candidate = schema.get("candidate") if isinstance(schema, dict) else {}
    candidate = candidate if isinstance(candidate, dict) else {}
    metadata = schema.get("metadata_filter") if isinstance(schema, dict) else {}
    metadata = metadata if isinstance(metadata, dict) else {}
    years = float(schema.get("total_experience_years") or schema.get("experience_years") or 0)
    doc = {
        "source": source_name,
        "cv_id": candidate.get("cv_id") or source_name,
        "candidate_name": candidate.get("name") or "",
        "total_experience_years": years,
        "skills": _profile_skill_names(schema),
        "education": metadata.get("education_min"),
        "cv_schema": schema,
        "updated_at": datetime.now(timezone.utc),
    }
    if file_hash:
        doc["file_hash"] = file_hash
    if original_filename:
        doc["original_filename"] = original_filename
    if isinstance(extra_metadata, dict):
        doc.update({key: value for key, value in extra_metadata.items() if value is not None})
    get_profiles_collection().update_one({"source": source_name}, {"$set": doc}, upsert=True)


def get_candidate_profile(source_name: str) -> dict:
    try:
        return get_profiles_collection().find_one({"source": source_name}, {"_id": 0}) or {}
    except Exception:
        return {}


def insert_chunks(chunks_with_embeddings, source_name, *, file_hash=None, original_filename=None, extra_metadata=None):
    """
    Optimized batch insert with minimal processing
    """
    docs = []

    for i, item in enumerate(chunks_with_embeddings):
        if len(item) == 3:
            chunk, embedding, metadata = item
        else:
            chunk, embedding = item
            metadata = {}

        doc = {
            "content": chunk,
            "text": chunk,
            "original_text": chunk,
            "embedding_text": chunk,
            "embedding": embedding,
            "source": source_name,
            "chunk_index": i,
            "uploaded_at": datetime.now(timezone.utc),
        }

        if file_hash:
            doc["file_hash"] = file_hash
        if original_filename:
            doc["original_filename"] = original_filename
        if isinstance(extra_metadata, dict):
            doc.update({key: value for key, value in extra_metadata.items() if value is not None})

        if metadata:
            doc.update(metadata)

        # Extract name only from first chunk for efficiency
        if i == 0 and "[NAME]" in chunk and "candidate_name" not in doc:
            try:
                name_start = chunk.index("[NAME]") + 6
                name_end = chunk.index("\n", name_start) if "\n" in chunk[name_start:] else len(chunk)
                doc["candidate_name"] = chunk[name_start:name_end].strip()
            except:
                pass
        
        docs.append(doc)
    
    # Batch insert with optimized batch size
    if docs:
        try:
            get_collection().delete_many({"source": source_name})
            get_collection().insert_many(docs, ordered=False)  # Faster, no order guarantee needed
        except Exception:
            # Fallback to ordered insert if unordered fails
            get_collection().delete_many({"source": source_name})
            get_collection().insert_many(docs, ordered=True)


def search_similar_chunks(query_embedding, k=10, source_filter: str = None):
    num_candidates = min(max(k * 3, 15), 50)  # Scale with k, max 50
    
    vector_search = {
        "index":        CV_VECTOR_INDEX_NAME,
        "path":         CV_VECTOR_FIELD,
        "queryVector":  query_embedding,
        "numCandidates": num_candidates,  # Optimized from 100
        "limit":        k,
    }

    if source_filter:
        vector_search["filter"] = {"source": {"$eq": source_filter}}

    pipeline = [{"$vectorSearch": vector_search}]
    
    # Optimize the pipeline - project only needed fields
    pipeline.append({
        "$project": {
            "content": {"$ifNull": ["$content", "$text"]},
            "text": 1,
            "source": 1,
            "section": 1,
            "candidate_name": 1,
            "_id": 0,
            "score": {"$meta": "vectorSearchScore"}
        }
    })
    
    try:
        return list(get_collection().aggregate(pipeline))
    except Exception as exc:
        log.warning("CV chat vector retrieval failed: %s: %s", type(exc).__name__, exc)
        return []


def search_similar_cv_chunks(query_embedding, k=10, source_filter: str = None):
    num_candidates = min(max(k * 10, 50), 300)

    vector_search = {
        "index": CV_VECTOR_INDEX_NAME,
        "path": CV_VECTOR_FIELD,
        "queryVector": query_embedding,
        "numCandidates": num_candidates,
        "limit": k,
    }
    if source_filter:
        vector_search["filter"] = {"source": {"$eq": source_filter}}

    pipeline = [
        {"$vectorSearch": vector_search},
        {
            "$project": {
                "_id": 0,
                "content": {"$ifNull": ["$content", "$text"]},
                "source": 1,
                "section": 1,
                "chunk_index": 1,
                "candidate_name": 1,
                "original_filename": 1,
                "text": 1,
                "original_text": 1,
                "embedding_text": 1,
                "skill_text": 1,
                "extracted_info": 1,
                "cv_experience": 1,
                "cv_schema": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]

    try:
        return list(get_collection().aggregate(pipeline))
    except Exception as exc:
        log.warning("CV matching vector retrieval failed: %s: %s", type(exc).__name__, exc)
        return []


def get_chunks_by_source_for_matching(source_name: str) -> list[dict]:
    try:
        docs = get_collection().find(
            {"source": source_name},
            {
                "_id": 0,
                "content": 1,
                "text": 1,
                "embedding": 1,
                "vector_embedding": 1,
                "source": 1,
                "chunk_index": 1,
                "section": 1,
                "original_text": 1,
                "embedding_text": 1,
                "skill_text": 1,
                "extracted_info": 1,
                "skip_embed": 1,
                "headings": 1,
                "llm_skills": 1,
                "raw_llm_section": 1,
                "llm_section": 1,
                "candidate_name": 1,
                "cv_experience": 1,
                "cv_schema": 1,
                "chunk_experience_months": 1,
                "chunk_experience_years": 1,
                "chunk_experience_duration": 1,
                "applied_jd_id": 1,
                "applied_jd_ids": 1,
                "applied_role": 1,
                "applied_level": 1,
            },
        ).sort("chunk_index", 1)
    except Exception:
        return []

    chunks = []
    profile = get_candidate_profile(source_name)
    for doc in docs:
        original_text = doc.get("text") or doc.get("original_text") or doc.get("content") or ""
        embedding_text = doc.get("embedding_text") or doc.get("content") or original_text

        chunk = {
            "source": doc.get("source"),
            "section": doc.get("section", "unknown"),
            "text": original_text,
            "embedding_text": embedding_text,
            "embedding": doc.get("embedding") or doc.get("vector_embedding"),
            "chunk_index": doc.get("chunk_index"),
        }

        for key in (
            "skill_text",
            "extracted_info",
            "skip_embed",
            "headings",
            "llm_skills",
            "raw_llm_section",
            "llm_section",
            "candidate_name",
            "cv_experience",
            "chunk_experience_months",
            "chunk_experience_years",
            "chunk_experience_duration",
            "applied_jd_id",
            "applied_jd_ids",
            "applied_role",
            "applied_level",
            "cv_schema",
        ):
            if key in doc:
                chunk[key] = doc[key]

        if not chunks and isinstance(profile.get("cv_schema"), dict):
            chunk["cv_schema"] = profile["cv_schema"]
        if profile.get("candidate_name") and not chunk.get("candidate_name"):
            chunk["candidate_name"] = profile["candidate_name"]
        for key in ("applied_jd_id", "applied_jd_ids", "applied_role", "applied_level"):
            if profile.get(key) and not chunk.get(key):
                chunk[key] = profile[key]
        chunks.append(chunk)

    return chunks


def get_all_cv_chunks_for_matching() -> list[dict]:
    try:
        docs = get_collection().find(
            {},
            {
                "_id": 0,
                "content": 1,
                "text": 1,
                "source": 1,
                "chunk_index": 1,
                "section": 1,
                "original_text": 1,
                "embedding_text": 1,
                "skill_text": 1,
                "extracted_info": 1,
                "llm_skills": 1,
                "candidate_name": 1,
                "cv_experience": 1,
                "cv_schema": 1,
                "applied_jd_id": 1,
                "applied_jd_ids": 1,
                "applied_role": 1,
                "applied_level": 1,
            },
        ).sort([("source", 1), ("chunk_index", 1)])
    except Exception:
        return []

    chunks = []
    profiles: dict[str, dict] = {}
    for doc in docs:
        source = doc.get("source")
        if source not in profiles:
            profiles[source] = get_candidate_profile(source)
        profile = profiles.get(source) or {}
        cv_schema = doc.get("cv_schema")
        if not isinstance(cv_schema, dict) and doc.get("chunk_index") == 0 and isinstance(profile.get("cv_schema"), dict):
            cv_schema = profile["cv_schema"]
        chunks.append({
            "source": source,
            "section": doc.get("section", "unknown"),
            "text": doc.get("text") or doc.get("original_text") or doc.get("content") or "",
            "embedding_text": doc.get("embedding_text") or doc.get("content") or "",
            "skill_text": doc.get("skill_text") or "",
            "extracted_info": doc.get("extracted_info") or {},
            "llm_skills": doc.get("llm_skills") or [],
            "candidate_name": doc.get("candidate_name") or profile.get("candidate_name"),
            "cv_experience": doc.get("cv_experience"),
            "cv_schema": cv_schema,
            "applied_jd_id": doc.get("applied_jd_id") or profile.get("applied_jd_id"),
            "applied_jd_ids": doc.get("applied_jd_ids") or profile.get("applied_jd_ids"),
            "applied_role": doc.get("applied_role") or profile.get("applied_role"),
            "applied_level": doc.get("applied_level") or profile.get("applied_level"),
            "chunk_index": doc.get("chunk_index"),
        })

    return chunks


def delete_all_documents():
    result = get_collection().delete_many({})
    try:
        get_profiles_collection().delete_many({})
    except Exception:
        pass
    return result.deleted_count


def count_documents():
    return get_collection().count_documents({})


def get_distinct_sources():
    try:
        return get_collection().distinct("source")
    except:
        return []


def source_exists(source_name: str) -> bool:
    try:
        return get_collection().count_documents({"source": source_name}, limit=1) > 0
    except:
        return False


def make_unique_source_name(filename: str, file_hash: str) -> str:
    if not source_exists(filename):
        return filename

    stem, ext = os.path.splitext(filename)
    candidate = f"{stem} [{file_hash[:8]}]{ext}"
    if not source_exists(candidate):
        return candidate

    counter = 2
    while True:
        candidate = f"{stem} [{file_hash[:8]}-{counter}]{ext}"
        if not source_exists(candidate):
            return candidate
        counter += 1


def delete_documents_by_source(source_name):
    result = get_collection().delete_many({"source": source_name})
    try:
        get_profiles_collection().delete_many({"source": source_name})
    except Exception:
        pass
    return result.deleted_count


def get_candidate_name(source_name: str) -> str | None:
    try:
        profile = get_candidate_profile(source_name)
        if profile.get("candidate_name"):
            return profile.get("candidate_name")
        doc = get_collection().find_one(
            {"source": source_name, "candidate_name": {"$exists": True}},
            {"candidate_name": 1}
        )
        return doc.get("candidate_name") if doc else None
    except:
        return None
