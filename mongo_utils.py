import os
from datetime import datetime, timezone
import certifi
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

# Connection pooling for better performance
MONGO_CLIENT_OPTS = {
    "maxPoolSize": 5,
    "minPoolSize": 1,
    "maxIdleTimeMS": 45000,
    "retryWrites": False  # Disable for better performance
}

_client = None
_chunks_collection = None
_profiles_collection = None

DB_NAME = "aws_rag_db"
CV_CHUNKS_COLLECTION_NAME = "cv_chunks"
CANDIDATE_PROFILES_COLLECTION_NAME = "candidates_profile"


def get_collection():
    """Return the light CV vector chunk collection."""
    global _client, _chunks_collection
    if _chunks_collection is None:
        _client = MongoClient(os.getenv("MONGO_URI"), tlsCAFile=certifi.where(), **MONGO_CLIENT_OPTS)
        _chunks_collection = _client[DB_NAME][CV_CHUNKS_COLLECTION_NAME]
    return _chunks_collection


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
    for key in ("required_skills", "preferred_skills"):
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
) -> None:
    """Store one global profile record per CV for rule matching."""
    candidate = schema.get("candidate") if isinstance(schema, dict) else {}
    candidate = candidate if isinstance(candidate, dict) else {}
    metadata = schema.get("metadata_filter") if isinstance(schema, dict) else {}
    metadata = metadata if isinstance(metadata, dict) else {}
    years = float(schema.get("total_experience_years") or schema.get("experience_years") or 0)
    education = metadata.get("education_min")
    doc = {
        "source": source_name,
        "cv_id": candidate.get("cv_id") or source_name,
        "candidate_name": candidate.get("name") or "",
        "total_experience_years": years,
        "skills": _profile_skill_names(schema),
        "education": education,
        "cv_schema": schema,
        "updated_at": datetime.now(timezone.utc),
    }
    if file_hash:
        doc["file_hash"] = file_hash
    if original_filename:
        doc["original_filename"] = original_filename
    get_profiles_collection().update_one({"source": source_name}, {"$set": doc}, upsert=True)


def get_candidate_profile(source_name: str) -> dict:
    try:
        return get_profiles_collection().find_one({"source": source_name}, {"_id": 0}) or {}
    except Exception:
        return {}


def insert_chunks(chunks_with_embeddings, source_name, *, file_hash=None, original_filename=None):
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
            "text": chunk,
            "vector_embedding": embedding,
            "source": source_name,
            "chunk_index": i,
            "uploaded_at": datetime.now(timezone.utc),
        }

        if file_hash:
            doc["file_hash"] = file_hash
        if original_filename:
            doc["original_filename"] = original_filename

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
            get_collection().insert_many(docs, ordered=False)  # Faster, no order guarantee needed
        except Exception:
            # Fallback to ordered insert if unordered fails
            get_collection().insert_many(docs, ordered=True)


def search_similar_chunks(query_embedding, k=10, source_filter: str = None):
    num_candidates = min(max(k * 3, 15), 50)  # Scale with k, max 50
    
    vector_search = {
        "index":        "vector_index",
        "path":         "vector_embedding",
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
            "content": {"$ifNull": ["$text", "$content"]},
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
    except Exception as e:
        return []


def search_similar_cv_chunks(query_embedding, k=10, source_filter: str = None):
    """Search CV chunks across all indexed resumes."""
    num_candidates = min(max(k * 10, 50), 300)

    vector_search = {
        "index": "vector_index",
        "path": "vector_embedding",
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
                "content": {"$ifNull": ["$text", "$content"]},
                "source": 1,
                "section": 1,
                "chunk_index": 1,
                "candidate_name": 1,
                "original_filename": 1,
                "text": 1,
                "original_text": 1,
                "embedding_text": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]

    try:
        return list(get_collection().aggregate(pipeline))
    except Exception:
        return []


def get_chunks_by_source_for_matching(source_name: str) -> list[dict]:
    try:
        docs = get_collection().find(
            {"source": source_name},
            {
                "_id": 0,
                "content": 1,
                "text": 1,
                "vector_embedding": 1,
                "embedding": 1,
                "source": 1,
                "chunk_index": 1,
                "section": 1,
                "original_text": 1,
                "embedding_text": 1,
                "headings": 1,
                "candidate_name": 1,
            },
        ).sort("chunk_index", 1)
    except Exception:
        return []

    chunks = []
    for doc in docs:
        original_text = doc.get("text") or doc.get("original_text") or doc.get("content") or ""
        embedding_text = doc.get("embedding_text") or doc.get("content") or original_text

        chunk = {
            "source": doc.get("source"),
            "section": doc.get("section", "unknown"),
            "text": original_text,
            "embedding_text": embedding_text,
            "embedding": doc.get("vector_embedding") or doc.get("embedding"),
            "chunk_index": doc.get("chunk_index"),
        }

        for key in ("headings", "candidate_name"):
            if key in doc:
                chunk[key] = doc[key]

        if not chunks:
            profile = get_candidate_profile(source_name)
            if isinstance(profile.get("cv_schema"), dict):
                chunk["cv_schema"] = profile["cv_schema"]
            if profile.get("candidate_name") and not chunk.get("candidate_name"):
                chunk["candidate_name"] = profile["candidate_name"]
        chunks.append(chunk)

    return chunks


def get_all_cv_chunks_for_matching() -> list[dict]:
    try:
        docs = get_collection().find(
            {},
            {
                "_id": 0,
                "content": 1,
                "source": 1,
                "chunk_index": 1,
                "section": 1,
                "text": 1,
                "original_text": 1,
                "embedding_text": 1,
                "candidate_name": 1,
            },
        ).sort([("source", 1), ("chunk_index", 1)])
    except Exception:
        return []

    chunks = []
    profiles: dict[str, dict] = {}
    for doc in docs:
        source = doc.get("source")
        chunk = {
            "source": doc.get("source"),
            "section": doc.get("section", "unknown"),
            "text": doc.get("text") or doc.get("original_text") or doc.get("content") or "",
            "embedding_text": doc.get("embedding_text") or doc.get("content") or "",
            "candidate_name": doc.get("candidate_name"),
            "chunk_index": doc.get("chunk_index"),
        }
        if source not in profiles:
            profiles[source] = get_candidate_profile(source)
        profile = profiles.get(source) or {}
        if doc.get("chunk_index") == 0 and isinstance(profile.get("cv_schema"), dict):
            chunk["cv_schema"] = profile["cv_schema"]
        if profile.get("candidate_name") and not chunk.get("candidate_name"):
            chunk["candidate_name"] = profile["candidate_name"]
        chunks.append(chunk)

    return chunks


def delete_all_documents():
    chunk_result = get_collection().delete_many({})
    get_profiles_collection().delete_many({})
    return chunk_result.deleted_count


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
    get_profiles_collection().delete_many({"source": source_name})
    return result.deleted_count


def get_candidate_name(source_name: str) -> str | None:
    try:
        doc = get_profiles_collection().find_one({"source": source_name}, {"candidate_name": 1})
        if not doc:
            doc = get_collection().find_one(
                {"source": source_name, "candidate_name": {"$exists": True}},
                {"candidate_name": 1}
            )
        return doc.get("candidate_name") if doc else None
    except:
        return None
