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

client     = MongoClient(os.getenv("MONGO_URI"), tlsCAFile=certifi.where(), **MONGO_CLIENT_OPTS)
db         = client["aws_rag_db"]
collection = db["documents"]


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
            "content": chunk,
            "embedding": embedding,
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
            collection.insert_many(docs, ordered=False)  # Faster, no order guarantee needed
        except Exception:
            # Fallback to ordered insert if unordered fails
            collection.insert_many(docs, ordered=True)


def search_similar_chunks(query_embedding, k=10, source_filter: str = None):
    """
    Optimized vector search - reduced numCandidates for speed
    """
    # Optimize: reduce numCandidates based on k value
    num_candidates = min(max(k * 3, 15), 50)  # Scale with k, max 50
    
    vector_search = {
        "index":        "vector_index",
        "path":         "embedding",
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
            "content": 1,
            "source": 1,
            "section": 1,
            "candidate_name": 1,
            "_id": 0,
            "score": {"$meta": "vectorSearchScore"}
        }
    })
    
    try:
        return list(collection.aggregate(pipeline))
    except Exception as e:
        return []


def get_chunks_by_source_for_matching(source_name: str) -> list[dict]:
    """
    Return the original enriched CV chunks for JD matching.

    New uploads include original_text, embedding_text, skill_text and extracted_info.
    Older indexed CVs still work through the content/original_text fallbacks.
    """
    try:
        docs = collection.find(
            {"source": source_name},
            {
                "_id": 0,
                "content": 1,
                "embedding": 1,
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
                "chunk_experience_months": 1,
                "chunk_experience_years": 1,
                "chunk_experience_duration": 1,
            },
        ).sort("chunk_index", 1)
    except Exception:
        return []

    chunks = []
    for doc in docs:
        original_text = doc.get("original_text") or doc.get("content") or ""
        embedding_text = doc.get("embedding_text") or doc.get("content") or original_text

        chunk = {
            "section": doc.get("section", "unknown"),
            "text": original_text,
            "embedding_text": embedding_text,
            "embedding": doc.get("embedding"),
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
        ):
            if key in doc:
                chunk[key] = doc[key]

        chunks.append(chunk)

    return chunks


def delete_all_documents():
    result = collection.delete_many({})
    return result.deleted_count


def count_documents():
    return collection.count_documents({})


def get_distinct_sources():
    """Get all CV sources with projection for speed"""
    try:
        return collection.distinct("source")
    except:
        return []


def source_exists(source_name: str) -> bool:
    try:
        return collection.count_documents({"source": source_name}, limit=1) > 0
    except:
        return False


def make_unique_source_name(filename: str, file_hash: str) -> str:
    """
    Build a stable but unique source name so every uploaded CV remains selectable.
    First upload keeps the original filename; later uploads with the same name get
    a short hash suffix instead of overwriting or mixing with old chunks.
    """
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
    result = collection.delete_many({"source": source_name})
    return result.deleted_count


@staticmethod
def get_candidate_name(source_name: str) -> str | None:
    """
    Get candidate name directly from DB - optimized single query
    """
    try:
        doc = collection.find_one(
            {"source": source_name, "candidate_name": {"$exists": True}},
            {"candidate_name": 1}
        )
        return doc.get("candidate_name") if doc else None
    except:
        return None
