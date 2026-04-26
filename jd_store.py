import certifi
import os

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
VECTOR_INDEX_NAME = "jd_vector_index"

SAMPLE_JDS = [
    {
        "id": "jd_001",
        "title": "Tester Intern / QA Intern",
        "content": """
        Vi tri: Tester Intern / QA Intern
        Yeu cau ky nang: Manual testing, Test case design,
        Bug reporting, Jira, Postman, API testing
        Kinh nghiem: Chua can kinh nghiem, uu tien co du an thuc te
        Ky nang mem: Ti mi, can than, tieng Anh co ban
        """,
    },
    {
        "id": "jd_002",
        "title": "Backend Developer Intern",
        "content": """
        Vi tri: Backend Developer Intern
        Yeu cau: Python hoac Node.js, REST API, SQL
        Kinh nghiem: Co project ca nhan la loi the
        Ky nang mem: Teamwork, giao tiep tot
        """,
    },
    {
        "id": "jd_003",
        "title": "Frontend Developer Intern",
        "content": """
        Vi tri: Frontend Developer Intern
        Yeu cau: HTML, CSS, JavaScript, React
        Kinh nghiem: Chua can kinh nghiem, uu tien co du an thuc te
        Ky nang mem: Teamwork, giao tiep tot
        """,
    },
    {
        "id": "jd_004",
        "title": "Data Analyst Intern",
        "content": """
        Vi tri: Data Analyst Intern
        Yeu cau: Excel, SQL, Python (Pandas), Data visualization
        Kinh nghiem: Chua can kinh nghiem, uu tien co du an thuc te
        Ky nang mem: Tinh toan, chinh xac, giao tiep tot
        """,
    },
]

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
    client = get_mongo_client()
    return client["aws_rag_db"][JD_COLLECTION_NAME]


def count_indexed_jds() -> int:
    try:
        return get_jd_store().count_documents({})
    except Exception:
        return 0


def ingest_jds() -> int:
    collection = get_jd_store()
    operations = []

    for jd in SAMPLE_JDS:
        content = jd["content"].strip()
        embedding = get_embedding(content)
        operations.append(
            UpdateOne(
                {"jd_id": jd["id"]},
                {
                    "$set": {
                        "jd_id": jd["id"],
                        "title": jd["title"],
                        "content": content,
                        "embedding": embedding,
                    }
                },
                upsert=True,
            )
        )

    if operations:
        collection.bulk_write(operations, ordered=False)

    collection.delete_many({"jd_id": {"$nin": [jd["id"] for jd in SAMPLE_JDS]}})
    return len(operations)


def search_similar_jds(query_embedding, k: int = 3):
    collection = get_jd_store()
    pipeline = [
        {
            "$vectorSearch": {
                "index": VECTOR_INDEX_NAME,
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": min(max(k * 5, 20), 100),
                "limit": k,
            }
        },
        {
            "$project": {
                "_id": 0,
                "jd_id": 1,
                "title": 1,
                "content": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]
    return list(collection.aggregate(pipeline))
