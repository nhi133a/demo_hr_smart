# debug.py
import os
import certifi
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

# ── Bước 1: Kiểm tra kết nối MongoDB ─────────────────
print("=== BƯỚC 1: Kết nối MongoDB ===")
try:
    client = MongoClient(os.getenv("MONGO_URI"), tlsCAFile=certifi.where())
    client.admin.command("ping")
    print("✅ Kết nối MongoDB thành công")
except Exception as e:
    print(f"❌ Lỗi kết nối: {e}")
    exit()

# ── Bước 2: Kiểm tra data trong collection ────────────
print("\n=== BƯỚC 2: Kiểm tra data ===")
db = client["aws_rag_db"]
collection = db["documents"]

count = collection.count_documents({})
print(f"Tổng số chunks: {count}")

if count == 0:
    print("❌ Collection rỗng — CV chưa được index vào MongoDB")
    exit()

# Xem 1 document mẫu
sample = collection.find_one({})
print(f"Sample document keys: {list(sample.keys())}")
print(f"Content preview: {sample.get('content', '')[:100]}")

embedding = sample.get("embedding")
if embedding is None:
    print("❌ Embedding là NULL — Bedrock chưa tạo được embedding")
elif isinstance(embedding, list):
    print(f"✅ Embedding tồn tại, số chiều: {len(embedding)}")
else:
    print(f"❌ Embedding sai format: {type(embedding)}")

# ── Bước 3: Kiểm tra sources ─────────────────────────
print("\n=== BƯỚC 3: Danh sách CV đã index ===")
sources = collection.distinct("source")
print(f"Sources: {sources}")

# ── Bước 4: Test vector search ───────────────────────
print("\n=== BƯỚC 4: Test Vector Search ===")
if embedding and isinstance(embedding, list):
    try:
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "embedding",
                    "queryVector": embedding,  # dùng chính embedding đã lưu làm query
                    "numCandidates": 10,
                    "limit": 3
                }
            }
        ]
        results = list(collection.aggregate(pipeline))
        print(f"✅ Vector search trả về {len(results)} kết quả")
        if results:
            print(f"Preview: {results[0].get('content', '')[:100]}")
    except Exception as e:
        print(f"❌ Vector search lỗi: {e}")