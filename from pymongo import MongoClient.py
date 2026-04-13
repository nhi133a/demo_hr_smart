from pymongo import MongoClient

uri = "mongodb+srv://nhi133a_db_user:IDm7iCmvuBwURphs@cluster0.f0cdnae.mongodb.net/?appName=Cluster0"

try:
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    client.server_info()
    print("✅ Kết nối MongoDB Atlas thành công")
except Exception as e:
    print("❌ Kết nối MongoDB Atlas thất bại:", e)