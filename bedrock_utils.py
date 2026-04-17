# Bedrock Utils: chứa hàm get_embedding để gọi API của Bedrock, trả về embedding vector cho text đầu vào. Hàm này được sử dụng trong rag_core.py để tạo embedding cho các chunk trước khi lưu vào MongoDB.
import os
import boto3
import json
from dotenv import load_dotenv #Chức năng: load biến môi trường từ file .env, giúp bảo mật thông tin nhạy cảm như AWS keys.
from pdf_utils import cv_pdf_to_semantic_chunks #Chức năng: trích xuất nội dung từ file PDF và chia thành các chunk có ngữ cảnh rõ ràng (vd: section, text) để dễ dàng tạo embedding và lưu vào MongoDB.

load_dotenv() #Chức năng: tải biến môi trường từ file .env để sử dụng trong việc cấu hình AWS keys cho việc gọi API của Bedrock.

session = boto3.Session(
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION")
)

bedrock = session.client("bedrock-runtime")
def get_embedding(text):
    # Ensure string
    if not isinstance(text, str):
        text = str(text)

    payload = {
        "inputText": text
    }

    response = bedrock.invoke_model(
        modelId="amazon.titan-embed-text-v2:0",
        body=json.dumps(payload),
        contentType="application/json",
        accept="application/json"
    )

    result = json.loads(response["body"].read())

    return result.get("embedding", [])