import json
import boto3 #Chức nang chính: giao tiếp với AWS services, ở đây là Bedrock.
import os #Chức năng: tương tác với hệ điều hành, ở đây dùng để lấy biến môi trường chứa AWS keys.
from dotenv import load_dotenv #Chức năng: load biến môi trường từ file .env, giúp bảo mật thông tin nhạy cảm như AWS keys.

load_dotenv() 

session = boto3.Session( 
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION")
)

bedrock = session.client("bedrock-runtime")

def generate_answer(prompt): #Chức năng: gửi prompt đến Bedrock để nhận câu trả lời từ LLM.
    body = {
        "inputText": prompt,
        "textGenerationConfig": {
            "maxTokenCount": 512,
            "temperature": 0.3,
            "topP": 0.9
        }
    }

    response = bedrock.invoke_model( #Chức năng: gọi API của Bedrock để tạo câu trả lời dựa trên prompt đã cho.
    modelId="anthropic.claude-3-haiku-20240307-v1:0",
    body=json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }),
    contentType="application/json"
)

    result = json.loads(response["body"].read()) #Chức năng: đọc và giải mã phản hồi JSON từ API của Bedrock để lấy câu trả lời.    
    return result["content"][0]["text"] #Chức năng: trả về phần text của câu trả lời từ phản hồi của Bedrock.