import os
import boto3
import json
from dotenv import load_dotenv

load_dotenv()

session = boto3.Session(
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION"),
)

bedrock = session.client("bedrock-runtime")


def get_embedding(text: str) -> list[float]:
    """
    Tạo embedding vector cho text bằng Amazon Titan Embed v2.
    Trả về list float (1536 chiều).
    """
    if not isinstance(text, str):
        text = str(text)

    response = bedrock.invoke_model(
        modelId="amazon.titan-embed-text-v2:0",
        contentType="application/json",
        accept="application/json",
        body=json.dumps({"inputText": text}),
    )
    result = json.loads(response["body"].read())
    return result.get("embedding", [])