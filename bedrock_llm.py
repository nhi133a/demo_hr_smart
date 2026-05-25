import json
import os
from dotenv import load_dotenv
from typing import List

from bedrock_utils import get_bedrock_client

load_dotenv()

BEDROCK_CHAT_MODEL = os.getenv("BEDROCK_CHAT_MODEL", "anthropic.claude-3-haiku-20240307-v1:0")
BEDROCK_MAX_TOKENS = int(os.getenv("BEDROCK_MAX_TOKENS", "3000"))


def generate_answer(prompt: str) -> str:
    response = get_bedrock_client().invoke_model(
        modelId=BEDROCK_CHAT_MODEL,
        contentType="application/json",
        accept="application/json",
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": BEDROCK_MAX_TOKENS,
            "temperature": 0,
            "messages": [
                {"role": "user", "content": prompt}
            ],
        }),
    )
    result = json.loads(response["body"].read())
    return result["content"][0]["text"]
