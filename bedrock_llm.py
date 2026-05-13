import json
import boto3
import os
from dotenv import load_dotenv
from typing import List

load_dotenv()

session = boto3.Session(
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION"),
)

bedrock = session.client("bedrock-runtime")


def generate_answer(prompt: str) -> str:
    response = bedrock.invoke_model(
        modelId="anthropic.claude-3-haiku-20240307-v1:0",
        contentType="application/json",
        accept="application/json",
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "temperature": 0.3,
            "top_p": 0.9,
            "messages": [
                {"role": "user", "content": prompt}
            ],
        }),
    )
    result = json.loads(response["body"].read())
    return result["content"][0]["text"]


def extract_skills_from_experience(experience_text: str) -> List[str]:
    if not experience_text or len(experience_text.strip()) < 20:
        return []

    prompt = f"""From this work experience: "{experience_text}"

Extract ONLY technical skills, tools, technologies, languages, frameworks, databases, and methodologies mentioned.
Format: one skill per line, no numbering, no explanation.
Example output:
Python
Flask
PostgreSQL
Docker
REST APIs

Extract skills:"""

    try:
        response = generate_answer(prompt)
        skills = [s.strip() for s in response.split('\n') if s.strip() and len(s.strip()) < 100]
        return skills[:15]
    except Exception as e:
        print(f"[WARNING] Failed to extract skills: {e}")
        return []