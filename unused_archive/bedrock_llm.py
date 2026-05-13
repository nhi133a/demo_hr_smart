import json
import os
from typing import List

import boto3
from botocore.exceptions import ProfileNotFound
from dotenv import load_dotenv

load_dotenv()

BEDROCK_LLM_MODEL = "anthropic.claude-3-haiku-20240307-v1:0"
DEFAULT_AWS_REGION = "us-east-1"

_bedrock_client = None


def _build_session_kwargs(include_profile: bool = True) -> dict:
    region = os.getenv("AWS_REGION") or DEFAULT_AWS_REGION
    access_key = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    session_token = os.getenv("AWS_SESSION_TOKEN")
    profile_name = os.getenv("AWS_PROFILE") or os.getenv("AWS_DEFAULT_PROFILE")

    kwargs = {"region_name": region}

    if access_key and secret_key:
        kwargs["aws_access_key_id"] = access_key
        kwargs["aws_secret_access_key"] = secret_key
        if session_token:
            kwargs["aws_session_token"] = session_token
        return kwargs

    if include_profile and profile_name:
        kwargs["profile_name"] = profile_name

    return kwargs


def get_bedrock_client():
    global _bedrock_client
    if _bedrock_client is not None:
        return _bedrock_client

    try:
        session = boto3.Session(**_build_session_kwargs(include_profile=True))
    except ProfileNotFound:
        session = boto3.Session(**_build_session_kwargs(include_profile=False))

    _bedrock_client = session.client("bedrock-runtime")
    return _bedrock_client


def generate_answer(prompt: str) -> str:
    response = get_bedrock_client().invoke_model(
        modelId=BEDROCK_LLM_MODEL,
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
        skills = [s.strip() for s in response.split("\n") if s.strip() and len(s.strip()) < 100]
        return skills[:15]
    except Exception as e:
        print(f"[WARNING] Failed to extract skills: {e}")
        return []
