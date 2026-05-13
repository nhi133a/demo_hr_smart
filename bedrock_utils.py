import json
import os
import random
import time

import boto3
from botocore.exceptions import ClientError, ProfileNotFound
from dotenv import load_dotenv

load_dotenv()

BEDROCK_EMBED_MODEL = "amazon.titan-embed-text-v2:0"
DEFAULT_AWS_REGION = os.getenv("AWS_REGION")
MAX_EMBED_RETRIES = 6
BASE_RETRY_DELAY_SECONDS = 1.0

_bedrock_client = None


def _build_session_kwargs(include_profile: bool = True) -> dict:
    region = os.getenv("AWS_REGION") or DEFAULT_AWS_REGION
    access_key = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    session_token = os.getenv("AWS_SESSION_TOKEN")
    profile_name = os.getenv("AWS_PROFILE") or os.getenv("AWS_DEFAULT_PROFILE")

    kwargs = {"region_name": region}

    # Explicit credentials should take priority over local profile configuration.
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


def get_embedding(text: str) -> list[float]:
    """
    Tạo embedding vector cho text bằng Amazon Titan Embed v2.
    Trả về list float (1536 chiều).
    """
    if not isinstance(text, str):
        text = str(text)

    for attempt in range(1, MAX_EMBED_RETRIES + 1):
        try:
            response = get_bedrock_client().invoke_model(
                modelId=BEDROCK_EMBED_MODEL,
                contentType="application/json",
                accept="application/json",
                body=json.dumps({"inputText": text}),
            )
            result = json.loads(response["body"].read())
            return result.get("embedding", [])
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code != "ThrottlingException" or attempt == MAX_EMBED_RETRIES:
                raise

            delay = BASE_RETRY_DELAY_SECONDS * (2 ** (attempt - 1))
            # Add a little jitter so repeated retries from the same batch do not align.
            time.sleep(delay + random.uniform(0, 0.5))

    return []
