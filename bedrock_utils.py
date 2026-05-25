import json
import math
import os
import random
import re
import time

import boto3
from botocore.exceptions import ClientError, ProfileNotFound
from dotenv import load_dotenv


load_dotenv()

BEDROCK_EMBED_MODEL = os.getenv("BEDROCK_EMBED_MODEL", "amazon.titan-embed-text-v2:0")
GOOGLE_EMBED_MODEL = os.getenv("GOOGLE_EMBED_MODEL", "models/embedding-001")
LOCAL_EMBED_MODEL = os.getenv(
    "LOCAL_EMBED_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)
LOCAL_EMBED_BACKEND = os.getenv("LOCAL_EMBED_BACKEND", "sentence_transformer").strip().lower()
LOCAL_EMBED_DIM = int(os.getenv("LOCAL_EMBED_DIM", "384"))
LOCAL_EMBED_LOCAL_FILES_ONLY = os.getenv("LOCAL_EMBED_LOCAL_FILES_ONLY", "true").strip().lower() in {
    "1",
    "true",
    "yes",
}
LOCAL_EMBED_HASH_FALLBACK = os.getenv("LOCAL_EMBED_HASH_FALLBACK", "true").strip().lower() in {
    "1",
    "true",
    "yes",
}
GOOGLE_EMBED_FALLBACK_MODELS = [
    model.strip()
    for model in os.getenv(
        "GOOGLE_EMBED_FALLBACK_MODELS",
        "models/embedding-001,models/text-embedding-004",
    ).split(",")
    if model.strip()
]
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local").strip().lower()
DEFAULT_AWS_REGION = os.getenv("AWS_REGION")
MAX_EMBED_RETRIES = 6
BASE_RETRY_DELAY_SECONDS = 1.0

_bedrock_client = None
_google_embeddings_by_model = {}
_local_embedding_model = None


def get_local_embedding_model():
    global _local_embedding_model
    if _local_embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _local_embedding_model = SentenceTransformer(
            LOCAL_EMBED_MODEL,
            local_files_only=LOCAL_EMBED_LOCAL_FILES_ONLY,
        )
    return _local_embedding_model


def get_hash_embedding(text: str, dim: int = LOCAL_EMBED_DIM) -> list[float]:
    tokens = re.findall(r"[\w+#.-]+", str(text or "").lower(), flags=re.UNICODE)
    vector = [0.0] * dim
    if not tokens:
        return vector

    for token in tokens:
        digest = int.from_bytes(token.encode("utf-8"), "little", signed=False)
        index = digest % dim
        sign = 1.0 if ((digest // dim) % 2 == 0) else -1.0
        vector[index] += sign

    norm = math.sqrt(sum(value * value for value in vector))
    if norm > 0:
        vector = [value / norm for value in vector]
    return vector


def _google_api_key() -> str:
    return (
        os.getenv("GOOGLE_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_GENAI_API_KEY")
        or ""
    ).strip()


def get_google_embeddings(model: str | None = None):
    model = model or GOOGLE_EMBED_MODEL
    if model in _google_embeddings_by_model:
        return _google_embeddings_by_model[model]

    api_key = _google_api_key()
    if not api_key:
        raise RuntimeError(
            "Missing Google embedding API key. Add GOOGLE_API_KEY=... to .env "
            "or set GEMINI_API_KEY/GOOGLE_GENAI_API_KEY."
        )

    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    embeddings = GoogleGenerativeAIEmbeddings(
        model=model,
        google_api_key=api_key,
    )
    _google_embeddings_by_model[model] = embeddings
    return embeddings


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


def get_bedrock_embedding(text: str) -> list[float]:
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
        except ClientError as exc:
            error_code = exc.response.get("Error", {}).get("Code", "")
            if error_code != "ThrottlingException" or attempt == MAX_EMBED_RETRIES:
                raise

            delay = BASE_RETRY_DELAY_SECONDS * (2 ** (attempt - 1))
            time.sleep(delay + random.uniform(0, 0.5))

    return []


def get_embedding(text: str) -> list[float]:
    """
    Create an embedding vector for CV/JD vector search.

    Default provider is local SentenceTransformer. Set EMBEDDING_PROVIDER to
    google or bedrock in .env to switch providers.
    """
    if not isinstance(text, str):
        text = str(text)

    if EMBEDDING_PROVIDER == "local":
        if LOCAL_EMBED_BACKEND in {"hash", "hashed", "hashing"}:
            return get_hash_embedding(text)

        try:
            vector = get_local_embedding_model().encode(
                text,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return vector.astype(float).tolist()
        except Exception:
            if LOCAL_EMBED_HASH_FALLBACK:
                return get_hash_embedding(text)
            raise

    if EMBEDDING_PROVIDER == "bedrock":
        return get_bedrock_embedding(text)

    models_to_try = []
    for model in [GOOGLE_EMBED_MODEL, *GOOGLE_EMBED_FALLBACK_MODELS]:
        if model not in models_to_try:
            models_to_try.append(model)

    last_error = None
    for model in models_to_try:
        try:
            return list(get_google_embeddings(model).embed_query(text))
        except Exception as exc:
            last_error = exc

    if last_error:
        raise last_error
    return []
