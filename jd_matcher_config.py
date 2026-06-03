import os


MATCH_MODE = os.getenv("MATCH_MODE", "balanced").strip().lower()
if MATCH_MODE not in {"fast", "balanced", "deep"}:
    MATCH_MODE = "balanced"

MATCH_CANDIDATE_POOL_LIMIT = max(1, int(os.getenv("MATCH_CANDIDATE_POOL_LIMIT", "50") or 50))
MATCH_RETRIEVAL_MULTIPLIER = max(1, int(os.getenv("MATCH_RETRIEVAL_MULTIPLIER", "3") or 3))
MATCH_RETRIEVAL_MIN_CANDIDATES = max(1, int(os.getenv("MATCH_RETRIEVAL_MIN_CANDIDATES", "10") or 10))
MATCH_DEEP_RERANK_LIMIT = max(1, int(os.getenv("MATCH_DEEP_RERANK_LIMIT", "30") or 30))
MATCH_DEEP_RERANK_MULTIPLIER = max(1, int(os.getenv("MATCH_DEEP_RERANK_MULTIPLIER", "3") or 3))

CROSS_ENCODER_ENABLED = os.getenv("CROSS_ENCODER_ENABLED", "true").strip().lower() in {
    "1",
    "true",
    "yes",
}
CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
CROSS_ENCODER_LOCAL_FILES_ONLY = os.getenv("CROSS_ENCODER_LOCAL_FILES_ONLY", "true").strip().lower() in {
    "1",
    "true",
    "yes",
}
CROSS_ENCODER_MAX_CHUNKS = max(1, int(os.getenv("CROSS_ENCODER_MAX_CHUNKS", "6") or 6))
CROSS_ENCODER_TOP_AVG = max(1, int(os.getenv("CROSS_ENCODER_TOP_AVG", "3") or 3))
CROSS_ENCODER_MAX_JD_CHARS = max(500, int(os.getenv("CROSS_ENCODER_MAX_JD_CHARS", "3500") or 3500))
CROSS_ENCODER_MAX_CHUNK_CHARS = max(300, int(os.getenv("CROSS_ENCODER_MAX_CHUNK_CHARS", "1200") or 1200))
CROSS_ENCODER_WEIGHT = max(0.0, min(1.0, float(os.getenv("CROSS_ENCODER_WEIGHT", "0.25") or 0.25)))

REQUIREMENT_EVIDENCE_ENABLED = os.getenv("REQUIREMENT_EVIDENCE_ENABLED", "true").strip().lower() in {
    "1",
    "true",
    "yes",
}
REQUIREMENT_EVIDENCE_THRESHOLD = max(
    0.0,
    min(1.0, float(os.getenv("REQUIREMENT_EVIDENCE_THRESHOLD", "0.55") or 0.55)),
)
REQUIREMENT_EVIDENCE_MAX_REQUIREMENTS = max(1, int(os.getenv("REQUIREMENT_EVIDENCE_MAX_REQUIREMENTS", "18") or 18))
REQUIREMENT_EVIDENCE_MAX_CHUNKS = max(1, int(os.getenv("REQUIREMENT_EVIDENCE_MAX_CHUNKS", "4") or 4))

LLM_REQUIREMENT_VERIFIER_ENABLED = os.getenv("LLM_REQUIREMENT_VERIFIER_ENABLED", "false").strip().lower() in {
    "1",
    "true",
    "yes",
}
LLM_REQUIREMENT_VERIFIER_THRESHOLD = max(
    0.0,
    min(1.0, float(os.getenv("LLM_REQUIREMENT_VERIFIER_THRESHOLD", "0.70") or 0.70)),
)
LLM_REQUIREMENT_VERIFIER_MAX_REQUIREMENTS = max(
    1,
    int(os.getenv("LLM_REQUIREMENT_VERIFIER_MAX_REQUIREMENTS", "10") or 10),
)
LLM_REQUIREMENT_VERIFIER_MAX_CHARS = max(
    500,
    int(os.getenv("LLM_REQUIREMENT_VERIFIER_MAX_CHARS", "2600") or 2600),
)
LLM_REQUIREMENT_VERIFIER_TIMEOUT_SECONDS = max(
    3,
    int(os.getenv("LLM_REQUIREMENT_VERIFIER_TIMEOUT_SECONDS", "12") or 12),
)
LLM_REQUIREMENT_VERIFIER_MIN_CANDIDATE_SCORE = max(
    0.0,
    float(os.getenv("LLM_REQUIREMENT_VERIFIER_MIN_CANDIDATE_SCORE", "1.0") or 1.0),
)
