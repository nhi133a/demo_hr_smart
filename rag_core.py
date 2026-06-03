from pathlib import Path
import hashlib
import logging

from bedrock_utils import get_embedding
from llm_provider import generate_answer
from mongo_utils import insert_chunks, make_unique_source_name, search_similar_chunks, upsert_candidate_profile
from pdf_utils import (
    convert_cv_pdf,
    cv_document_to_chunks,
    cv_markdown_to_chunks,
    extract_cv_schema,
)



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _normalize_chunk(chunk) -> tuple[str, str]:
    if isinstance(chunk, dict):
        section = chunk.get("section", "unknown")
        text = chunk.get("text") or chunk.get("content") or str(chunk)
    else:
        section = "unknown"
        text = str(chunk)
    return section, text


def _embed_chunks(chunks: list, filename: str) -> list[tuple[str, list, dict]]:
    embedded = []
    total = len(chunks)

    for i, chunk in enumerate(chunks, 1):
        section, text = _normalize_chunk(chunk)
        if not text.strip():
            continue

        log.info("  [%d/%d] section=%-12s | preview: %.60s", i, total, section, text.replace("\n", " "))

        try:
            embedding = get_embedding(text)
        except Exception as exc:
            log.warning("  [%d/%d] Embed failed, skipping chunk: %s", i, total, exc)
            continue

        log.info("  [%d/%d] embedding dim=%d", i, total, len(embedding))

        metadata = {"section": section}
        if isinstance(chunk, dict):
            for key in ("candidate_name", "headings"):
                if key in chunk:
                    metadata[key] = chunk[key]

        embedded.append((text, embedding, metadata))

    return embedded


def index_cv(
    cv_id: str,
    schema: dict,
    *,
    file_hash: str = None,
    original_filename: str = None,
    chunks: list,
    storage_metadata: dict | None = None,
) -> int:
    log.info("Raw vector chunks for '%s': %d", cv_id, len(chunks))

    upsert_candidate_profile(
        cv_id,
        schema,
        file_hash=file_hash,
        original_filename=original_filename,
        extra_metadata=storage_metadata,
    )
    embedded = _embed_chunks(chunks, original_filename or cv_id)
    insert_chunks(
        embedded,
        source_name=cv_id,
        file_hash=file_hash,
        original_filename=original_filename,
        extra_metadata=storage_metadata,
    )
    return len(embedded)


def process_cv(file_path: str) -> dict:
    path = Path(file_path)
    file_bytes = path.read_bytes()
    file_hash = hashlib.sha256(file_bytes).hexdigest()
    cv_id = make_unique_source_name(path.name, file_hash)

    doc, markdown_text = convert_cv_pdf(path)

    chunks = (
        cv_document_to_chunks(doc, markdown_text=markdown_text)
        if doc is not None
        else cv_markdown_to_chunks(markdown_text, {})
    )
    schema = extract_cv_schema(markdown_text, chunks=chunks)
    # reconcile_cv_schema_with_chunks removed because function no longer exists in pdf_utils.py
    # CV schema is already normalized via extract_cv_schema().
    schema.setdefault("candidate", {})["cv_id"] = cv_id


    index_cv(cv_id, schema, file_hash=file_hash, original_filename=path.name, chunks=chunks)
    return schema


def process_cv_bytes(pdf_bytes: bytes, filename: str, *, storage_metadata: dict | None = None) -> tuple[dict, int, str]:
    file_hash = hashlib.sha256(pdf_bytes).hexdigest()
    cv_id = make_unique_source_name(filename, file_hash)

    doc, markdown_text = convert_cv_pdf(pdf_bytes)

    chunks = (
        cv_document_to_chunks(doc, markdown_text=markdown_text)
        if doc is not None
        else cv_markdown_to_chunks(markdown_text, {})
    )
    schema = extract_cv_schema(markdown_text, chunks=chunks)
    # reconcile_cv_schema_with_chunks removed because function no longer exists in pdf_utils.py
    schema.setdefault("candidate", {})["cv_id"] = cv_id

    count = index_cv(
        cv_id,
        schema,
        file_hash=file_hash,
        original_filename=filename,
        chunks=chunks,
        storage_metadata=storage_metadata,
    )
    return schema, count, cv_id


def process_pdf_and_store(pdf_bytes: bytes, filename: str) -> int:
    _, count, _ = process_cv_bytes(pdf_bytes, filename)
    return count


def process_multiple_pdfs(
    file_list,
    *,
    store_originals_to_s3: bool = False,
    extra_metadata: dict | None = None,
) -> tuple[int, list[str]]:
    if store_originals_to_s3:
        from s3_utils import upload_cv_pdf_to_s3

    log.info("Processing %d files", len(file_list))
    uploaded_sources: list[str] = []
    total = 0

    for idx, file in enumerate(file_list, 1):
        log.info("File %d/%d: %s", idx, len(file_list), file.name)
        pdf_bytes = file.read()
        file_hash = hashlib.sha256(pdf_bytes).hexdigest()
        storage_metadata = None
        if store_originals_to_s3:
            storage_metadata = upload_cv_pdf_to_s3(pdf_bytes, file.name, file_hash=file_hash)
        if isinstance(extra_metadata, dict):
            storage_metadata = {**(storage_metadata or {}), **extra_metadata}
        _, count, source_name = process_cv_bytes(
            pdf_bytes,
            file.name,
            storage_metadata=storage_metadata,
        )
        total += count
        uploaded_sources.append(source_name)

    log.info("Completed: %d chunks", total)
    return total, uploaded_sources


def answer_question(question: str, k: int = 3, source_filter: str = None) -> str:
    log.info("Question: %s", question)
    log.info("k=%d | source_filter=%s", k, source_filter or "all CVs")

    query_embedding = get_embedding(question)
    log.info("embedding dim=%d", len(query_embedding))

    results = search_similar_chunks(query_embedding, k=k, source_filter=source_filter)
    log.info("MongoDB returned %d chunks", len(results))

    if not results:
        log.warning("No relevant chunk found")
        return "No relevant content found. Try rephrasing your question."

    for i, doc in enumerate(results, 1):
        log.info("chunk %d: %s", i, str(doc["content"]).replace("\n", " ")[:80])

    context = "\n\n".join(str(doc["content"]) for doc in results)
    prompt = f"""Here is the information extracted from the CV:

{context}

Based on the above, please answer concisely:
{question}"""

    log.info("Calling local LLM...")
    answer = generate_answer(prompt)
    log.info("Answer: %s", answer[:120].replace("\n", " "))
    return answer
