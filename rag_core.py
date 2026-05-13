from pdf_utils import cv_pdf_to_semantic_chunks
from bedrock_utils import get_embedding
from local_llm import generate_answer
from mongo_utils import insert_chunks, make_unique_source_name, search_similar_chunks
import hashlib
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# =============================================================================
# HELPERS
# =============================================================================

def _normalize_chunk(chunk) -> tuple[str, str]:
    if isinstance(chunk, dict):
        section = chunk.get("section", "unknown")   # fix: đồng bộ với pdf_utils
        text    = chunk.get("embedding_text") or chunk.get("text") or chunk.get("content") or str(chunk)
    else:
        section = "unknown"                          # fix: đồng bộ với pdf_utils
        text    = str(chunk)
    return section, text


def _embed_chunks(chunks: list, filename: str) -> list[tuple[str, list]]:
    embedded = []                                    # fix: khởi tạo list trước khi dùng
    total = len(chunks)

    for i, chunk in enumerate(chunks, 1):
        section, text = _normalize_chunk(chunk)
        if not text.strip():
            continue

        log.info("  [%d/%d] section=%-12s | preview: %.60s", i, total, section, text.replace("\n", " "))

        try:
            embedding = get_embedding(text)
        except Exception as e:
            log.warning("  [%d/%d] ⚠️  Embed thất bại, bỏ qua chunk: %s", i, total, e)
            continue

        log.info("  [%d/%d] embedding dim=%d", i, total, len(embedding))

        metadata = {"section": section}
        if isinstance(chunk, dict):
            for key in (
                "candidate_name",
                "extracted_info",
                "skill_text",
                "skip_embed",
                "headings",
                "llm_skills",
                "raw_llm_section",
                "llm_section",
                "cv_experience",
                "chunk_experience_months",
                "chunk_experience_years",
                "chunk_experience_duration",
            ):
                if key in chunk:
                    metadata[key] = chunk[key]

            # Keep both the original chunk text and the enriched text that was
            # embedded so JD matching can use the same chunking pass later.
            if "text" in chunk:
                metadata["original_text"] = chunk["text"]
            if "embedding_text" in chunk:
                metadata["embedding_text"] = chunk["embedding_text"]

        embedded.append((text, embedding, metadata))

    return embedded


# =============================================================================
# PUBLIC API
# =============================================================================

def process_pdf_and_store(pdf_bytes: bytes, filename: str) -> int:
    file_hash = hashlib.sha256(pdf_bytes).hexdigest()
    source_name = make_unique_source_name(filename, file_hash)
    log.info("📄 Bắt đầu xử lý: %s (%d bytes)", filename, len(pdf_bytes))

    chunks = cv_pdf_to_semantic_chunks(pdf_bytes)
    log.info("✂️  %d chunk từ '%s'", len(chunks), filename)

    embedded = _embed_chunks(chunks, filename)

    log.info("💾 Lưu %d chunk vào MongoDB (source='%s')", len(embedded), filename)
    insert_chunks(
        embedded,
        source_name=source_name,
        file_hash=file_hash,
        original_filename=filename,
    )
    log.info("✅ Hoàn tất '%s' — %d chunk", filename, len(embedded))

    return len(embedded)


def process_multiple_pdfs(file_list) -> int:
    log.info("📂 Bắt đầu xử lý %d file", len(file_list))
    total = 0

    for idx, file in enumerate(file_list, 1):
        log.info("── File %d/%d: %s", idx, len(file_list), file.name)
        count = process_pdf_and_store(file.read(), file.name)
        total += count

    log.info("🏁 Hoàn tất — tổng %d chunk", total)
    return total


def answer_question(question: str, k: int = 3, source_filter: str = None) -> str:
    log.info("❓ Câu hỏi: %s", question)
    log.info("   k=%d | source_filter=%s", k, source_filter or "tất cả CV")

    query_embedding = get_embedding(question)
    log.info("   embedding dim=%d", len(query_embedding))

    results = search_similar_chunks(query_embedding, k=k, source_filter=source_filter)
    log.info("   MongoDB trả về %d chunk", len(results))

    if not results:
        log.warning("   ⚠️  Không tìm thấy chunk phù hợp")
        return "No relevant content found. Try rephrasing your question."

    for i, doc in enumerate(results, 1):
        log.info("   chunk %d: %s", i, str(doc["content"]).replace("\n", " ")[:80])

    context = "\n\n".join(str(doc["content"]) for doc in results)
    prompt = f"""Here is the information extracted from the CV:

{context}

Based on the above, please answer concisely:
{question}"""

    log.info("   Calling local LLM...")           # fix: bỏ log Bedrock Claude sai
    answer = generate_answer(prompt)
    log.info("   ✅ Trả lời: %s", answer[:120].replace("\n", " "))
    return answer
