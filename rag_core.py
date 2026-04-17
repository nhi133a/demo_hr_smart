from pdf_utils import cv_pdf_to_semantic_chunks #Chức năng: trích xuất nội dung từ file PDF và chia thành các chunk có ngữ cảnh rõ ràng (vd: section, text) để dễ dàng tạo embedding và lưu vào MongoDB.
from bedrock_utils import get_embedding #Chức năng: tạo embedding vector cho text bằng cách gọi API của AWS Bedrock.
from local_llm import generate_answer as generate_answer_openai #Chức năng: tạo câu trả lời từ LLM cục bộ (ở đây là OpenAI) dựa trên prompt đã cho.
from mongo_utils import insert_chunks, search_similar_chunks #Chức năng: tương tác với MongoDB để lưu trữ các chunk đã embed và tìm kiếm các chunk tương tự dựa trên embedding vector.
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _normalize_chunk(chunk) -> tuple[str, str]: #Chức năng: chuẩn hóa chunk đầu vào để đảm bảo có định dạng (section, text) rõ ràng trước khi embed và lưu vào MongoDB.
    """
    FIX: pdf_utils trả về {"section": ..., "text": ...}
    Trả về (section, text) để lưu context rõ ràng vào MongoDB.
    """
    if isinstance(chunk, dict): #Chức năng: nếu chunk là dict, cố gắng lấy section và text từ các key tương ứng, nếu không có thì dùng default.
        section = chunk.get("section", "other")
        text    = chunk.get("text") or chunk.get("content") or str(chunk)
    else: 
        section = "other"
        text    = str(chunk)
    return section, text


def _build_chunk_text(section: str, text: str) -> str: #Chức năng: thêm label section vào đầu text để giúp LLM nhận diện đúng ngữ cảnh của chunk khi tạo embedding và lưu vào MongoDB.
    """
    Thêm section label vào text trước khi embed và lưu.
    Giúp LLM nhận diện đúng context (NAME, SKILLS, EXPERIENCE...)
    """
    return f"[{section.upper()}]\n{text}"


def process_pdf_and_store(pdf_bytes: bytes, filename: str) -> int: #Chức năng: xử lý file PDF, trích xuất các chunk có ngữ cảnh rõ ràng, tạo embedding cho từng chunk và lưu vào MongoDB. Trả về số lượng chunk đã lưu.
    log.info("📄 Bắt đầu xử lý file: %s (%d bytes)", filename, len(pdf_bytes))

    chunks = cv_pdf_to_semantic_chunks(pdf_bytes)
    log.info("✂️  pdf_utils trả về %d chunk từ file '%s'", len(chunks), filename)

    embedded_chunks = []
    for i, chunk in enumerate(chunks, 1):
        section, text = _normalize_chunk(chunk)
        chunk_text    = _build_chunk_text(section, text)
        log.info("  [%d/%d] section=%-12s | text preview: %.60s", i, len(chunks), section, text.replace("\n", " "))

        embedding     = get_embedding(chunk_text)
        log.info("  [%d/%d] ✔ embedding nhận được (dim=%d)", i, len(chunks), len(embedding))

        embedded_chunks.append((chunk_text, embedding))

    log.info("💾 Lưu %d chunk vào MongoDB (source='%s')", len(embedded_chunks), filename)
    insert_chunks(embedded_chunks, source_name=filename)
    log.info("✅ Hoàn tất file '%s' — tổng %d chunk đã lưu", filename, len(embedded_chunks))

    return len(embedded_chunks)


def process_multiple_pdfs(file_list) -> int: #Chức năng: xử lý nhiều file PDF, trích xuất các chunk có ngữ cảnh rõ ràng, tạo embedding cho từng chunk và lưu vào MongoDB. Trả về tổng số lượng chunk đã lưu từ tất cả các file.
    log.info("📂 Bắt đầu xử lý %d file PDF", len(file_list))
    total = 0

    for file_idx, file in enumerate(file_list, 1):
        log.info("── File %d/%d: %s", file_idx, len(file_list), file.name)
        pdf_bytes = file.read()
        chunks    = cv_pdf_to_semantic_chunks(pdf_bytes)
        log.info("   pdf_utils trả về %d chunk từ '%s'", len(chunks), file.name)

        embedded_chunks = []
        for i, chunk in enumerate(chunks, 1):
            section, text = _normalize_chunk(chunk)
            chunk_text    = _build_chunk_text(section, text)
            log.info("   [%d/%d] section=%-12s | text preview: %.60s", i, len(chunks), section, text.replace("\n", " "))

            embedding     = get_embedding(chunk_text)
            log.info("   [%d/%d] ✔ embedding nhận được (dim=%d)", i, len(chunks), len(embedding))

            embedded_chunks.append((chunk_text, embedding))

        log.info("   💾 Lưu %d chunk (source='%s')", len(embedded_chunks), file.name)
        insert_chunks(embedded_chunks, source_name=file.name)
        total += len(embedded_chunks)
        log.info("   ✅ Xong '%s' — %d chunk", file.name, len(embedded_chunks))

    log.info("🏁 Hoàn tất tất cả — tổng cộng %d chunk đã lưu", total)
    return total


def answer_question(question: str, k: int = 3, source_filter: str = None) -> str: #Chức năng: trả lời câu hỏi dựa trên các chunk đã lưu trong MongoDB. Tìm kiếm các chunk tương tự dựa trên embedding của câu hỏi, sau đó tạo prompt và gọi LLM để tạo câu trả lời. Có thể lọc kết quả tìm kiếm theo tên file CV để chỉ tập trung vào một CV cụ thể.
    """
    source_filter: tên file CV (vd: "nguyen_van_a.pdf")
    → chỉ search chunks của CV đó trong MongoDB.

    LƯU Ý: MongoDB Atlas vector_index phải có filter field "source":
    {
      "fields": [
        {"type": "vector", "path": "embedding", ...},
        {"type": "filter", "path": "source"}   ← bắt buộc
      ]
    }
    """
    log.info("❓ Câu hỏi: %s", question)
    log.info("   Tham số: k=%d, source_filter=%s", k, source_filter or "None (tất cả CV)")

    query_embedding = get_embedding(str(question))
    log.info("   ✔ Đã tạo embedding cho câu hỏi (dim=%d)", len(query_embedding))

    results = search_similar_chunks(
        query_embedding,
        k=k,
        source_filter=source_filter
    )
    log.info("   MongoDB trả về %d chunk liên quan", len(results))

    if not results:
        log.warning("   ⚠️  Không tìm thấy chunk nào phù hợp")
        return "No relevant content found in the documents. Try rephrasing your question."

    for i, doc in enumerate(results, 1):
        preview = str(doc["content"]).replace("\n", " ")[:80]
        log.info("   chunk %d: %s", i, preview)

    context = "\n\n".join([str(doc["content"]) for doc in results])
    prompt  = f"""Here is the information extracted from the CV:

{context}

Based on the above, please answer concisely:
{question}"""

    log.info("   🤖 Gọi LLM để tạo câu trả lời...")
    answer = generate_answer_openai(prompt) #Chức năng: gọi LLM cục bộ (ở đây là OpenAI) để tạo câu trả lời dựa trên prompt đã xây dựng từ các chunk tìm được.
    log.info("   ✅ LLM trả lời: %s", answer[:120].replace("\n", " "))
    return answer