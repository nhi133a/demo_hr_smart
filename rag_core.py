from pdf_utils import extract_text_from_pdf, cv_pdf_to_section_chunks
from bedrock_utils import get_embedding
from local_llm import generate_answer as generate_answer_openai
from mongo_utils import insert_chunks, search_similar_chunks


# ✅ helper: normalize chunk → string
def _normalize_chunk(chunk):
    if isinstance(chunk, dict):
        return chunk.get("content", str(chunk))
    return str(chunk)


# FIX 3: thống nhất cách truyền file — dùng bytes cho cả 2 hàm
def process_pdf_and_store(pdf_bytes, filename):
    text = extract_text_from_pdf(pdf_bytes)
    chunks = cv_pdf_to_section_chunks(pdf_bytes)

    embedded_chunks = []
    for chunk in chunks:
        chunk_text = _normalize_chunk(chunk)   # ✅ FIX
        embedding = get_embedding(chunk_text)  # ✅ FIX
        embedded_chunks.append((chunk_text, embedding))

    insert_chunks(embedded_chunks, source_name=filename)
    return len(embedded_chunks)


def process_multiple_pdfs(file_list):
    total_chunks = 0
    for file in file_list:
        pdf_bytes = file.read()
        text = extract_text_from_pdf(pdf_bytes)
        chunks = cv_pdf_to_section_chunks(pdf_bytes)

        embedded_chunks = []
        for chunk in chunks:
            chunk_text = _normalize_chunk(chunk)   # ✅ FIX
            embedding = get_embedding(chunk_text)  # ✅ FIX
            embedded_chunks.append((chunk_text, embedding))

        insert_chunks(embedded_chunks, source_name=file.name)
        total_chunks += len(embedded_chunks)

    return total_chunks


def answer_question(question: str, k: int = 3) -> str:
    query_embedding = get_embedding(str(question))  # ✅ thêm safety
    results = search_similar_chunks(query_embedding, k=k)

    if not results:
        return "No relevant content found in the documents. Try rephrasing your question."

    context = "\n".join([str(doc["content"]) for doc in results])  # ✅ safety

    prompt = f"""Here is the information we have: {context}.
    Taking this information into account, clearly answer the following question: {question} """

    return generate_answer_openai(prompt)