# run_test.py
from rag_core import process_pdf_and_store, answer_question

# Test với 1 file PDF thật
with open("cv.pdf", "rb") as f:
    pdf_bytes = f.read()

count = process_pdf_and_store(pdf_bytes, "cv.pdf")
print(f"Đã lưu {count} chunk")

# Test hỏi đáp
answer = answer_question("Ứng viên có kỹ năng gì?")
print(f"Trả lời: {answer}")