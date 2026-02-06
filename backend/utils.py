# File: backend/main.py
import fitz  # PyMuPDF,Mở file PDF ,Mở file PDF

def get_pdf_chunks(pdf_path, max_chars=1000, overlap=200):
    doc = fitz.open(pdf_path)
    chunks = []
    text = ""
    for page in doc:
        text += page.get_text()

    start = 0
    while start < len(text):
        end = start + max_chars
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += max_chars - overlap

    doc.close()
    return chunks

# Function to calculate cosine similarity
# between two vectors
def cosine_similarity(a, b):#So sánh độ giống nhau giữa: embedding của câu hỏi người dùng, embedding của từng đoạn PDF
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x ** 2 for x in a) ** 0.5
    norm_b = sum(x ** 2 for x in b) ** 0.5
    print(f"[DEBUG] Cosine similarity: {dot_product}, Norm A: {norm_a}, Norm B: {norm_b}")
    return dot_product / (norm_a * norm_b + 1e-8)  # small epsilon to avoid zero division
    #Đây là retrieval trong RAG.
