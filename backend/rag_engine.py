# File: backend/main.py

import os# làm việc với file, folder
import fitz  # PyMuPDF , đọc nội dung PDF
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain_community.llms import Ollama # gọi AI local (LLM) để trả lời
# from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaEmbeddings # biến text thành vector (để search giống nhau)
from utils import get_pdf_chunks, cosine_similarity # đo độ giống nhau giữa câu hỏi và từng đoạn PDF
# Đây là “đồ nghề”: đọc PDF, biến chữ thành vector, gọi AI trả lời.
# Set base URL for Ollama from environment variable
OLLAMA_API_URL = os.getenv('OLLAMA_API_URL', 'http://ollama:11434') # địa chỉ con Ollama server đang chạy trong Docker
EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf' # model dùng để vector hoá PDF
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF' # model dùng để trả lời câu hỏi

# LangChain LLM and embedding clients, Khởi tạo AI (LLM) và Embedding
llm = Ollama(
    model=LANGUAGE_MODEL,
    base_url=OLLAMA_API_URL,
    temperature=0,
    top_p=0.9,
    num_ctx=4096,       
    num_predict=512
)

embedding_client = OllamaEmbeddings(# 1 model trả lời , 1 model biến text → vector
    model=EMBEDDING_MODEL,
    base_url=OLLAMA_API_URL
)
# VECTOR_DB – “database tự chế”,  Đây là database trong RAM
VECTOR_DB = []

def embed_text(text): # “Biến chữ thành toán học để so sánh giống nhau.”
    return embedding_client.embed_query(text)

def add_chunk_to_database(chunk):
    embedding = embed_text(chunk)
    VECTOR_DB.append((chunk,     embedding))  # Lưu vào VECTOR_DB
#Tắt server là mất hết
#Mỗi lần restart lại phải đọc PDF lại từ đầu

def ingest_pdfs(folder_path='documents'): # < Đọc PDF và chia nhỏ>
    if VECTOR_DB:
        print("[INFO] VECTOR_DB already populated.")    # Nếu DB đã có dữ liệu thì không đọc lại PDF nữa
        return
    for filename in os.listdir(folder_path):
        print(f"[DEBUG] Processing file: {filename}")
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            print(f"[INFO] Ingesting: {file_path}")
            chunks = get_pdf_chunks(file_path)
            for chunk in chunks:
                print(f"[DEBUG] Adding chunk: {chunk[:80]}...")
                add_chunk_to_database(chunk)
# 47-55 “Cho AI đọc tài liệu trước khi hỏi.”
def retrieve(query, top_n=3): # Retrieve – tìm đoạn PDF liên quan nhất, Biến câu hỏi thành vector.
    query_embedding = embed_text(query)
    similarities = []
    for chunk, embedding in VECTOR_DB:
        similarity = cosine_similarity(query_embedding, embedding)
        similarities.append((chunk, similarity)) # 59-62 , So câu hỏi với từng đoạn PDF → tính điểm giống nhau.
    similarities.sort(key=lambda x: x[1], reverse=True) #  Sắp xếp: đoạn nào giống nhất thì lên đầu.
    
    print(f"[INFO] Top {top_n} matches for query '{query}':")
    for chunk, score in similarities[:top_n]: # 
        print(f"  [SCORE: {score:.4f}] {chunk[:100]}")
    
    return similarities[:top_n] # Lấy top N đoạn liên quan nhất. , Đây chính là Retrieval trong RAG.

def process_query(user_input): # Đảm bảo PDF đã được load vào VECTOR_DB.
    ingest_pdfs() # process_query – trái tim của RAG

    retrieved_knowledge = retrieve(user_input, top_n=6)# Lấy 6 đoạn PDF liên quan nhất tới câu hỏi.

    context = "\n\n".join(
        [f"[SOURCE {i+1}]\n{chunk}" for i, (chunk, _) in enumerate(retrieved_knowledge)] # Ghép các đoạn PDF lại thành context cho AI đọc.
    )

    prompt = f"""
You are a legal assistant.

You MUST answer the question using ONLY the information provided in the context below.
DO NOT use any outside knowledge.
DO NOT guess or make up information.

If the answer cannot be found in the context, reply exactly:
"Không tìm thấy thông tin trong tài liệu."

Context:
{context}

Question:
{user_input}

Answer:
- Quote or paraphrase relevant articles if applicable
- Be concise and accurate
"""
# Ép AI chỉ được dùng PDF, 
    return llm.invoke(prompt)

