# File: backend/main.py

import os
import fitz  # PyMuPDF
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain_community.llms import Ollama
# from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaEmbeddings
from utils import get_pdf_chunks, cosine_similarity

# Set base URL for Ollama from environment variable
OLLAMA_API_URL = os.getenv('OLLAMA_API_URL', 'http://ollama:11434')

EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'

# LangChain LLM and embedding clients
llm = Ollama(
    model=LANGUAGE_MODEL,
    base_url=OLLAMA_API_URL,
    temperature=0,
    top_p=0.9,
    num_ctx=4096,       
    num_predict=512
)

embedding_client = OllamaEmbeddings(
    model=EMBEDDING_MODEL,
    base_url=OLLAMA_API_URL
)

VECTOR_DB = []

def embed_text(text):
    return embedding_client.embed_query(text)

def add_chunk_to_database(chunk):
    embedding = embed_text(chunk)
    VECTOR_DB.append((chunk, embedding))

def ingest_pdfs(folder_path='documents'):
    if VECTOR_DB:
        print("[INFO] VECTOR_DB already populated.")
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

def retrieve(query, top_n=3):
    query_embedding = embed_text(query)
    similarities = []
    for chunk, embedding in VECTOR_DB:
        similarity = cosine_similarity(query_embedding, embedding)
        similarities.append((chunk, similarity))
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    print(f"[INFO] Top {top_n} matches for query '{query}':")
    for chunk, score in similarities[:top_n]:
        print(f"  [SCORE: {score:.4f}] {chunk[:100]}")
    
    return similarities[:top_n]

def process_query(user_input):
    ingest_pdfs()

    retrieved_knowledge = retrieve(user_input, top_n=6)

    context = "\n\n".join(
        [f"[SOURCE {i+1}]\n{chunk}" for i, (chunk, _) in enumerate(retrieved_knowledge)]
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

    return llm.invoke(prompt)

