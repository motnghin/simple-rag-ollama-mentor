# backend/main.py
import os
os.environ['OLLAMA_API_URL'] = os.getenv('OLLAMA_API_URL', 'http://ollama:11434')

from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from rag_engine import process_query, ingest_pdfs

app = FastAPI() # Khởi tạo fastapi

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def load_documents():
    ingest_pdfs("documents") # Load PDf khi server khởi động 

@app.get("/")
def read_root():
    ingest_pdfs()
    return {"message": "RAG Backend is running."}

@app.get("/query/")
def query_knowledge(query: str):#Nhận câu hỏi từ frontend
    response = process_query(query)# Gọi core logic RAG
    return {"response": response} 

if __name__ == "__main__":
    print("🔄 Starting PDF ingestion and embedding...")
    ingest_pdfs()
    print("✅ PDF ingestion and embedding completed.")
#