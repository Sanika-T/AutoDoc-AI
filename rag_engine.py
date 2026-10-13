import os
import re
import time
import hashlib
from typing import List, Dict, Any

import chromadb
import fitz
import numpy as np
from google import genai
from google.genai import types

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
COLLECTION_NAME = "autodoc_hld_knowledge"

_client = None
_collection = None

def get_gemini_client():
    global _client
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is missing. Add it to .env.")
    if _client is None:
        _client = genai.Client(api_key=key)
    return _client

def extract_documents(uploaded_files) -> List[Dict[str, Any]]:
    """Extract text while retaining source filename and page number."""
    docs = []
    for uploaded in uploaded_files:
        name = uploaded.name
        raw = uploaded.getvalue()
        suffix = name.lower().rsplit(".", 1)[-1]
        if suffix == "pdf":
            pdf = fitz.open(stream=raw, filetype="pdf")
            for page_index, page in enumerate(pdf):
                text = page.get_text("text").strip()
                if text:
                    docs.append({
                        "text": text, "source": name, "page": page_index + 1
                    })
        else:
            text = raw.decode("utf-8", errors="ignore")
            # Text/Markdown has no page structure; treat the file as page 1.
            docs.append({"text": text, "source": name, "page": 1})
    docs = [d for d in docs if d["text"].strip()]
    if not docs:
        raise ValueError("No readable text found. If the PDF is scanned, run OCR first.")
    return docs

def chunk_documents(docs: List[Dict[str, Any]], chunk_size: int = 1000, overlap: int = 150):
    chunks = []
    for doc in docs:
        text = re.sub(r"\s+", " ", doc["text"]).strip()
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()
            if len(chunk_text) > 30:
                chunks.append({
                    "text": chunk_text,
                    "source": doc["source"],
                    "page": int(doc["page"]),
                    "chunk_id": len(chunks)
                })
            if end >= len(text):
                break
            start = max(end - overlap, start + 1)
    return chunks

def embed_texts(texts: List[str], task_type: str) -> List[List[float]]:
    client = get_gemini_client()
    vectors = []
    # Small batches keep requests manageable and allow progress-friendly execution.
    for start in range(0, len(texts), 8):
        batch = texts[start:start + 8]
        result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=768),
        )
        for embedding in result.embeddings:
            vectors.append(embedding.values)
        if start + 8 < len(texts):
            time.sleep(0.15)
    return vectors

def _new_collection():
    global _collection
    db = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        db.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    _collection = db.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    return _collection

def build_index(docs, chunk_size=1000, overlap=150):
    global _collection
    chunks = chunk_documents(docs, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        raise ValueError("The documents did not contain enough text to index.")
    vectors = embed_texts([c["text"] for c in chunks], "RETRIEVAL_DOCUMENT")
    collection = _new_collection()
    # Chroma metadata values must be scalar types.
    ids, texts, metadatas = [], [], []
    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        digest = hashlib.sha1(f"{chunk['source']}|{chunk['page']}|{i}".encode()).hexdigest()[:16]
        ids.append(digest)
        texts.append(chunk["text"])
        metadatas.append({
            "source": chunk["source"],
            "page": int(chunk["page"]),
            "chunk_id": int(i)
        })
    for start in range(0, len(ids), 100):
        collection.add(
            ids=ids[start:start+100],
            documents=texts[start:start+100],
            metadatas=metadatas[start:start+100],
            embeddings=vectors[start:start+100]
        )
    _collection = collection
    return {
        "documents": len(set(c["source"] for c in chunks)),
        "chunks": len(chunks),
        "file_names": sorted(set(c["source"] for c in chunks))
    }

def _get_collection():
    global _collection
    if _collection is None:
        db = chromadb.PersistentClient(path=CHROMA_PATH)
        try:
            _collection = db.get_collection(COLLECTION_NAME)
        except Exception:
            return None
    return _collection

def retrieve(question: str, top_k: int = 4):
    collection = _get_collection()
    if collection is None or collection.count() == 0:
        raise RuntimeError("No knowledge index exists. Upload documents and build the index.")
    query_vector = embed_texts([question], "RETRIEVAL_QUERY")[0]
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"]
    )
    sources = []
    for doc, metadata, distance in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        sources.append({
            "text": doc,
            "source": metadata["source"],
            "page": metadata["page"],
            # Chroma cosine distance is typically 0..2; this is similarity-like, not confidence.
            "similarity": max(-1.0, min(1.0, 1.0 - float(distance)))
        })
    return sources

def answer_question(question: str, top_k: int = 4):
    sources = retrieve(question, top_k=top_k)
    context = "\n\n".join(
        f"[Source {i}: {s['source']}, page {s['page']}]\n{s['text']}"
        for i, s in enumerate(sources, 1)
    )
    prompt = f"""
You are AutoDoc AI, an assistant for understanding automotive High-Level Design (HLD) documents.

STRICT RULES:
1. Answer only using the CONTEXT below.
2. If the context does not contain enough evidence, say: "I couldn't find enough evidence in the uploaded document to answer that reliably."
3. Never invent component names, signal names, requirements, measurements, or page references.
4. Cite every important factual claim using the exact source labels, such as [Source 1].
5. Separate explicit facts from reasonable interpretations.
6. Be concise and technically clear. Use bullets when helpful.
7. This is an academic assistant, not a formal engineering approval tool.

QUESTION:
{question}

CONTEXT:
{context}

Write the answer with source citations. Do not cite sources that are not present in the context.
"""
    response = get_gemini_client().models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=900
        )
    )
    answer = response.text or "The model returned no text. Please try again."
    return {"answer": answer, "sources": sources}

def clear_index():
    global _collection
    db = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        db.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    _collection = None

def get_index_stats():
    collection = _get_collection()
    return collection.count() if collection is not None else 0
