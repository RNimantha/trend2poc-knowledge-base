import re
import json
import logging
from typing import Any, List, Tuple, Dict

import openai
import faiss
from tqdm import tqdm

from app.config import settings


logging.basicConfig(level=logging.INFO)


class DocumentChunk:
    def __init__(self, doc_id: int, chunk_id: int, text: str):
        self.doc_id = doc_id
        self.chunk_id = chunk_id
        self.text = text


def chunk_document(text: str, max_chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split a document into overlapping chunks of max_chunk_size characters."""
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + max_chunk_size, text_len)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == text_len:
            break
        start = end - overlap  # overlap
    return chunks


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Get embeddings for a list of texts using OpenAI."""
    embeddings = []
    batch_size = 16
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding texts"):
        batch = texts[i : i + batch_size]
        try:
            response = openai.embeddings.create(
                model=settings.embedding_model_name,
                input=batch
            )
            batch_embeddings = [item["embedding"] for item in response["data"]]
            embeddings.extend(batch_embeddings)
        except openai.error.OpenAIError as e:
            logging.error(f"OpenAI embedding API error: {e}")
            raise
    return embeddings


def build_faiss_index(embeddings: List[List[float]]) -> faiss.IndexFlatIP:
    """Build a FAISS index for inner product similarity (cosine similarity with normalized vectors)."""
    import numpy as np

    dim = len(embeddings[0])
    index = faiss.IndexFlatIP(dim)

    # Normalize embeddings to unit length for cosine similarity
    np_embeddings = np.array(embeddings, dtype="float32")
    faiss.normalize_L2(np_embeddings)

    index.add(np_embeddings)
    return index


def normalize_query_embedding(query_embedding: List[float]) -> List[float]:
    import numpy as np
    np_emb = np.array(query_embedding, dtype="float32")
    faiss.normalize_L2(np_emb)
    return np_emb


def keyword_filter(chunks: List[DocumentChunk], query: str) -> List[int]:
    """Return indices of chunks that contain at least one keyword from the query."""
    query_keywords = set(re.findall(r"\w+", query.lower()))
    matched_indices = []
    for idx, chunk in enumerate(chunks):
        chunk_words = set(re.findall(r"\w+", chunk.text.lower()))
        if query_keywords.intersection(chunk_words):
            matched_indices.append(idx)
    return matched_indices


def hybrid_retrieve(
    query: str,
    chunks: List[DocumentChunk],
    index: faiss.IndexFlatIP,
    embeddings: List[List[float]],
    top_k: int = 5
) -> List[Tuple[DocumentChunk, float]]:
    """Retrieve top_k chunks by hybrid semantic + keyword filtering.

    Returns list of (DocumentChunk, score) tuples.
    """
    # Embed query
    try:
        response = openai.embeddings.create(
            model=settings.embedding_model_name,
            input=[query]
        )
        query_embedding = response["data"][0]["embedding"]
    except openai.error.OpenAIError as e:
        logging.error(f"OpenAI embedding API error: {e}")
        raise

    import numpy as np
    query_emb_np = np.array(query_embedding, dtype="float32")
    faiss.normalize_L2(query_emb_np)

    # Search FAISS
    D, I = index.search(np.expand_dims(query_emb_np, axis=0), top_k * 3)  # get more for filtering

    # Keyword filter
    keyword_matched_indices = set(keyword_filter(chunks, query))

    # Combine semantic and keyword filtering
    candidates = []
    for score, idx in zip(D[0], I[0]):
        if idx == -1:
            continue
        if idx in keyword_matched_indices:
            candidates.append((chunks[idx], float(score)))

    # If no candidates after filtering, fallback to top semantic
    if not candidates:
        candidates = [(chunks[idx], float(score)) for score, idx in zip(D[0], I[0]) if idx != -1][:top_k]

    # Simple re-rank by keyword overlap count
    query_keywords = set(re.findall(r"\w+", query.lower()))

    def keyword_overlap_score(chunk: DocumentChunk) -> int:
        chunk_words = set(re.findall(r"\w+", chunk.text.lower()))
        return len(query_keywords.intersection(chunk_words))

    candidates.sort(key=lambda x: (keyword_overlap_score(x[0]), x[1]), reverse=True)

    return candidates[:top_k]


def generate_answer(query: str, retrieved_chunks: List[DocumentChunk]) -> str:
    """Generate an answer grounded in retrieved documents."""
    context_text = "\n---\n".join(chunk.text for chunk in retrieved_chunks)

    prompt = (
        f"You are an AI assistant. Use the following context to answer the question."
        f" If the answer is not contained within the context, say 'I don't know.'\n\n"
        f"Context:\n{context_text}\n\nQuestion: {query}\nAnswer:"
    )

    try:
        response = openai.chat.completions.create(
            model=settings.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300
        )
        answer = response.choices[0].message.content.strip()
    except openai.error.OpenAIError as e:
        logging.error(f"OpenAI generation API error: {e}")
        raise

    return answer


def load_documents_from_json(path: str) -> List[Dict[str, Any]]:
    """Load documents from a JSON file with format [{"id": int, "text": str}, ...]"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def prepare_document_chunks(documents: List[Dict[str, Any]]) -> List[DocumentChunk]:
    """Chunk all documents and return list of DocumentChunk."""
    all_chunks = []
    for doc in documents:
        doc_id = doc.get("id")
        text = doc.get("text", "")
        chunks = chunk_document(text)
        for i, chunk_text in enumerate(chunks):
            all_chunks.append(DocumentChunk(doc_id=doc_id, chunk_id=i, text=chunk_text))
    return all_chunks
