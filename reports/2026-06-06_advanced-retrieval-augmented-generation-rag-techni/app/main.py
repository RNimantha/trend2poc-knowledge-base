import sys
import json
import logging
from typing import List

import openai

from app.config import settings
from app.core import (
    load_documents_from_json,
    prepare_document_chunks,
    embed_texts,
    build_faiss_index,
    hybrid_retrieve,
    generate_answer,
    DocumentChunk
)

logging.basicConfig(level=logging.INFO)


SAMPLE_INPUT_PATH = "examples/sample_input.json"


def main() -> None:
    openai.api_key = settings.OPENAI_API_KEY

    # Load sample documents
    try:
        documents = load_documents_from_json(SAMPLE_INPUT_PATH)
    except Exception as e:
        logging.error(f"Failed to load documents from {SAMPLE_INPUT_PATH}: {e}")
        sys.exit(1)

    # Prepare chunks
    chunks: List[DocumentChunk] = prepare_document_chunks(documents)
    if not chunks:
        logging.error("No document chunks available.")
        sys.exit(1)

    # Embed chunks
    try:
        chunk_texts = [chunk.text for chunk in chunks]
        embeddings = embed_texts(chunk_texts)
    except Exception as e:
        logging.error(f"Embedding failed: {e}")
        sys.exit(1)

    # Build FAISS index
    try:
        index = build_faiss_index(embeddings)
    except Exception as e:
        logging.error(f"Failed to build FAISS index: {e}")
        sys.exit(1)

    # Prompt user for query
    print("Enter your query (or press Enter to use sample query):", end=" ")
    user_query = input().strip()

    if not user_query:
        # Use sample query from sample input
        sample_query = "What are the benefits of renewable energy?"
        print(f"Using sample query: {sample_query}")
        user_query = sample_query

    # Retrieve relevant chunks
    try:
        retrieved = hybrid_retrieve(user_query, chunks, index, embeddings, top_k=5)
    except Exception as e:
        logging.error(f"Retrieval failed: {e}")
        sys.exit(1)

    print(f"\nRetrieved {len(retrieved)} relevant document chunks.")

    # Generate answer
    try:
        answer = generate_answer(user_query, [chunk for chunk, _ in retrieved])
    except Exception as e:
        logging.error(f"Generation failed: {e}")
        sys.exit(1)

    print(f"\nGenerated Answer:\n{answer}")


if __name__ == "__main__":
    main()
