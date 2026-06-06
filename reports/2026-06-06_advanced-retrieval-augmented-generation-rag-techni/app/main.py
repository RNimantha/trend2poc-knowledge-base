import json
import logging
import sys
from pathlib import Path
from typing import List

from app.config import settings
from app.core import HyPERetrieval


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_documents_from_json(path: Path) -> List[str]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            logger.error("Input JSON must be a list of document strings.")
            sys.exit(1)
        return data
    except Exception as e:
        logger.error(f"Failed to load documents from {path}: {e}")
        sys.exit(1)


def main() -> None:
    print("Loading documents from examples/sample_input.json...")
    documents = load_documents_from_json(Path(__file__).parent.parent / "examples" / "sample_input.json")

    retriever = HyPERetrieval()

    print("Chunking documents and building HyPE index (this may take a moment)...")
    chunks = retriever.chunk_documents(documents)
    retriever.build_index(chunks)

    print("HyPE index built. You can now enter queries.")

    while True:
        query = input("Enter your query (or 'exit' to quit): ").strip()
        if query.lower() in {"exit", "quit"}:
            print("Exiting.")
            break
        if not query:
            continue

        retrieved = retriever.retrieve(query, top_k=3)
        if not retrieved:
            print("No relevant chunks found.")
            continue

        print("\nRetrieved Chunks:")
        for i, (chunk_text, score) in enumerate(retrieved, 1):
            snippet = chunk_text[:200].replace('\n', ' ') + ("..." if len(chunk_text) > 200 else "")
            print(f"- Chunk {i} (score: {score:.3f}): {snippet}")

        retrieved_texts = [chunk for chunk, _ in retrieved]
        response = retriever.generate_response(query, retrieved_texts)

        print("\nGenerated Response:")
        print(response)
        print("\n" + "-" * 40 + "\n")


if __name__ == "__main__":
    main()
