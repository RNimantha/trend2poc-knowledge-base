# hype-rag-poc

## Advanced Retrieval-Augmented Generation (RAG) Techniques Including Hypothetical Prompt Embeddings (HyPE)

This project demonstrates a minimal Retrieval-Augmented Generation pipeline using Hypothetical Prompt Embeddings (HyPE) to improve semantic retrieval alignment.

---

### Prerequisites

- Python 3.11+
- OpenAI API key (for generating hypothetical questions and generation)

### Installation

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd hype-rag-poc
   ```

2. Copy `.env.example` to `.env` and set your OpenAI API key:
   ```bash
   cp .env.example .env
   # Edit .env to add your OPENAI_API_KEY
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Configuration

- The `.env` file must contain your OpenAI API key as `OPENAI_API_KEY`.

### Running the Demo

Run the main application:

```bash
python app/main.py
```

You will be prompted to enter queries. The system will retrieve relevant document chunks using HyPE-based retrieval and generate a response grounded on the retrieved context.

### Example Output

```
Enter your query (or 'exit' to quit): What is HyPE?

Retrieved Chunks:
- Chunk 1: Hypothetical prompt embeddings (HyPE) improve semantic retrieval by ...
- Chunk 2: HyPE generates hypothetical questions to better align embeddings ...

Generated Response:
Hypothetical Prompt Embeddings (HyPE) is a technique that enhances retrieval by generating hypothetical questions for document chunks, embedding those questions, and using them to improve semantic search alignment.
```

### How This POC Demonstrates the Concept

- Splits documents into chunks.
- Uses OpenAI LLM to generate hypothetical questions for each chunk.
- Embeds these hypothetical questions using sentence-transformers.
- Builds an in-memory FAISS vector index of these embeddings.
- At query time, embeds the user query and retrieves relevant chunks based on HyPE embeddings.
- Demonstrates a simple generation step grounded on retrieved context.

### Known Limitations

- No persistent storage; index is in-memory only.
- Requires internet and OpenAI API key; no offline fallback.
- Hypothetical question generation is limited to a small fixed number per chunk.
- Generation is a simple prompt concatenation demo, not a full RAG pipeline.
- Basic error handling only; no advanced recovery.
