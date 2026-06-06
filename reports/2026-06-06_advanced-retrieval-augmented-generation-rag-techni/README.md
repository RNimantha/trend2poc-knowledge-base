# minimal-rag-poc

A minimal Retrieval-Augmented Generation (RAG) pipeline combining semantic vector search with keyword filtering to ground LLM generation in retrieved documents.

## Prerequisites

- Python 3.11+
- OpenAI API key (for embeddings and generation)

## Installation

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd minimal-rag-poc
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

## Configuration

Set your OpenAI API key in the `.env` file:

```
OPENAI_API_KEY=
```

## Running the POC

Run the main application:

```bash
python app/main.py
```

You will be prompted to enter a query or press Enter to use the sample query from `examples/sample_input.json`.

## Example Output

```
Enter your query (or press Enter to use sample query): What are the benefits of renewable energy?

Retrieved 3 relevant document chunks.

Generated Answer:
Renewable energy offers multiple benefits including reducing greenhouse gas emissions, decreasing dependence on fossil fuels, and promoting sustainable development. It helps mitigate climate change and can provide economic growth through new jobs in clean energy sectors.
```

## How This POC Demonstrates the Concept

This project shows a minimal RAG pipeline by:

- Chunking documents into smaller pieces
- Generating semantic embeddings using OpenAI
- Building an in-memory FAISS vector index
- Performing hybrid retrieval combining semantic similarity and keyword filtering
- Augmenting the LLM prompt with retrieved documents
- Generating grounded answers using OpenAI's text generation

## Known Limitations

- Does not include advanced re-ranker models beyond simple keyword overlap scoring.
- Uses OpenAI API for embeddings and generation, requiring internet and API key.
- No persistent vector database; index built in-memory each run.
- No fine-tuning or reinforcement learning feedback loops.
- Limited to small document sets for demonstration purposes.
