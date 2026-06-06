# token-optimization-poc

A minimal proof-of-concept demonstrating core token optimization techniques including token counting, prompt compression, context management, model tiering, and semantic caching in a simulated LLM interaction.

## Prerequisites

- Python 3.11 or higher
- No real API keys required for mock mode

## Installation

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd token-optimization-poc
   ```
2. Copy `.env.example` to `.env` and fill in any required variables (none required for mock mode):
   ```bash
   cp .env.example .env
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

The project uses environment variables loaded via `pydantic-settings` and `python-dotenv`. The `.env` file supports:

- `OPENAI_API_KEY` (optional): API key for OpenAI models if switching from mock to real API calls.
- `USE_MOCK_MODEL` (optional): Flag (`true` or `false`) to use mock models for offline demonstration. Defaults to `true`.

## Running the POC

Run the main app with example input:

```bash
python app/main.py examples/sample_input.json
```

## Example Output

```json
{
  "input": "Explain token optimization techniques.",
  "token_count": 5,
  "compressed_prompt": "Explain token optimization techniques.",
  "context": "Explain token optimization techniques.",
  "model": "mock-tier-1",
  "response": "[Mock Tier 1 Model Response] Explanation of token optimization techniques.",
  "cache_hit": false
}
```

## How This POC Demonstrates the Concept

- **Token Counting:** Uses `tiktoken` to count tokens in prompts.
- **Prompt Compression:** Applies a simple heuristic to shorten prompts.
- **Context Management:** Simulates context truncation and summarization.
- **Model Tiering:** Routes prompts to mock models with different token costs and qualities.
- **Semantic Caching:** Caches responses based on token overlap similarity.

## Known Limitations

- Does not integrate with real LLM APIs by default; uses mock models.
- Prompt compression is heuristic and simplistic.
- Semantic caching uses simple token set overlap rather than embeddings.
- Context management is basic truncation and simulated summarization.
- Model tiering is simulated with mock models.
