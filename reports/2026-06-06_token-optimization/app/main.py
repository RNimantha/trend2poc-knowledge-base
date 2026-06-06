import sys
import json
from app.config import settings
from app.core import TokenCounter, compress_prompt, ContextManager, ModelRouter, SemanticCache
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_input(file_path: str) -> str:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, dict) or "input" not in data:
                raise ValueError("Input JSON must be an object with an 'input' field.")
            return data["input"]
    except FileNotFoundError:
        logger.error(f"Input file not found: {file_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        logger.error(f"Input file is not valid JSON: {file_path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading input file: {e}")
        sys.exit(1)


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <input_json_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    user_input = load_input(input_file)

    token_counter = TokenCounter()
    context_manager = ContextManager()
    model_router = ModelRouter()
    semantic_cache = SemanticCache()

    token_count = token_counter.count_tokens(user_input)

    compressed_prompt = compress_prompt(user_input)

    context = context_manager.manage_context(compressed_prompt)

    cached_response = semantic_cache.get_cached_response(context)
    cache_hit = cached_response is not None

    if cache_hit:
        response = cached_response
    else:
        model_info = model_router.route(context, token_count)
        response = model_router.generate_response(context, model_info["name"])
        semantic_cache.add_to_cache(context, response)

    output = {
        "input": user_input,
        "token_count": token_count,
        "compressed_prompt": compressed_prompt,
        "context": context,
        "model": model_info["name"] if not cache_hit else "cache",
        "response": response,
        "cache_hit": cache_hit,
    }

    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
