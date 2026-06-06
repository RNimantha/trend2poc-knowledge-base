import tiktoken
from typing import Optional, Dict, Any
from app.config import settings
import logging

logger = logging.getLogger(__name__)

class TokenCounter:
    """Counts tokens in text using tiktoken."""

    def __init__(self, encoding_name: str = "gpt2") -> None:
        try:
            self.encoder = tiktoken.get_encoding(encoding_name)
        except Exception as e:
            logger.error(f"Failed to get encoding '{encoding_name}': {e}")
            raise

    def count_tokens(self, text: str) -> int:
        try:
            tokens = self.encoder.encode(text)
            return len(tokens)
        except Exception as e:
            logger.error(f"Token encoding failed: {e}")
            raise


def compress_prompt(prompt: str, max_length: int = 50) -> str:
    """
    Compress prompt heuristically by truncating and removing filler words.
    This is a simplistic heuristic compression.
    """
    filler_words = {"please", "kindly", "could you", "would you", "the", "a", "an"}
    words = prompt.split()
    filtered_words = [w for w in words if w.lower() not in filler_words]
    compressed = " ".join(filtered_words)
    if len(compressed) > max_length:
        compressed = compressed[:max_length].rstrip()
        # Avoid cutting in middle of word
        if not compressed[-1].isspace() and len(compressed) < len(filtered_words):
            compressed = compressed.rsplit(' ', 1)[0]
    return compressed


class ContextManager:
    """Manages context window with truncation and simulated summarization."""

    def __init__(self, max_context_tokens: int = 100):
        self.max_context_tokens = max_context_tokens
        self.token_counter = TokenCounter()

    def manage_context(self, prompt: str) -> str:
        token_count = self.token_counter.count_tokens(prompt)
        if token_count <= self.max_context_tokens:
            return prompt
        # Simulate summarization by truncating and appending ellipsis
        tokens = self.token_counter.encoder.encode(prompt)
        truncated_tokens = tokens[: self.max_context_tokens - 3]  # reserve for '...'
        truncated_text = self.token_counter.encoder.decode(truncated_tokens) + "..."
        return truncated_text


class ModelRouter:
    """Routes prompts to mock models based on token count and manages model tiers."""

    def __init__(self) -> None:
        # Simulated models with token cost and quality
        self.models = [
            {"name": "mock-tier-1", "max_tokens": 50, "quality": "low"},
            {"name": "mock-tier-2", "max_tokens": 150, "quality": "medium"},
            {"name": "mock-tier-3", "max_tokens": 300, "quality": "high"},
        ]

    def route(self, prompt: str, token_count: int) -> Dict[str, Any]:
        # Select model based on token count
        for model in self.models:
            if token_count <= model["max_tokens"]:
                return model
        return self.models[-1]  # fallback to highest tier

    def generate_response(self, prompt: str, model_name: str) -> str:
        # Mock response generation
        return f"[Mock {model_name.capitalize()} Model Response] {prompt}"


class SemanticCache:
    """Simple semantic cache using token set overlap similarity."""

    def __init__(self, similarity_threshold: float = 0.5) -> None:
        self.cache: Dict[str, str] = {}
        self.token_counter = TokenCounter()
        self.similarity_threshold = similarity_threshold

    def _token_set(self, text: str) -> set[int]:
        try:
            tokens = self.token_counter.encoder.encode(text)
            return set(tokens)
        except Exception as e:
            logger.error(f"Failed to encode text for caching: {e}")
            return set()

    def _similarity(self, set1: set[int], set2: set[int]) -> float:
        if not set1 or not set2:
            return 0.0
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        return len(intersection) / len(union)

    def get_cached_response(self, prompt: str) -> Optional[str]:
        prompt_tokens = self._token_set(prompt)
        for cached_prompt, response in self.cache.items():
            cached_tokens = self._token_set(cached_prompt)
            similarity = self._similarity(prompt_tokens, cached_tokens)
            if similarity >= self.similarity_threshold:
                return response
        return None

    def add_to_cache(self, prompt: str, response: str) -> None:
        self.cache[prompt] = response
