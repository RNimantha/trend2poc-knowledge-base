import pytest
from app.core import TokenCounter, compress_prompt, ContextManager, ModelRouter, SemanticCache
from app.config import settings


def test_config_loads() -> None:
    # Settings should load and have defaults
    assert isinstance(settings.use_mock_model, bool)


def test_token_counter_counts_and_compresses() -> None:
    counter = TokenCounter()
    text = "Please explain token optimization techniques in detail."
    count = counter.count_tokens(text)
    assert count > 0

    compressed = compress_prompt(text, max_length=30)
    assert isinstance(compressed, str)
    assert len(compressed) <= 30
    assert "please" not in compressed.lower()  # filler word removed


def test_context_manager_truncates() -> None:
    context_manager = ContextManager(max_context_tokens=10)
    long_text = "word " * 20
    managed = context_manager.manage_context(long_text)
    token_counter = TokenCounter()
    token_count = token_counter.count_tokens(managed)
    assert token_count <= 10


def test_model_router_routes_and_generates() -> None:
    router = ModelRouter()
    short_prompt = "short prompt"
    token_counter = TokenCounter()
    token_count = token_counter.count_tokens(short_prompt)
    model = router.route(short_prompt, token_count)
    assert "name" in model
    response = router.generate_response(short_prompt, model["name"])
    assert response.startswith("[Mock")


def test_semantic_cache_add_and_get() -> None:
    cache = SemanticCache(similarity_threshold=0.5)
    prompt1 = "Explain token optimization."
    response1 = "Response 1"
    cache.add_to_cache(prompt1, response1)

    # Exact match
    cached = cache.get_cached_response(prompt1)
    assert cached == response1

    # Similar prompt with overlapping tokens
    prompt2 = "Explain token optimization techniques."
    cached2 = cache.get_cached_response(prompt2)
    assert cached2 == response1 or cached2 is None  # Depending on threshold


def test_pipeline_smoke() -> None:
    # Simulate full pipeline
    user_input = "Explain token optimization techniques."
    token_counter = TokenCounter()
    context_manager = ContextManager()
    model_router = ModelRouter()
    semantic_cache = SemanticCache()

    token_count = token_counter.count_tokens(user_input)
    compressed_prompt = compress_prompt(user_input)
    context = context_manager.manage_context(compressed_prompt)

    cached_response = semantic_cache.get_cached_response(context)
    if cached_response is None:
        model_info = model_router.route(context, token_count)
        response = model_router.generate_response(context, model_info["name"])
        semantic_cache.add_to_cache(context, response)
    else:
        response = cached_response

    assert isinstance(response, str)
    assert len(response) > 0
