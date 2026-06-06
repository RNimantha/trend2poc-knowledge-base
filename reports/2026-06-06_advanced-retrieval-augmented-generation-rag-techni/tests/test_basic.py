import pytest
from unittest.mock import patch, MagicMock

from app.config import settings
from app.core import (
    chunk_document,
    embed_texts,
    build_faiss_index,
    hybrid_retrieve,
    generate_answer,
    prepare_document_chunks,
    DocumentChunk
)


def test_config_loads() -> None:
    assert settings.OPENAI_API_KEY is not None
    assert isinstance(settings.OPENAI_API_KEY, str)


@patch("openai.embeddings.create")
@patch("openai.chat.completions.create")
def test_rag_pipeline(mock_chat_create: MagicMock, mock_embed_create: MagicMock) -> None:
    # Mock embedding response
    def embed_side_effect(*args, **kwargs):
        inputs = kwargs.get("input") or args[1]
        data = [{"embedding": [0.1] * 1536} for _ in inputs]
        return {"data": data}

    mock_embed_create.side_effect = embed_side_effect

    # Mock generation response
    mock_chat_create.return_value = MagicMock(choices=[MagicMock(message=MagicMock(content="This is a mocked answer."))])

    # Sample documents
    documents = [
        {"id": 1, "text": "Renewable energy is energy from natural sources that are constantly replenished."},
        {"id": 2, "text": "Solar and wind power are common types of renewable energy."}
    ]

    chunks = prepare_document_chunks(documents)
    assert len(chunks) > 0

    texts = [chunk.text for chunk in chunks]
    embeddings = embed_texts(texts)
    assert len(embeddings) == len(texts)

    index = build_faiss_index(embeddings)

    query = "What is renewable energy?"
    retrieved = hybrid_retrieve(query, chunks, index, embeddings, top_k=2)
    assert len(retrieved) > 0

    answer = generate_answer(query, [chunk for chunk, _ in retrieved])
    assert isinstance(answer, str)
    assert "mocked answer" in answer.lower()
