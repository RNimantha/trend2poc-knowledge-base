import pytest
from unittest.mock import patch

from app.config import settings
from app.core import HyPERetrieval


@pytest.fixture
def sample_documents() -> list[str]:
    return [
        "This is a sample document chunk about HyPE.",
        "Another chunk discussing retrieval-augmented generation techniques.",
    ]


@patch("app.core.openai.ChatCompletion.create")
def test_generate_hypothetical_questions(mock_openai_create, sample_documents):
    # Mock OpenAI response
    mock_openai_create.return_value = {
        "choices": [
            {
                "message": {
                    "content": "1. What is HyPE?\n2. How does HyPE improve retrieval?\n3. What are the benefits of HyPE?"
                }
            }
        ]
    }

    retriever = HyPERetrieval(hypothetical_questions_per_chunk=3)
    questions = retriever.generate_hypothetical_questions(sample_documents[0])
    assert len(questions) == 3
    assert questions[0].startswith("What")


def test_chunk_documents(sample_documents):
    retriever = HyPERetrieval()
    chunks = retriever.chunk_documents(sample_documents, chunk_size=5)
    assert len(chunks) >= len(sample_documents)


@patch("app.core.HyPERetrieval.generate_hypothetical_questions")
@patch("app.core.HyPERetrieval.embed_texts")
def test_build_index_and_retrieve(mock_embed_texts, mock_generate_hyp_qs, sample_documents):
    # Mock hypothetical questions
    mock_generate_hyp_qs.side_effect = lambda chunk: [f"What about: {chunk[:10]}?", f"Explain: {chunk[:10]}."]
    # Mock embeddings as random normalized vectors
    import numpy as np
    def fake_embed(texts):
        arr = np.random.randn(len(texts), 384)
        arr /= np.linalg.norm(arr, axis=1, keepdims=True)
        return arr
    mock_embed_texts.side_effect = fake_embed

    retriever = HyPERetrieval(hypothetical_questions_per_chunk=2)
    chunks = retriever.chunk_documents(sample_documents, chunk_size=10)
    retriever.build_index(chunks)

    results = retriever.retrieve("HyPE retrieval", top_k=2)
    assert len(results) > 0
    for chunk_text, score in results:
        assert isinstance(chunk_text, str)
        assert isinstance(score, float)


@patch("app.core.openai.ChatCompletion.create")
def test_generate_response(mock_openai_create):
    mock_openai_create.return_value = {
        "choices": [
            {"message": {"content": "HyPE is a technique to improve retrieval."}}
        ]
    }
    retriever = HyPERetrieval()
    response = retriever.generate_response("What is HyPE?", ["Chunk about HyPE."])
    assert "HyPE" in response


def test_config_loads():
    assert settings.openai_api_key is not None
