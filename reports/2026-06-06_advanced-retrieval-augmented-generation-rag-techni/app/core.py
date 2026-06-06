import logging
import time
from typing import Any, Dict, List, Tuple

import faiss
import numpy as np
import openai
from sentence_transformers import SentenceTransformer

from app.config import settings


logger = logging.getLogger(__name__)


class HyPERetrieval:
    """Class implementing HyPE-based retrieval."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        openai_model: str = "gpt-3.5-turbo",
        hypothetical_questions_per_chunk: int = 3,
    ) -> None:
        self.embedding_model = SentenceTransformer(model_name)
        self.openai_model = openai_model
        self.hypothetical_questions_per_chunk = hypothetical_questions_per_chunk
        self.index = None  # FAISS index
        self.chunk_id_to_text: Dict[int, str] = {}
        self.hyp_qid_to_chunk_id: Dict[int, int] = {}
        self.hyp_q_embeddings: np.ndarray | None = None

        openai.api_key = settings.openai_api_key

    def chunk_documents(self, documents: List[str], chunk_size: int = 200) -> List[str]:
        """Splits documents into chunks of approximately chunk_size tokens (approximate by words)."""
        chunks = []
        for doc in documents:
            words = doc.split()
            for i in range(0, len(words), chunk_size):
                chunk = " ".join(words[i : i + chunk_size])
                chunks.append(chunk)
        logger.debug(f"Chunked documents into {len(chunks)} chunks.")
        return chunks

    def generate_hypothetical_questions(self, chunk: str) -> List[str]:
        """Generate hypothetical questions for a given chunk using OpenAI chat completion."""
        prompt = (
            f"Given the following document chunk, generate {self.hypothetical_questions_per_chunk} "
            "diverse, relevant, and insightful questions that someone might ask about this content. "
            "Return the questions as a numbered list.\n\n" + chunk
        )

        try:
            response = openai.ChatCompletion.create(
                model=self.openai_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=150,
                n=1,
            )
            text = response.choices[0].message.content.strip()
            questions = self._parse_numbered_list(text)
            if not questions:
                logger.warning("No hypothetical questions generated; falling back to default question.")
                questions = ["What is this chunk about?"]
            return questions
        except openai.error.OpenAIError as e:
            logger.error(f"OpenAI API error during hypothetical question generation: {e}")
            return ["What is this chunk about?"]

    @staticmethod
    def _parse_numbered_list(text: str) -> List[str]:
        """Parse a numbered list from text into a list of strings."""
        lines = text.splitlines()
        questions = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Expect lines like '1. question'
            if line[0].isdigit() and line[1:3] == ". ":
                question = line[3:].strip()
                questions.append(question)
            else:
                # fallback: if line does not start with number, just add it
                questions.append(line)
        return questions

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts using the sentence-transformers model."""
        embeddings = self.embedding_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings

    def build_index(self, chunks: List[str]) -> None:
        """Build FAISS index over hypothetical question embeddings linked to chunks."""
        self.chunk_id_to_text = {i: chunk for i, chunk in enumerate(chunks)}

        all_hyp_questions = []
        hyp_qid_to_chunk_id = {}

        for chunk_id, chunk_text in self.chunk_id_to_text.items():
            hyp_questions = self.generate_hypothetical_questions(chunk_text)
            # Limit to configured number
            hyp_questions = hyp_questions[: self.hypothetical_questions_per_chunk]
            for hyp_q in hyp_questions:
                hyp_qid_to_chunk_id[len(all_hyp_questions)] = chunk_id
                all_hyp_questions.append(hyp_q)

        if not all_hyp_questions:
            raise RuntimeError("No hypothetical questions generated; cannot build index.")

        self.hyp_q_embeddings = self.embed_texts(all_hyp_questions)

        dim = self.hyp_q_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # Cosine similarity with normalized embeddings
        self.index.add(self.hyp_q_embeddings)

        self.hyp_qid_to_chunk_id = hyp_qid_to_chunk_id
        logger.info(f"Built FAISS index with {self.index.ntotal} hypothetical question embeddings.")

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Retrieve top_k relevant chunks for the query using HyPE embeddings."""
        if self.index is None or self.hyp_q_embeddings is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        query_embedding = self.embed_texts([query])

        distances, indices = self.index.search(query_embedding, top_k * 2)  # Search more to deduplicate

        seen_chunks = set()
        results: List[Tuple[str, float]] = []

        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            chunk_id = self.hyp_qid_to_chunk_id.get(idx)
            if chunk_id is None or chunk_id in seen_chunks:
                continue
            seen_chunks.add(chunk_id)
            chunk_text = self.chunk_id_to_text[chunk_id]
            results.append((chunk_text, float(dist)))
            if len(results) >= top_k:
                break

        return results

    def generate_response(self, query: str, retrieved_chunks: List[str]) -> str:
        """Generate a simple response grounded on retrieved chunks using OpenAI."""
        context = "\n---\n".join(retrieved_chunks)
        prompt = (
            f"You are an assistant that answers questions based on the provided context. "
            f"Use the context to answer the question as accurately as possible.\n\n"
            f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        )

        try:
            response = openai.ChatCompletion.create(
                model=self.openai_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=200,
                n=1,
            )
            answer = response.choices[0].message.content.strip()
            return answer
        except openai.error.OpenAIError as e:
            logger.error(f"OpenAI API error during generation: {e}")
            return "Sorry, I could not generate a response at this time."
