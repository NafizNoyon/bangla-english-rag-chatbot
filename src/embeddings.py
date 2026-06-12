from typing import List

from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class MultilingualSentenceTransformerEmbeddings(Embeddings):
    """
    Custom LangChain-compatible embedding class using SentenceTransformer.

    This model supports multilingual semantic similarity and is suitable for
    Bangla-English PDF document retrieval.
    """

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple document chunks.
        """
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a user query.
        """
        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return embedding.tolist()


def get_embedding_model() -> MultilingualSentenceTransformerEmbeddings:
    """
    Load and return the multilingual embedding model.
    """
    return MultilingualSentenceTransformerEmbeddings()


def generate_chunk_embeddings(chunks, embedding_model) -> List[List[float]]:
    """
    Generate embeddings for LangChain document chunks.
    """
    texts = [chunk.page_content for chunk in chunks]
    return embedding_model.embed_documents(texts)


def get_embedding_summary(embeddings: List[List[float]]) -> dict:
    """
    Generate summary statistics for embeddings.
    """
    if not embeddings:
        return {
            "total_embeddings": 0,
            "embedding_dimension": 0,
        }

    return {
        "total_embeddings": len(embeddings),
        "embedding_dimension": len(embeddings[0]),
    }