from typing import List, Dict
from uuid import uuid4

from langchain_core.documents import Document
from langchain_chroma import Chroma


def create_vector_store(
    chunks: List[Document],
    embedding_model,
    collection_name: str = "pdf_rag_collection"
) -> Chroma:
    """
    Create an in-memory Chroma vector store from document chunks.

    A unique collection name is used each time to avoid duplicate records
    during Streamlit reruns.
    """
    unique_collection_name = f"{collection_name}_{uuid4().hex[:8]}"

    ids = [
        f"{chunk.metadata.get('source', 'pdf')}_"
        f"page_{chunk.metadata.get('page_number', 'unknown')}_"
        f"chunk_{chunk.metadata.get('chunk_id', index + 1)}"
        for index, chunk in enumerate(chunks)
    ]

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name=unique_collection_name,
        ids=ids
    )

    return vector_store


def search_relevant_chunks(
    vector_store: Chroma,
    query: str,
    k: int = 4
) -> List[Dict]:
    """
    Search the vector store and return unique relevant chunks.

    Lower distance score usually means higher semantic similarity.
    Duplicate chunks are removed based on source, page number, and chunk ID.
    """
    raw_results = vector_store.similarity_search_with_score(
        query=query,
        k=max(k * 3, 10)
    )

    formatted_results = []
    seen_chunks = set()

    for document, score in raw_results:
        source = document.metadata.get("source", "Unknown")
        page_number = document.metadata.get("page_number", "Unknown")
        chunk_id = document.metadata.get("chunk_id", "Unknown")

        unique_key = (source, page_number, chunk_id)

        if unique_key in seen_chunks:
            continue

        seen_chunks.add(unique_key)

        formatted_results.append(
            {
                "rank": len(formatted_results) + 1,
                "score": round(float(score), 4),
                "content": document.page_content,
                "source": source,
                "page_number": page_number,
                "chunk_id": chunk_id,
            }
        )

        if len(formatted_results) == k:
            break

    return formatted_results


def get_vector_store_summary(chunks: List[Document]) -> Dict:
    """
    Return basic vector store summary.
    """
    return {
        "total_vectors": len(chunks),
        "status": "Ready" if chunks else "Empty"
    }