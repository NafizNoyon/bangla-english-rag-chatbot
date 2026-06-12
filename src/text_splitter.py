from typing import List, Dict
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_documents_from_pages(extracted_pages: List[Dict]) -> List[Document]:
    """
    Convert extracted page-level text into LangChain Document objects.
    Each document keeps source and page number as metadata.
    """
    documents = []

    for page in extracted_pages:
        document = Document(
            page_content=page["text"],
            metadata={
                "source": page["source"],
                "page_number": page["page_number"],
                "character_count": page["character_count"],
            }
        )
        documents.append(document)

    return documents


def split_documents_into_chunks(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Document]:
    """
    Split documents into smaller chunks for embedding and vector search.
    Bangla punctuation separators are included for better multilingual chunking.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            "।",
            ".",
            "?",
            "!",
            " ",
            ""
        ],
        length_function=len,
    )

    chunks = text_splitter.split_documents(documents)

    for chunk_index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = chunk_index + 1
        chunk.metadata["chunk_characters"] = len(chunk.page_content)

    return chunks


def get_chunk_summary(chunks: List[Document]) -> Dict:
    """
    Generate summary statistics for created chunks.
    """
    if not chunks:
        return {
            "total_chunks": 0,
            "average_chunk_characters": 0,
            "largest_chunk": 0,
            "smallest_chunk": 0,
        }

    chunk_lengths = [len(chunk.page_content) for chunk in chunks]

    return {
        "total_chunks": len(chunks),
        "average_chunk_characters": round(sum(chunk_lengths) / len(chunk_lengths), 2),
        "largest_chunk": max(chunk_lengths),
        "smallest_chunk": min(chunk_lengths),
    }