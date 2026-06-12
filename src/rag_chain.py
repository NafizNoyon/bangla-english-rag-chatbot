from typing import List, Dict
import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv(override=True)


def format_retrieved_context(relevant_chunks: List[Dict]) -> str:
    """
    Convert retrieved chunks into a clean context block for the LLM.
    """
    context_parts = []

    for chunk in relevant_chunks:
        context_parts.append(
            f"Rank: {chunk['rank']}\n"
            f"Source: {chunk['source']}\n"
            f"Page: {chunk['page_number']}\n"
            f"Chunk ID: {chunk['chunk_id']}\n"
            f"Distance Score: {chunk['score']}\n"
            f"Content:\n{chunk['content']}"
        )

    return "\n\n---\n\n".join(context_parts)


def generate_rag_answer(
    question: str,
    relevant_chunks: List[Dict],
    model_name: str = "gemini-2.5-flash-lite"
) -> Dict:
    """
    Generate a source-grounded answer using Gemini and retrieved PDF chunks.
    """
    if not relevant_chunks:
        return {
            "answer": "No relevant context was found in the uploaded PDF.",
            "sources": [],
            "success": False
        }

    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        return {
            "answer": (
                "Google API key was not found. Please add GOOGLE_API_KEY "
                "inside your .env file."
            ),
            "sources": [],
            "success": False
        }

    context = format_retrieved_context(relevant_chunks)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a professional multilingual RAG assistant.

                You must answer only from the provided PDF context.

                Answering rules:
                1. If the answer is present in the context, answer clearly and directly.
                2. If the answer requires synthesis, combine information from multiple chunks.
                3. Always mention source page references.
                4. If the user asks in Bangla, answer in Bangla.
                5. If the user asks in English, answer in English.
                6. Do not invent information outside the provided context.
                7. If the context is insufficient, say that the uploaded PDF does not provide enough information.
                8. Keep the answer concise but complete.

                Citation format:
                Use this format inside the answer:
                (Source: PDF name, Page X, Chunk Y)
                """
            ),
            (
                "human",
                """
                User Question:
                {question}

                Retrieved PDF Context:
                {context}

                Generate a source-grounded answer.
                """
            )
        ]
    )

    try:
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.2,
            google_api_key=api_key
        )

        chain = prompt | llm

        response = chain.invoke(
            {
                "question": question,
                "context": context
            }
        )

        sources = [
            {
                "source": chunk["source"],
                "page_number": chunk["page_number"],
                "chunk_id": chunk["chunk_id"],
                "score": chunk["score"]
            }
            for chunk in relevant_chunks
        ]

        return {
            "answer": response.content,
            "sources": sources,
            "success": True
        }

    except Exception as error:
        return {
            "answer": (
                "The answer could not be generated because the LLM request failed.\n\n"
                f"Technical reason: {str(error)}"
            ),
            "sources": [
                {
                    "source": chunk["source"],
                    "page_number": chunk["page_number"],
                    "chunk_id": chunk["chunk_id"],
                    "score": chunk["score"]
                }
                for chunk in relevant_chunks
            ],
            "success": False
        }