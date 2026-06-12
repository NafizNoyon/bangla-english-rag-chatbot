import streamlit as st

from src.pdf_loader import extract_text_from_pdf, get_pdf_summary
from src.text_splitter import (
    create_documents_from_pages,
    split_documents_into_chunks,
    get_chunk_summary
)
from src.embeddings import (
    get_embedding_model,
    generate_chunk_embeddings,
    get_embedding_summary
)
from src.vector_store import (
    create_vector_store,
    search_relevant_chunks,
    get_vector_store_summary
)
from src.rag_chain import generate_rag_answer


st.set_page_config(
    page_title="Bangla-English RAG Chatbot",
    page_icon="📚",
    layout="wide"
)


@st.cache_resource
def load_embedding_model():
    return get_embedding_model()


st.title("📚 Bangla-English RAG Chatbot")
st.subheader("PDF Document-based Question Answering System")

st.markdown(
    """
    Upload a Bangla or English PDF document. The system will extract text,
    split it into chunks, generate multilingual embeddings, store them in ChromaDB,
    retrieve relevant chunks using semantic vector search, and generate a
    source-grounded answer using Gemini.
    """
)

uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"]
)

if uploaded_file is not None:
    st.success(f"Uploaded file: {uploaded_file.name}")

    extracted_pages = extract_text_from_pdf(
        uploaded_file,
        source_name=uploaded_file.name
    )

    pdf_summary = get_pdf_summary(extracted_pages)

    st.subheader("PDF Extraction Summary")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Pages with Extracted Text", pdf_summary["pages_with_text"])

    with col2:
        st.metric("Total Characters", pdf_summary["total_characters"])

    if extracted_pages:
        documents = create_documents_from_pages(extracted_pages)

        chunks = split_documents_into_chunks(
            documents,
            chunk_size=1000,
            chunk_overlap=200
        )

        chunk_summary = get_chunk_summary(chunks)

        st.subheader("Text Chunking Summary")

        col3, col4, col5, col6 = st.columns(4)

        with col3:
            st.metric("Total Chunks", chunk_summary["total_chunks"])

        with col4:
            st.metric("Average Chunk Size", chunk_summary["average_chunk_characters"])

        with col5:
            st.metric("Largest Chunk", chunk_summary["largest_chunk"])

        with col6:
            st.metric("Smallest Chunk", chunk_summary["smallest_chunk"])

        st.subheader("Embedding Generation Summary")

        with st.spinner("Loading multilingual embedding model and generating embeddings..."):
            embedding_model = load_embedding_model()
            embeddings = generate_chunk_embeddings(chunks, embedding_model)
            embedding_summary = get_embedding_summary(embeddings)

        col7, col8 = st.columns(2)

        with col7:
            st.metric("Total Embeddings", embedding_summary["total_embeddings"])

        with col8:
            st.metric("Embedding Dimension", embedding_summary["embedding_dimension"])

        st.subheader("Vector Store Summary")

        with st.spinner("Creating ChromaDB vector store..."):
            vector_store = create_vector_store(
                chunks=chunks,
                embedding_model=embedding_model
            )

            vector_store_summary = get_vector_store_summary(chunks)

        col9, col10 = st.columns(2)

        with col9:
            st.metric("Total Vectors in ChromaDB", vector_store_summary["total_vectors"])

        with col10:
            st.metric("Vector Store Status", vector_store_summary["status"])

        st.subheader("Ask a Question from Your PDF")

        question = st.text_input(
            "Enter your question",
            placeholder="Example: What is the main methodology of this paper?"
        )

        top_k = st.slider(
            "Number of relevant chunks to retrieve",
            min_value=1,
            max_value=8,
            value=6
        )

        if question:
            with st.spinner("Searching relevant chunks from the PDF..."):
                search_results = search_relevant_chunks(
                    vector_store=vector_store,
                    query=question,
                    k=top_k
                )

            st.subheader("Generated Answer")

            with st.spinner("Generating source-grounded answer using Gemini..."):
                rag_response = generate_rag_answer(
                    question=question,
                    relevant_chunks=search_results
                )

            if rag_response.get("success"):
                st.success("Answer generated successfully.")
            else:
                st.warning("Answer generation completed with an issue.")

            st.markdown(rag_response["answer"])

            st.subheader("Answer Sources")

            if rag_response["sources"]:
                for source in rag_response["sources"]:
                    st.write(
                        f"Source: {source['source']} | "
                        f"Page: {source['page_number']} | "
                        f"Chunk: {source['chunk_id']} | "
                        f"Distance Score: {source['score']}"
                    )
            else:
                st.info("No sources available.")

            st.subheader("Retrieved Relevant Chunks")

            if search_results:
                for result in search_results:
                    expander_title = (
                        f"Rank {result['rank']} | "
                        f"Page {result['page_number']} | "
                        f"Chunk {result['chunk_id']} | "
                        f"Distance Score: {result['score']}"
                    )

                    with st.expander(expander_title, expanded=result["rank"] == 1):
                        st.write(f"**Source:** {result['source']}")
                        st.write(f"**Page:** {result['page_number']}")
                        st.write(f"**Chunk ID:** {result['chunk_id']}")
                        st.write(f"**Distance Score:** {result['score']}")
                        st.write(result["content"])
            else:
                st.warning("No relevant chunks found.")

        st.subheader("Chunk Preview")

        selected_chunk_id = st.selectbox(
            "Select a chunk to preview",
            options=[chunk.metadata["chunk_id"] for chunk in chunks]
        )

        selected_chunk = next(
            chunk for chunk in chunks
            if chunk.metadata["chunk_id"] == selected_chunk_id
        )

        st.write(
            f"Source: {selected_chunk.metadata['source']} | "
            f"Page: {selected_chunk.metadata['page_number']} | "
            f"Chunk ID: {selected_chunk.metadata['chunk_id']}"
        )

        st.text_area(
            label="Chunk Text",
            value=selected_chunk.page_content,
            height=300
        )

    else:
        st.warning(
            "No readable text found in this PDF. It may be scanned/image-based."
        )

else:
    st.info("Please upload a PDF file to begin.")