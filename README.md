# Bangla-English RAG Chatbot

A multilingual PDF-based Retrieval-Augmented Generation chatbot that allows users to upload Bangla or English PDF documents, ask questions, retrieve relevant document chunks using vector search, and generate source-grounded answers with page references using Gemini.

## Project Overview

This project demonstrates an end-to-end LLM application pipeline for document-based question answering. The system extracts text from uploaded PDFs, splits the document into meaningful chunks, generates multilingual embeddings, stores them in ChromaDB, retrieves relevant chunks through semantic search, and produces final answers using Gemini.

The project is designed as a portfolio-grade AI engineering project focused on RAG, multilingual NLP, vector search, and LLM-powered document intelligence.

## Key Features

- PDF upload and text extraction
- Page-wise document processing
- Text chunking with overlap
- Multilingual embeddings for Bangla and English
- ChromaDB vector storage
- Semantic similarity search
- Gemini-powered answer generation
- Source-grounded answers with page and chunk references
- Streamlit web interface
- Secure API key management using `.env`

## Tech Stack

- Python
- Streamlit
- LangChain
- ChromaDB
- Sentence Transformers
- Google Gemini
- pypdf
- python-dotenv

## RAG Pipeline

```text
PDF Upload
→ Text Extraction
→ Text Chunking
→ Multilingual Embeddings
→ ChromaDB Vector Store
→ Semantic Search
→ Gemini Answer Generation
→ Source-grounded Response