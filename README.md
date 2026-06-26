# Minimal RAG in Pure Python

A minimal implementation of Retrieval-Augmented Generation (RAG) built from scratch in Python using Ollama.

The goal of this project is to understand the core ideas behind RAG without using frameworks such as LangChain or LlamaIndex.

## Features

* Local LLM inference using Ollama
* Local embeddings using `nomic-embed-text`
* Semantic search using cosine similarity
* Basic document chunking
* Prompt augmentation
* Streaming LLM responses
* Pure Python implementation

## Project Structure

```
rag-minimal/
├── documents.py      # Reads and chunks the knowledge base
├── embedder.py       # Generates embeddings using Ollama
├── rag.py            # Complete RAG pipeline
├── search.py         # Inspect semantic search results
├── test.txt          # Example knowledge base
├── requirements.txt
└── README.md
```

## Architecture

```
Knowledge Base (test.txt)
          │
          ▼
     Chunking
          │
          ▼
     Embeddings
          │
          ▼
Document Embeddings
          │
          ▼
User Question
          │
          ▼
Question Embedding
          │
          ▼
Cosine Similarity Search
          │
          ▼
Most Relevant Chunk
          │
          ▼
Prompt Construction
          │
          ▼
Qwen (Ollama)
          │
          ▼
Generated Answer
```

## Requirements

* Python 3.x
* Ollama
* Qwen 2.5 3B
* nomic-embed-text

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Pull the required models:

```bash
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
```

Start the Ollama server before running the project.

## Run

Run the RAG pipeline:

```bash
python rag.py
```

Inspect semantic search results:

```bash
python search.py
```

## Current Limitations

* Uses linear search over embeddings
* Retrieves only the single best chunk
* Embeddings are regenerated every run
* Uses simple delimiter-based chunking
* No vector database

## Future Improvements

* Persistent embeddings
* Top-K retrieval
* Better chunking strategies
* Metadata support
* Vector database integration (FAISS/Chroma)
* PDF document ingestion

## Purpose

This project is intended as an educational implementation to understand how Retrieval-Augmented Generation works under the hood before using higher-level frameworks.
