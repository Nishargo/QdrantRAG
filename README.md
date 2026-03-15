# RAG Hybrid Search Engine

A semantic search engine built with C#, Ollama, and Qdrant Cloud.

## What it does
- Loads any PDF and splits it into chunks
- Converts chunks to vectors using Ollama (nomic-embed-text)
- Stores vectors in Qdrant Cloud vector database
- Searches using Hybrid Search - Vector similarity + BM25 keyword search combined
- Results ranked with Reciprocal Rank Fusion (RRF)

## Tech Stack
- **Language:** C# / .NET 8
- **Embeddings:** Ollama (nomic-embed-text) — runs locally, free
- **Vector Database:** Qdrant Cloud
- **PDF Parsing:** PdfPig
- **Search:** Hybrid (Vector + BM25)

## Setup

### Requirements
- .NET 8 SDK
- Ollama installed and running
- Qdrant Cloud account (free tier works)

### Installation
1. Clone the repo
   git clone https://github.com/yourusername/QdrantRAG.git
   cd QdrantRAG

2. Create .env file with your keys
   QDRANT_URL=your_qdrant_url_here
   QDRANT_API_KEY=your_qdrant_api_key_here

3. Pull the embedding model
   ollama pull nomic-embed-text

4. Run
   dotnet run

## How it works
1. PDF text is extracted page by page
2. Text is cleaned (spaces fixed) and split into 40-word overlapping chunks
3. Each chunk is embedded into a 768-dimensional vector
4. Vectors stored in Qdrant with original text as payload
5. At query time - both vector search AND BM25 run in parallel
6. Results merged with Reciprocal Rank Fusion for best accuracy

## Day 2+3 of 30-day AI Engineering sprint
Part of a 30-day plan to align with AI/LLM engineering role requirements.