# Enterprise RAG Pipeline

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-Production-green.svg)
![Qdrant](https://img.shields.io/badge/Qdrant-Hybrid_Search-orange.svg)

A production-ready Retrieval-Augmented Generation (RAG) system built with cutting-edge open-source components. This platform enables precise, hallucination-free AI question answering over custom document knowledge bases.

## 🏗️ Architecture

This pipeline is built on the following advanced technologies:
- **Dense Embeddings:** `BAAI/bge-m3` (State-of-the-art multilingual model).
- **Sparse Search:** BM25 (For exact keyword matching).
- **Vector Database:** `Qdrant` (Supporting true Hybrid Search).
- **Reranking:** `FlashRank` (Cross-encoder reranking for maximum context relevance).
- **Generation:** Integration with LLMs for accurate, context-grounded synthesis.
- **Serving:** `FastAPI` (RESTful API with health checks and CORS).

## 🚀 Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/gowthampeddi09/rag_pipeline.git
cd rag_pipeline

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the API Server

The FastAPI server automatically handles building the knowledge base from your `data_source` directory upon startup.

```bash
# Start the server
python server.py
```
The API will be available at `http://localhost:8000`. 
Check the health status at `http://localhost:8000/health`.

### Interacting with the Pipeline

You can query the RAG system directly via the REST API:

```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"question": "What is the primary function of this system?"}'
```

## 📁 Project Structure

```
rag_pipeline/
├── data_source/           # Drop your PDFs, TXTs, or markdown files here
├── embeddings.py          # BGE-M3 embedding logic
├── evaluate.py            # Evaluation suite for retrieval accuracy
├── generator.py           # LLM generation and prompt construction
├── ingest.py              # Document parsing and chunking logic
├── pipeline.py            # Core RAG orchestration (Retrieval + Generation)
├── retriever.py           # Qdrant Hybrid Search & FlashRank reranking
└── server.py              # FastAPI production server
```

## 📊 Results & Capabilities

- **Hybrid Retrieval:** By combining dense (semantic) and sparse (keyword) search, the system effectively overcomes the "lost in the middle" problem and handles highly technical acronyms.
- **Zero-Shot Reranking:** FlashRank ensures that out of 50 retrieved chunks, the absolute most relevant ones are passed to the LLM, reducing context window costs and preventing hallucinations.
