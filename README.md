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

### 1. Environment & Dependencies

This pipeline is part of the `AI_Agents` workspace. A pre-configured virtual environment is located at the workspace root (`AI_Agents/.venv`).

If you want to set up a new environment or install dependencies manually:

```bash
# From AI_Agents/ or rag_pipeline/ directory:
pip install -r requirements.txt
```

Ensure your `.env` file exists at the root `AI_Agents/.env` with your API keys:
```env
GOOGLE_API_KEY=your_gemini_key
GROQ_API_KEY=your_groq_key
```

---

### 2. Running the System

You can run the pipeline either via the **FastAPI REST server** or directly from the **Python CLI**.

#### Option A: Start the FastAPI Server (Recommended)

From the project root (`AI_Agents`):
```bash
.\.venv\Scripts\python -m uvicorn rag_pipeline.server:app --host 0.0.0.0 --port 8000
```

Or from inside `rag_pipeline/`:
```bash
python server.py
```

Upon startup, the server automatically scans and indexes documents in `data_source/`.

#### Option B: Run Directly from Python CLI

To test indexing and run a query directly in your terminal:
```bash
python pipeline.py
```

---

### 3. Accessing the API in Your Browser

> ⚠️ **Important:** Do not open `http://0.0.0.0:8000/` in your browser. `0.0.0.0` is a server bind address and will return `ERR_ADDRESS_INVALID`. Always use **`localhost`** or **`127.0.0.1`**.

Once the server is running, open:
* **API Overview & Status:** [http://localhost:8000/](http://localhost:8000/)
* **Interactive Swagger UI (Test Queries):** [http://localhost:8000/docs](http://localhost:8000/docs)
* **Health Check Probe:** [http://localhost:8000/health](http://localhost:8000/health)
* **Readiness Check Probe:** [http://localhost:8000/ready](http://localhost:8000/ready)

---

### 4. Querying via REST API

#### Using PowerShell:
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/query" -Method Post -ContentType "application/json" -Body '{"question": "What is the key summary of the document?"}'
```

#### Using cURL:
```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"question": "What is the key summary of the document?"}'
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
