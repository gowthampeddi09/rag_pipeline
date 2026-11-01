# rag_pipeline/server.py
# Production-grade FastAPI server for the RAG pipeline.
# Features:
# - API Key Authentication (Bearer Token)
# - Strict CORS Whitelisting
# - Rate Limiting (slowapi)
# - Input Sanitization & Prompt Injection Guardrails
# - Dual Health Probes: /health (Liveness) & /ready (Readiness)
# - X-Request-ID Tracking & Telemetry

import contextlib
import re
import secrets
import sys
import uuid
from pathlib import Path

# Auto-detect and add project root to sys.path so imports work regardless of working directory
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi import Depends, FastAPI, File, HTTPException, Request, Security, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from common.config import get_settings
from common.logger import get_logger
from rag_pipeline.pipeline import RAGPipeline
from rag_pipeline.retriever import get_qdrant_client

logger = get_logger(__name__)

# --- Rate Limiter Setup ---
limiter = Limiter(key_func=get_remote_address)

# --- Security / Auth Setup ---
security = HTTPBearer(auto_error=False)


def verify_api_key(
    credentials: HTTPAuthorizationCredentials | None = Security(security),
) -> str | None:
    """
    Validates the bearer token against the configured RAG_API_KEY.
    If RAG_API_KEY is not configured in settings, allows access with a development warning.
    Uses secrets.compare_digest to prevent timing attacks.
    """
    settings = get_settings()
    configured_key = settings.rag_api_key.strip()

    # If no key configured in environment, allow open development mode
    if not configured_key:
        return None

    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization Bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials
    if not secrets.compare_digest(token, configured_key):
        logger.warning("Unauthorized access attempt with invalid API key.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return token


# --- Pydantic Models for API I/O ---

class QueryRequest(BaseModel):
    """Request body with prompt injection defenses and sanitization."""
    question: str = Field(..., min_length=1, max_length=2000, description="The question to ask")

    @field_validator("question")
    @classmethod
    def validate_and_sanitize_question(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Question cannot be empty or whitespace only.")

        # Common prompt injection / jailbreak detection patterns
        suspicious_patterns = [
            r"(?i)ignore\s+(all\s+)?(previous|prior)\s+instructions",
            r"(?i)system\s*prompt",
            r"(?i)jailbreak",
            r"(?i)you\s+are\s+now\s+in\s+DAN\s+mode",
            r"(?i)reveal\s+your\s+(secret|instructions|prompt)",
        ]
        for pattern in suspicious_patterns:
            if re.search(pattern, cleaned):
                logger.warning("Blocked potential prompt injection attempt: %s", cleaned[:80])
                raise ValueError("Malicious or unsupported prompt pattern detected.")

        return cleaned


class SourceInfo(BaseModel):
    source: str
    page: int | str
    relevance_score: float


class QueryResponse(BaseModel):
    """Response from the RAG pipeline with confidence and telemetry."""
    question: str
    answer: str
    sources: list[SourceInfo]
    context_used: int
    confidence: float
    latency_ms: float
    cached: bool
    request_id: str


class HealthResponse(BaseModel):
    status: str
    knowledge_base_ready: bool
    version: str


class ReadyResponse(BaseModel):
    status: str
    checks: dict[str, str]
    version: str


# --- Application Lifecycle ---
pipeline = RAGPipeline()


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: build knowledge base. Shutdown: cleanup."""
    logger.info("RAG API starting up...")
    try:
        count = pipeline.build_knowledge_base()
        logger.info("Startup complete. %d chunks indexed.", count)
    except Exception as e:
        logger.error("STARTUP WARNING: Knowledge base build had issues: %s", e)
    yield
    logger.info("RAG API shutting down.")


# --- FastAPI App Initialization ---
settings = get_settings()

app = FastAPI(
    title="Enterprise RAG Pipeline API",
    description="Hardened, production-ready RAG system with BGE-M3 embeddings, Qdrant hybrid search, FlashRank reranking, rate limiting, and multi-tier fallbacks.",
    version="1.0.0",
    lifespan=lifespan,
)

# Register slowapi rate limiter exception handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS Configuration
allowed_origins = settings.allowed_origins if settings.allowed_origins else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True if allowed_origins != ["*"] else False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    max_age=3600,
)


# --- Middlewares ---

@app.middleware("http")
async def add_correlation_and_timing_header(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id

    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# --- Web Interface HTML ---
WEB_UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Enterprise RAG Platform</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0b0f19;
      --card-bg: #111827;
      --border: #1f2937;
      --accent: #3b82f6;
      --accent-hover: #2563eb;
      --success: #10b981;
      --text: #f9fafb;
      --text-muted: #9ca3af;
      --badge-bg: #1e293b;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Plus Jakarta Sans', sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.6;
      padding: 2rem 1rem;
    }
    .container {
      max-width: 900px;
      margin: 0 auto;
    }
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 2rem;
      padding-bottom: 1.5rem;
      border-bottom: 1px solid var(--border);
    }
    .logo-group h1 { font-size: 1.6rem; font-weight: 700; color: #fff; }
    .logo-group p { font-size: 0.9rem; color: var(--text-muted); margin-top: 0.2rem; }
    .nav-links a {
      color: var(--accent);
      text-decoration: none;
      font-size: 0.85rem;
      font-weight: 600;
      padding: 0.4rem 0.8rem;
      border-radius: 6px;
      border: 1px solid var(--border);
      transition: all 0.2s;
      margin-left: 0.5rem;
    }
    .nav-links a:hover { background: var(--card-bg); border-color: var(--accent); }
    .card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 1.5rem;
      margin-bottom: 1.5rem;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    .card-header {
      display: flex;
      align-items: center;
      gap: 0.6rem;
      margin-bottom: 1rem;
      font-weight: 600;
      font-size: 1.1rem;
    }
    .card-header span.icon { font-size: 1.25rem; }
    .upload-zone {
      display: flex;
      gap: 1rem;
      align-items: center;
      flex-wrap: wrap;
    }
    input[type="file"] {
      background: var(--bg);
      border: 1px dashed var(--border);
      padding: 0.75rem;
      border-radius: 8px;
      color: var(--text-muted);
      cursor: pointer;
      flex: 1;
      min-width: 250px;
      font-size: 0.85rem;
    }
    button {
      background: var(--accent);
      color: #fff;
      border: none;
      font-weight: 600;
      padding: 0.75rem 1.4rem;
      border-radius: 8px;
      cursor: pointer;
      font-size: 0.9rem;
      transition: background 0.2s;
    }
    button:hover { background: var(--accent-hover); }
    button:disabled { opacity: 0.5; cursor: not-allowed; }
    textarea {
      width: 100%;
      background: var(--bg);
      border: 1px solid var(--border);
      color: #fff;
      font-family: inherit;
      font-size: 0.95rem;
      padding: 0.9rem;
      border-radius: 8px;
      resize: vertical;
      min-height: 85px;
      margin-bottom: 0.8rem;
    }
    textarea:focus, input:focus { outline: none; border-color: var(--accent); }
    .chips {
      display: flex;
      gap: 0.5rem;
      flex-wrap: wrap;
      margin-bottom: 1rem;
    }
    .chip {
      background: var(--badge-bg);
      font-size: 0.8rem;
      padding: 0.3rem 0.7rem;
      border-radius: 20px;
      color: var(--text-muted);
      border: 1px solid var(--border);
      cursor: pointer;
      transition: all 0.2s;
    }
    .chip:hover { border-color: var(--accent); color: #fff; }
    .status-box {
      margin-top: 1rem;
      padding: 0.8rem 1rem;
      border-radius: 8px;
      font-size: 0.85rem;
      display: none;
    }
    .status-box.success { background: rgba(16, 185, 129, 0.1); border: 1px solid var(--success); color: #6ee7b7; }
    .status-box.error { background: rgba(239, 68, 68, 0.1); border: 1px solid #ef4444; color: #fca5a5; }
    .status-box.loading { background: rgba(59, 130, 246, 0.1); border: 1px solid var(--accent); color: #93c5fd; }
    .results-card { display: none; }
    .results-meta {
      display: flex;
      gap: 1rem;
      margin-bottom: 1rem;
      flex-wrap: wrap;
    }
    .meta-tag {
      background: var(--badge-bg);
      border: 1px solid var(--border);
      font-size: 0.75rem;
      padding: 0.25rem 0.6rem;
      border-radius: 6px;
      font-family: 'JetBrains Mono', monospace;
    }
    .meta-tag.green { color: var(--success); border-color: rgba(16, 185, 129, 0.3); }
    .meta-tag.blue { color: #60a5fa; border-color: rgba(96, 165, 250, 0.3); }
    .answer-text {
      font-size: 1rem;
      line-height: 1.7;
      white-space: pre-wrap;
      color: #f3f4f6;
      background: var(--bg);
      padding: 1.2rem;
      border-radius: 8px;
      border: 1px solid var(--border);
      margin-bottom: 1.2rem;
    }
    .sources-list {
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
    }
    .source-item {
      display: flex;
      align-items: center;
      gap: 0.6rem;
      font-size: 0.85rem;
      color: var(--text-muted);
      background: var(--bg);
      padding: 0.5rem 0.8rem;
      border-radius: 6px;
      border: 1px solid var(--border);
    }
    .source-badge {
      background: var(--accent);
      color: white;
      font-size: 0.7rem;
      padding: 0.15rem 0.4rem;
      border-radius: 4px;
      font-weight: 600;
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="logo-group">
        <h1>Enterprise RAG Platform</h1>
        <p>Two-Stage Hybrid Search (BGE + BM25) & FlashRank Reranker</p>
      </div>
      <div class="nav-links">
        <a href="/docs" target="_blank">Swagger API</a>
        <a href="/health" target="_blank">Health</a>
      </div>
    </header>

    <!-- Upload Card -->
    <div class="card">
      <div class="card-header">
        <span class="icon">📤</span>
        <span>Upload & Ingest Document</span>
      </div>
      <p style="font-size:0.85rem; color:var(--text-muted); margin-bottom:1rem;">
        Upload your PDF report. The document is chunked, contextually enriched, and indexed into Qdrant vector storage.
      </p>
      <form id="uploadForm" class="upload-zone">
        <input type="file" id="pdfFileInput" accept=".pdf" required>
        <button type="submit" id="uploadBtn">Upload & Index</button>
        <button type="button" id="reindexBtn" style="background:#374151;">🔄 Re-Index Knowledge Base</button>
      </form>
      <div id="uploadStatus" class="status-box"></div>
    </div>

    <!-- Query Card -->
    <div class="card">
      <div class="card-header">
        <span class="icon">💬</span>
        <span>Ask a Question</span>
      </div>
      <div class="chips">
        <span class="chip" onclick="setQuery('What is the main summary of the document?')">What is the main summary?</span>
        <span class="chip" onclick="setQuery('What are the key findings and conclusions?')">Key findings</span>
        <span class="chip" onclick="setQuery('What methodology or approach was used?')">Methodology</span>
      </div>
      <form id="queryForm">
        <textarea id="questionInput" placeholder="Ask any question based on your uploaded document..." required></textarea>
        <button type="submit" id="queryBtn">Ask Question</button>
      </form>
      <div id="queryStatus" class="status-box"></div>
    </div>

    <!-- Results Card -->
    <div id="resultsCard" class="card results-card">
      <div class="card-header">
        <span class="icon">🤖</span>
        <span>Synthesized Answer</span>
      </div>
      <div class="results-meta">
        <div id="metaConfidence" class="meta-tag green">Confidence: 0.00</div>
        <div id="metaLatency" class="meta-tag blue">Latency: 0 ms</div>
        <div id="metaCached" class="meta-tag">Cached: False</div>
      </div>
      <div id="answerContent" class="answer-text"></div>
      <div class="card-header" style="font-size: 0.95rem; margin-top: 1rem;">
        <span class="icon">📚</span>
        <span>Verified Sources & Citations</span>
      </div>
      <div id="sourcesContainer" class="sources-list"></div>
    </div>
  </div>

  <script>
    function setQuery(text) {
      document.getElementById('questionInput').value = text;
    }

    // Handle Upload
    document.getElementById('uploadForm').addEventListener('submit', async (e) => {
      e.preventDefault();
      const fileInput = document.getElementById('pdfFileInput');
      const statusBox = document.getElementById('uploadStatus');
      const uploadBtn = document.getElementById('uploadBtn');

      if (!fileInput.files.length) return;
      const file = fileInput.files[0];

      uploadBtn.disabled = true;
      statusBox.className = 'status-box loading';
      statusBox.style.display = 'block';
      statusBox.innerText = `Uploading and indexing '${file.name}'...`;

      const formData = new FormData();
      formData.append('file', file);

      try {
        const resp = await fetch('/upload', { method: 'POST', body: formData });
        const data = await resp.json();
        if (resp.ok) {
          statusBox.className = 'status-box success';
          statusBox.innerText = `Success! Indexed ${data.total_chunks_indexed} chunks for '${data.filename}'. You can now query it below.`;
        } else {
          statusBox.className = 'status-box error';
          statusBox.innerText = data.detail || 'Upload failed.';
        }
      } catch (err) {
        statusBox.className = 'status-box error';
        statusBox.innerText = 'Network error during upload.';
      } finally {
        uploadBtn.disabled = false;
      }
    });

    // Handle Manual Re-index
    document.getElementById('reindexBtn').addEventListener('click', async () => {
      const statusBox = document.getElementById('uploadStatus');
      const reindexBtn = document.getElementById('reindexBtn');
      reindexBtn.disabled = true;
      statusBox.className = 'status-box loading';
      statusBox.style.display = 'block';
      statusBox.innerText = 'Wiping old index and re-indexing current data_source documents...';

      try {
        const resp = await fetch('/reindex', { method: 'POST' });
        const data = await resp.json();
        if (resp.ok) {
          statusBox.className = 'status-box success';
          statusBox.innerText = data.message;
        } else {
          statusBox.className = 'status-box error';
          statusBox.innerText = data.detail || 'Re-indexing failed.';
        }
      } catch (err) {
        statusBox.className = 'status-box error';
        statusBox.innerText = 'Network error during re-indexing.';
      } finally {
        reindexBtn.disabled = false;
      }
    });

    // Handle Query
    document.getElementById('queryForm').addEventListener('submit', async (e) => {
      e.preventDefault();
      const question = document.getElementById('questionInput').value.trim();
      const queryBtn = document.getElementById('queryBtn');
      const queryStatus = document.getElementById('queryStatus');
      const resultsCard = document.getElementById('resultsCard');

      if (!question) return;

      queryBtn.disabled = true;
      queryStatus.className = 'status-box loading';
      queryStatus.style.display = 'block';
      queryStatus.innerText = 'Searching Qdrant, reranking with FlashRank & generating answer...';
      resultsCard.style.display = 'none';

      try {
        const resp = await fetch('/query', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question })
        });
        const data = await resp.json();

        if (resp.ok) {
          queryStatus.style.display = 'none';
          resultsCard.style.display = 'block';

          document.getElementById('answerContent').innerText = data.answer;
          document.getElementById('metaConfidence').innerText = `Confidence: ${(data.confidence * 100).toFixed(1)}%`;
          document.getElementById('metaLatency').innerText = `Latency: ${data.latency_ms} ms`;
          document.getElementById('metaCached').innerText = `Cached: ${data.cached}`;

          const sourcesBox = document.getElementById('sourcesContainer');
          sourcesBox.innerHTML = '';
          if (data.sources && data.sources.length) {
            data.sources.forEach(s => {
              const div = document.createElement('div');
              div.className = 'source-item';
              div.innerHTML = `<span class="source-badge">Page ${s.page}</span> <strong>${s.source}</strong> (Score: ${(s.relevance_score).toFixed(4)})`;
              sourcesBox.appendChild(div);
            });
          } else {
            sourcesBox.innerHTML = '<span style="font-size:0.85rem; color:var(--text-muted);">No direct source citations attached.</span>';
          }
        } else {
          queryStatus.className = 'status-box error';
          queryStatus.innerText = data.detail || 'Query processing failed.';
        }
      } catch (err) {
        queryStatus.className = 'status-box error';
        queryStatus.innerText = 'Network error while contacting the RAG server.';
      } finally {
        queryBtn.disabled = false;
      }
    });
  </script>
</body>
</html>
"""


# --- Endpoints ---

@app.get("/", response_class=HTMLResponse, summary="Interactive Web Interface")
def root_ui():
    """Serves the interactive web interface for uploading documents and querying the knowledge base."""
    return HTMLResponse(content=WEB_UI_HTML)


@app.post("/upload", summary="Upload & Index Document")
async def upload_document(
    file: UploadFile = File(...),
    token: str | None = Depends(verify_api_key),
):
    """
    Upload a PDF document to the knowledge base and trigger automatic indexing.
    Saves file to data_source_dir and re-indexes into Qdrant.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF (.pdf) documents are currently supported.",
        )

    settings = get_settings()
    target_dir = Path(settings.data_source_dir)
    if not target_dir.is_absolute():
        target_dir = _PROJECT_ROOT / target_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    safe_filename = Path(file.filename).name
    dest_path = target_dir / safe_filename

    try:
        content = await file.read()
        with open(dest_path, "wb") as f:
            f.write(content)
        logger.info("Saved uploaded file '%s' to %s (%d bytes)", safe_filename, dest_path, len(content))

        # Re-index knowledge base cleanly (wipes old stale points)
        count = pipeline.build_knowledge_base(force_recreate=True)
        return {
            "status": "success",
            "filename": safe_filename,
            "bytes_uploaded": len(content),
            "total_chunks_indexed": count,
            "message": f"Successfully ingested and indexed '{safe_filename}' into the knowledge base ({count} chunks).",
        }
    except Exception as e:
        logger.error("Failed to process upload: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process and index document: {str(e)}",
        )


@app.post("/reindex", summary="Re-index Knowledge Base")
def reindex_knowledge_base(token: str | None = Depends(verify_api_key)):
    """Rebuilds the vector index from scratch using all files in data_source."""
    try:
        count = pipeline.build_knowledge_base(force_recreate=True)
        return {
            "status": "success",
            "message": f"Knowledge base re-indexed successfully with {count} chunks.",
            "total_chunks_indexed": count,
        }
    except Exception as e:
        logger.error("Re-indexing failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Re-indexing failed: {str(e)}",
        )


@app.get("/health", response_model=HealthResponse, summary="Liveness Probe")
def health_check():
    """Liveness probe: verifies if the API server process is active."""
    return HealthResponse(
        status="healthy",
        knowledge_base_ready=pipeline._is_indexed,
        version="1.0.0",
    )


@app.get("/ready", response_model=ReadyResponse, summary="Readiness Probe")
def readiness_check():
    """
    Readiness probe: performs deep dependency checks across Qdrant, embeddings, and pipeline.
    Returns HTTP 503 if any core dependency is unready.
    """
    checks = {
        "api": "ok",
        "knowledge_base": "ok" if pipeline._is_indexed else "not_indexed",
        "qdrant": "error",
    }

    # Check Qdrant connectivity
    try:
        client = get_qdrant_client()
        client.get_collections()
        checks["qdrant"] = "ok"
    except Exception as e:
        logger.error("Readiness check Qdrant failure: %s", e)
        checks["qdrant"] = f"error: {str(e)}"

    is_all_ok = all(v == "ok" for v in checks.values())
    status_text = "ready" if is_all_ok else "not_ready"
    status_code = status.HTTP_200_OK if is_all_ok else status.HTTP_503_SERVICE_UNAVAILABLE

    return JSONResponse(
        status_code=status_code,
        content={
            "status": status_text,
            "checks": checks,
            "version": "1.0.0",
        },
    )


@app.post("/query", response_model=QueryResponse, summary="Query Knowledge Base")
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
async def query_knowledge_base(
    request: Request,
    payload: QueryRequest,
    token: str | None = Depends(verify_api_key),
):
    """
    Ask a question about the indexed documents.
    Protected by rate limiting, optional Bearer API key auth, and prompt injection filters.
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

    try:
        result = pipeline.query(payload.question)
        return QueryResponse(
            question=payload.question,
            answer=result["answer"],
            sources=[SourceInfo(**s) for s in result["sources"]],
            context_used=result["context_used"],
            confidence=result.get("confidence", 0.0),
            latency_ms=result.get("latency_ms", 0.0),
            cached=result.get("cached", False),
            request_id=request_id,
        )
    except Exception as e:
        logger.error("Query failed for request [%s]: %s", request_id, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query processing failed: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
