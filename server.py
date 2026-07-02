# rag_pipeline/server.py
# FastAPI server for the RAG pipeline — production-ready with CORS, health checks, JSON I/O.

import contextlib

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from common.config import get_settings
from common.logger import get_logger
from rag_pipeline.pipeline import RAGPipeline

logger = get_logger(__name__)

# --- Pydantic Models for API I/O ---

class QueryRequest(BaseModel):
    """Request body for querying the knowledge base."""
    question: str = Field(..., min_length=1, max_length=2000, description="The question to ask")

class SourceInfo(BaseModel):
    source: str
    page: int | str
    relevance_score: float

class QueryResponse(BaseModel):
    """Response from the RAG pipeline."""
    question: str
    answer: str
    sources: list[SourceInfo]
    context_used: int

class HealthResponse(BaseModel):
    status: str
    knowledge_base_ready: bool
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
        logger.error("STARTUP FAILED: %s", e)
    yield
    logger.info("RAG API shutting down.")


# --- FastAPI App ---
app = FastAPI(
    title="RAG Pipeline API",
    description="Production RAG system with BGE-M3 embeddings, Qdrant hybrid search, and FlashRank reranking.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow frontend apps to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, summary="Health Check")
def health_check():
    """Check if the API and knowledge base are operational."""
    return HealthResponse(
        status="healthy",
        knowledge_base_ready=pipeline._is_indexed,
        version="1.0.0",
    )


@app.post("/query", response_model=QueryResponse, summary="Query Knowledge Base")
async def query_knowledge_base(request: QueryRequest):
    """
    Ask a question about the indexed documents.
    Uses hybrid retrieval (dense + sparse) with reranking for maximum accuracy.
    """
    if not pipeline._is_indexed:
        raise HTTPException(status_code=503, detail="Knowledge base not ready. Check /health.")

    try:
        result = pipeline.query(request.question)
        return QueryResponse(
            question=request.question,
            answer=result["answer"],
            sources=[SourceInfo(**s) for s in result["sources"]],
            context_used=result["context_used"],
        )
    except Exception as e:
        logger.error("Query failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
