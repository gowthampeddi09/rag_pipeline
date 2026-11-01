# rag_pipeline/retriever.py
# Hybrid retrieval: Qdrant (dense + sparse search) + FlashRank reranking.
# This is the core retrieval engine of the RAG pipeline.

import sys
from pathlib import Path

# Auto-detect and add project root to sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
    PointStruct,
    SparseVector,
    Prefetch,
    FusionQuery,
    Fusion,
)

from flashrank import Ranker, RerankRequest

from common.config import get_settings
from common.logger import get_logger

logger = get_logger(__name__)

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Module-level singletons
_qdrant_client: QdrantClient | None = None
_reranker: Ranker | None = None


def get_qdrant_client() -> QdrantClient:
    """
    Returns a persistent Qdrant client singleton with automatic retry and local disk fallback.
    Ensures data persists across restarts rather than losing data in memory.
    """
    global _qdrant_client
    if _qdrant_client is not None:
        return _qdrant_client

    settings = get_settings()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=1, max=4),
        retry=retry_if_exception_type(Exception),
        reraise=False,
    )
    def _try_connect_server():
        client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port, timeout=5)
        client.get_collections()
        return client

    try:
        # Try connecting to remote/docker Qdrant server
        client = _try_connect_server()
        if client:
            logger.info("Connected to Qdrant server at %s:%d", settings.qdrant_host, settings.qdrant_port)
            _qdrant_client = client
            return _qdrant_client
    except Exception as e:
        logger.warning("Remote Qdrant server connection failed: %s", e)

    # Fallback to local persistent disk storage (persists on disk rather than losing data in memory)
    storage_path = str(settings.data_source_dir + "/../qdrant_local_storage")
    logger.info("Using local persistent Qdrant storage at '%s'", storage_path)
    _qdrant_client = QdrantClient(path=storage_path)

    return _qdrant_client


def get_reranker() -> Ranker:
    """Returns a singleton FlashRank reranker model."""
    global _reranker
    if _reranker is not None:
        return _reranker

    logger.info("Loading FlashRank reranker model...")
    _reranker = Ranker(model_name="ms-marco-MultiBERT-L-12", cache_dir="./models_cache")
    logger.info("FlashRank reranker loaded.")
    return _reranker


def ensure_collection(dense_dim: int = 1024, force_recreate: bool = False):
    """
    Creates the Qdrant collection if it doesn't exist.
    Configured for hybrid search with both dense and sparse vectors.
    """
    settings = get_settings()
    client = get_qdrant_client()
    collection_name = settings.qdrant_collection

    collections = [c.name for c in client.get_collections().collections]
    if collection_name in collections:
        if force_recreate:
            logger.info("Recreating collection '%s' to clear stale documents...", collection_name)
            client.delete_collection(collection_name=collection_name)
        else:
            logger.info("Collection '%s' already exists.", collection_name)
            return

    logger.info("Creating Qdrant collection '%s' (dense_dim=%d)...", collection_name, dense_dim)

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": VectorParams(size=dense_dim, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(index=SparseIndexParams()),
        },
    )
    logger.info("Collection '%s' created.", collection_name)


def index_documents(chunks: list[dict], embeddings: dict, force_recreate: bool = True):
    """
    Indexes document chunks into Qdrant with both dense and sparse vectors.

    Args:
        chunks: List of {'text': str, 'metadata': dict}
        embeddings: Dict with 'dense' (list[list[float]]) and 'sparse' (list[dict])
        force_recreate: If True, wipes old points so deleted documents don't persist
    """
    settings = get_settings()
    client = get_qdrant_client()
    collection_name = settings.qdrant_collection

    # Ensure collection exists with correct dimensions
    dense_dim = len(embeddings["dense"][0])
    ensure_collection(dense_dim=dense_dim, force_recreate=force_recreate)

    logger.info("Indexing %d chunks into Qdrant...", len(chunks))

    points = []
    for i, (chunk, dense_vec, sparse_vec) in enumerate(
        zip(chunks, embeddings["dense"], embeddings["sparse"])
    ):
        point = PointStruct(
            id=i,
            vector={
                "dense": dense_vec,
                "sparse": SparseVector(
                    indices=sparse_vec["indices"],
                    values=sparse_vec["values"],
                ),
            },
            payload={
                "text": chunk["text"],
                **chunk["metadata"],
            },
        )
        points.append(point)

    # Batch upsert (Qdrant handles deduplication by ID)
    BATCH_SIZE = 100
    for batch_start in range(0, len(points), BATCH_SIZE):
        batch = points[batch_start : batch_start + BATCH_SIZE]
        client.upsert(collection_name=collection_name, points=batch)

    logger.info("Indexed %d chunks successfully.", len(chunks))


def hybrid_search(query_embedding: dict, top_k: int | None = None) -> list[dict]:
    """
    Performs hybrid search (dense + sparse) with Reciprocal Rank Fusion.

    Args:
        query_embedding: Dict with 'dense' (list[float]) and 'sparse' (dict)
        top_k: Number of results to return (default from config)

    Returns:
        List of {'text': str, 'metadata': dict, 'score': float}
    """
    settings = get_settings()
    client = get_qdrant_client()
    collection_name = settings.qdrant_collection
    k = top_k or settings.retrieval_top_k

    logger.info("Performing hybrid search (top_k=%d)...", k)

    # Use Qdrant's query API with prefetch + fusion
    results = client.query_points(
        collection_name=collection_name,
        prefetch=[
            Prefetch(
                query=query_embedding["dense"],
                using="dense",
                limit=k,
            ),
            Prefetch(
                query=SparseVector(
                    indices=query_embedding["sparse"]["indices"],
                    values=query_embedding["sparse"]["values"],
                ),
                using="sparse",
                limit=k,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),  # Reciprocal Rank Fusion
        limit=k,
    )

    search_results = []
    for point in results.points:
        search_results.append({
            "text": point.payload.get("text", ""),
            "metadata": {
                "source": point.payload.get("source", ""),
                "page": point.payload.get("page", 0),
                "type": point.payload.get("type", ""),
            },
            "score": point.score,
        })

    logger.info("Hybrid search returned %d results.", len(search_results))
    return search_results


def rerank(
    query: str,
    search_results: list[dict],
    top_k: int | None = None,
    min_confidence: float | None = None,
) -> list[dict]:
    """
    Reranks search results using FlashRank cross-encoder for precision
    and drops results below the confidence threshold to eliminate hallucinations.

    Args:
        query: The user's question
        search_results: Results from hybrid_search
        top_k: Number of reranked results to return
        min_confidence: Minimum score threshold (defaults to settings.confidence_threshold)

    Returns:
        Reranked list of {'text': str, 'metadata': dict, 'score': float} above threshold
    """
    settings = get_settings()
    k = top_k or settings.rerank_top_k
    threshold = min_confidence if min_confidence is not None else settings.confidence_threshold

    if not search_results:
        return []

    ranker = get_reranker()

    # Prepare for FlashRank
    passages = [{"id": i, "text": r["text"], "meta": r["metadata"]} for i, r in enumerate(search_results)]
    rerank_request = RerankRequest(query=query, passages=passages)

    logger.info("Reranking %d candidates (threshold=%.2f)...", len(passages), threshold)
    reranked = ranker.rerank(rerank_request)

    results = []
    for item in reranked:
        score = float(item.get("score", 0.0))
        if score >= threshold:
            results.append({
                "text": item["text"],
                "metadata": item["meta"],
                "score": score,
            })
        else:
            logger.debug("Filtered out low-confidence chunk (score %.4f < threshold %.2f)", score, threshold)

        if len(results) >= k:
            break

    if results:
        logger.info("Reranking complete. Returned %d high-confidence chunks. Top score: %.4f", len(results), results[0]["score"])
    else:
        logger.warning("No chunks met confidence threshold (%.2f) for query: '%s'", threshold, query[:80])

    return results
