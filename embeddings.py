# rag_pipeline/embeddings.py
# Using FastEmbed for lightning-fast, production-ready embeddings.
# Avoids heavy PyTorch/Transformers dependencies.

from functools import lru_cache
from fastembed import TextEmbedding, SparseTextEmbedding

from common.config import get_settings
from common.logger import get_logger

logger = get_logger(__name__)


@lru_cache()
def get_dense_model() -> TextEmbedding:
    """Loads the dense embedding model (e.g., BAAI/bge-large-en-v1.5) via FastEmbed."""
    # FastEmbed uses BAAI/bge-small-en-v1.5 by default which is extremely fast and high quality
    logger.info("Loading FastEmbed dense model...")
    return TextEmbedding(model_name="BAAI/bge-small-en-v1.5")


@lru_cache()
def get_sparse_model() -> SparseTextEmbedding:
    """Loads the sparse embedding model (BM25) via FastEmbed for hybrid search."""
    logger.info("Loading FastEmbed sparse model (BM25)...")
    return SparseTextEmbedding(model_name="Qdrant/bm25")


def embed_documents(texts: list[str]) -> dict:
    """
    Embeds a list of text chunks using FastEmbed.
    Returns a dict with 'dense' and 'sparse' vectors.
    """
    logger.info("Embedding %d document chunks with FastEmbed...", len(texts))

    dense_model = get_dense_model()
    sparse_model = get_sparse_model()

    # FastEmbed returns generators, so we cast to lists
    dense_vecs = list(dense_model.embed(texts))
    sparse_vecs = list(sparse_model.embed(texts))

    result = {
        "dense": [vec.tolist() for vec in dense_vecs],
        "sparse": [{"indices": list(vec.indices), "values": list(vec.values)} for vec in sparse_vecs],
    }

    logger.info("Embedding complete. Dense dim: %d", len(result["dense"][0]))
    return result


def embed_query(query: str) -> dict:
    """
    Embeds a single query.
    """
    dense_model = get_dense_model()
    sparse_model = get_sparse_model()

    dense_vecs = list(dense_model.query_embed(query))
    sparse_vecs = list(sparse_model.query_embed(query))

    return {
        "dense": dense_vecs[0].tolist(),
        "sparse": {"indices": list(sparse_vecs[0].indices), "values": list(sparse_vecs[0].values)},
    }
