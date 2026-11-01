# rag_pipeline/pipeline.py
# End-to-end RAG pipeline orchestration with query caching, confidence evaluation, and telemetry.

import sys
import time
from pathlib import Path

# Auto-detect and add project root to sys.path so imports work regardless of working directory
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from common.logger import get_logger
from rag_pipeline.cache import get_query_cache
from rag_pipeline.ingest import ingest_directory
from rag_pipeline.embeddings import embed_documents, embed_query
from rag_pipeline.retriever import index_documents, hybrid_search, rerank
from rag_pipeline.generator import generate_answer

logger = get_logger(__name__)


class RAGPipeline:
    """
    Production RAG Pipeline.
    Flow: Cache Check → Ingest → Embed (BGE-M3) → Index (Qdrant) → Hybrid Search → Rerank (FlashRank) → Generate
    """

    def __init__(self):
        self._is_indexed = False
        self.cache = get_query_cache()

    def build_knowledge_base(self, directory_path: str | None = None, force_recreate: bool = True) -> int:
        """Ingests documents, generates embeddings, and indexes into Qdrant."""
        logger.info("=" * 60)
        logger.info("BUILDING KNOWLEDGE BASE")
        logger.info("=" * 60)

        chunks = ingest_directory(directory_path)
        if not chunks:
            logger.warning("No documents found to ingest.")
            return 0

        texts = [c["text"] for c in chunks]
        embeddings = embed_documents(texts)
        index_documents(chunks, embeddings, force_recreate=force_recreate)

        self._is_indexed = True
        # Invalidate old cache when new documents are indexed
        self.cache.clear()

        logger.info("Knowledge base built. %d chunks indexed.", len(chunks))
        return len(chunks)

    def query(self, question: str) -> dict:
        """
        Queries the knowledge base: Cache Check → Embed → Hybrid Search → Rerank → Generate.
        Returns dict with 'answer', 'sources', 'context_used', 'confidence', 'latency_ms', 'cached'.
        """
        start_time = time.time()

        # 1. Check Cache Layer first
        cached_result = self.cache.get(question)
        if cached_result is not None:
            cached_result["latency_ms"] = round((time.time() - start_time) * 1000, 2)
            return cached_result

        if not self._is_indexed:
            count = self.build_knowledge_base()
            if count == 0:
                return {
                    "answer": "No documents available.",
                    "sources": [],
                    "context_used": 0,
                    "confidence": 0.0,
                    "latency_ms": round((time.time() - start_time) * 1000, 2),
                    "cached": False,
                }

        logger.info("QUERY: %s", question[:100])

        # 2. Embedding + Retrieval + Reranking
        query_embedding = embed_query(question)
        search_results = hybrid_search(query_embedding)
        reranked_results = rerank(question, search_results)

        # 3. LLM Generation
        result = generate_answer(question, reranked_results)
        result["latency_ms"] = round((time.time() - start_time) * 1000, 2)
        result["cached"] = False

        # 4. Populate Cache
        self.cache.set(question, result)

        logger.info(
            "Pipeline complete in %.2f ms. Sources: %d, Confidence: %.4f",
            result["latency_ms"],
            len(result["sources"]),
            result.get("confidence", 0.0),
        )
        return result


if __name__ == "__main__":
    pipeline = RAGPipeline()
    print("\n--- Indexing Knowledge Base ---")
    indexed_count = pipeline.build_knowledge_base()
    print(f"Indexed {indexed_count} chunks.")

    sample_query = "What is the key summary of the document?"
    print(f"\n--- Testing Query: '{sample_query}' ---")
    res = pipeline.query(sample_query)
    print("\n[Answer]:\n", res["answer"])
    print("\n[Confidence]:", res.get("confidence"))
    print("[Latency]:", res.get("latency_ms"), "ms")
    print("[Sources]:", len(res.get("sources", [])))

