# rag_pipeline/pipeline.py
# End-to-end RAG pipeline orchestration.

from common.logger import get_logger
from rag_pipeline.ingest import ingest_directory
from rag_pipeline.embeddings import embed_documents, embed_query
from rag_pipeline.retriever import index_documents, hybrid_search, rerank
from rag_pipeline.generator import generate_answer

logger = get_logger(__name__)


class RAGPipeline:
    """
    Production RAG Pipeline.
    Flow: Ingest → Embed (BGE-M3) → Index (Qdrant) → Hybrid Search → Rerank → Generate
    """

    def __init__(self):
        self._is_indexed = False

    def build_knowledge_base(self, directory_path: str | None = None) -> int:
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
        index_documents(chunks, embeddings)

        self._is_indexed = True
        logger.info("Knowledge base built. %d chunks indexed.", len(chunks))
        return len(chunks)

    def query(self, question: str) -> dict:
        """
        Queries the knowledge base: Embed → Hybrid Search → Rerank → Generate.
        Returns dict with 'answer', 'sources', 'context_used'.
        """
        if not self._is_indexed:
            count = self.build_knowledge_base()
            if count == 0:
                return {"answer": "No documents available.", "sources": [], "context_used": 0}

        logger.info("QUERY: %s", question[:100])

        query_embedding = embed_query(question)
        search_results = hybrid_search(query_embedding)
        reranked_results = rerank(question, search_results)
        result = generate_answer(question, reranked_results)

        logger.info("Pipeline complete. Sources: %d", len(result["sources"]))
        return result
