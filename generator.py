# rag_pipeline/generator.py
# Answer generation using LLM with proper prompt templates, token budgeting, and citations.

import sys
from pathlib import Path
import tiktoken

# Auto-detect and add project root to sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from common.config import get_settings
from common.models import get_chat_model
from common.logger import get_logger

logger = get_logger(__name__)

# System prompt for RAG generation — production-grade instruction set
RAG_SYSTEM_PROMPT = """You are a precise and helpful AI assistant. Your task is to answer the user's question based on the provided context.

RULES:
1. Answer accurately using the provided context. If asked for a summary or overview, synthesize the key topics, financial figures, sections, and findings present in the retrieved passages.
2. Be concise but thorough. Provide specific details, figures, names, and facts when available.
3. Cite your sources by referencing the document name and page number when available (e.g. "[Source: document.pdf | Page: X]").
4. Do NOT hallucinate or assume facts not supported by the context.
5. Only if the context is completely unrelated or empty, state: "I don't have enough information to answer this question based on the available documents."
6. Structure your answer clearly with paragraphs or bullet points when appropriate."""

RAG_USER_PROMPT = """CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""


def count_tokens(text: str) -> int:
    """Estimates token count using tiktoken."""
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        # Fallback estimation: ~4 chars per token
        return len(text) // 4


def enforce_context_token_budget(chunks: list[dict], max_tokens: int | None = None) -> list[dict]:
    """Ensures that the total context passed to the LLM stays within token budget."""
    settings = get_settings()
    budget = max_tokens or settings.max_context_tokens

    budgeted_chunks = []
    total_tokens = 0

    for chunk in chunks:
        tokens = count_tokens(chunk.get("text", ""))
        if total_tokens + tokens > budget:
            logger.warning(
                "Context token budget reached (%d tokens). Truncating remaining chunks.",
                total_tokens,
            )
            break
        budgeted_chunks.append(chunk)
        total_tokens += tokens

    return budgeted_chunks


def build_rag_chain():
    """
    Builds a LangChain RAG generation chain.
    Uses ChatPromptTemplate with system/human message separation.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM_PROMPT),
        ("human", RAG_USER_PROMPT),
    ])

    llm = get_chat_model()
    chain = prompt | llm | StrOutputParser()

    return chain


def generate_answer(query: str, context_chunks: list[dict]) -> dict:
    """
    Generates an answer using the LLM based on retrieved and reranked context.

    Args:
        query: The user's question
        context_chunks: List of {'text': str, 'metadata': dict, 'score': float}

    Returns:
        Dict with 'answer', 'sources', 'context_used', and 'confidence'
    """
    logger.info("Generating answer for query: '%s'", query[:100])

    if not context_chunks:
        return {
            "answer": "No relevant documents with sufficient confidence were found to answer this question.",
            "sources": [],
            "context_used": 0,
            "confidence": 0.0,
        }

    # Enforce context token budget
    budgeted_chunks = enforce_context_token_budget(context_chunks)

    # Format context with source attribution
    context_parts = []
    sources = []
    scores = []

    for chunk in budgeted_chunks:
        context_parts.append(chunk["text"])
        score = round(chunk.get("score", 0.0), 4)
        scores.append(score)
        source_info = {
            "source": chunk["metadata"].get("source", "Unknown"),
            "page": chunk["metadata"].get("page", "N/A"),
            "relevance_score": score,
        }
        if source_info not in sources:
            sources.append(source_info)

    context_text = "\n\n---\n\n".join(context_parts)
    top_confidence = max(scores) if scores else 0.0

    # Run the chain
    chain = build_rag_chain()
    try:
        answer = chain.invoke({
            "context": context_text,
            "question": query,
        })
        logger.info("Answer generated successfully (%d chars, confidence=%.4f).", len(answer), top_confidence)
    except Exception as e:
        logger.error("LLM generation failed: %s", e)
        answer = "An error occurred while generating the answer. Please try again."

    return {
        "answer": answer,
        "sources": sources,
        "context_used": len(budgeted_chunks),
        "confidence": top_confidence,
    }
