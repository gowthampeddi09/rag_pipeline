# rag_pipeline/generator.py
# Answer generation using LLM with proper prompt templates and citations.

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from common.models import get_chat_model
from common.logger import get_logger

logger = get_logger(__name__)

# System prompt for RAG generation — production-grade instruction set
RAG_SYSTEM_PROMPT = """You are a precise and helpful AI assistant. Your task is to answer the user's question based ONLY on the provided context.

RULES:
1. Answer ONLY from the provided context. If the context does not contain enough information, say "I don't have enough information to answer this question based on the available documents."
2. Be concise but thorough. Provide specific details, numbers, and facts when available.
3. Cite your sources by referencing the document name and page number when available.
4. Do NOT make up information or use knowledge outside the context.
5. If the question is ambiguous, state your interpretation before answering.
6. Structure your answer clearly with paragraphs or bullet points when appropriate."""

RAG_USER_PROMPT = """CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""


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
        Dict with 'answer', 'sources', and 'context_used'
    """
    logger.info("Generating answer for query: '%s'", query[:100])

    if not context_chunks:
        return {
            "answer": "No relevant documents were found to answer this question.",
            "sources": [],
            "context_used": 0,
        }

    # Format context with source attribution
    context_parts = []
    sources = []
    for chunk in context_chunks:
        context_parts.append(chunk["text"])
        source_info = {
            "source": chunk["metadata"].get("source", "Unknown"),
            "page": chunk["metadata"].get("page", "N/A"),
            "relevance_score": round(chunk.get("score", 0.0), 4),
        }
        if source_info not in sources:
            sources.append(source_info)

    context_text = "\n\n---\n\n".join(context_parts)

    # Run the chain
    chain = build_rag_chain()
    try:
        answer = chain.invoke({
            "context": context_text,
            "question": query,
        })
        logger.info("Answer generated successfully (%d chars).", len(answer))
    except Exception as e:
        logger.error("LLM generation failed: %s", e)
        answer = "An error occurred while generating the answer. Please try again."

    return {
        "answer": answer,
        "sources": sources,
        "context_used": len(context_chunks),
    }
