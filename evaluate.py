# rag_pipeline/evaluate.py
# RAGAS evaluation: 4 core metrics + BM25 baseline comparison.
# Build a golden dataset, evaluate retrieval AND generation separately.

import os
import sys
from pathlib import Path

# Auto-detect and add project root to sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_recall, context_precision, faithfulness, answer_relevancy
from rank_bm25 import BM25Okapi

from common.config import get_settings
from common.models import get_chat_model, get_gemini_model
from common.logger import get_logger
from rag_pipeline.pipeline import RAGPipeline
from rag_pipeline.ingest import ingest_directory
from langchain_core.messages import HumanMessage
import random
import json

logger = get_logger(__name__)

def generate_synthetic_testset(chunks: list[dict], num_questions: int = 5) -> dict:
    """
    Generates a synthetic dataset of questions and ground truth answers from the document chunks.
    """
    logger.info("Generating synthetic testset (%d questions)...", num_questions)
    
    if not chunks:
        logger.warning("No chunks available for synthetic generation.")
        return {"question": [], "ground_truth": []}
        
    sampled_chunks = random.sample(chunks, min(num_questions, len(chunks)))
    llm = get_gemini_model()
    
    questions = []
    ground_truths = []
    
    prompt_template = """
    Based on the following document context, generate one specific question that can be answered using this context, and then provide the exact answer (ground truth) based on the context.
    
    CONTEXT:
    {context}
    
    Return the result strictly as a JSON object with 'question' and 'ground_truth' keys. Do not include any other text.
    """
    
    for chunk in sampled_chunks:
        try:
            prompt = prompt_template.format(context=chunk["text"])
            response = llm.invoke([HumanMessage(content=prompt)])
            
            # Very basic JSON parsing from response
            text = response.content.strip()
            if text.startswith("```json"):
                text = text[7:-3].strip()
            elif text.startswith("```"):
                text = text[3:-3].strip()
                
            data = json.loads(text)
            if "question" in data and "ground_truth" in data:
                questions.append(data["question"])
                ground_truths.append(data["ground_truth"])
        except Exception as e:
            logger.error("Failed to generate synthetic pair: %s", e)
            
    logger.info("Generated %d synthetic Q&A pairs.", len(questions))
    return {"question": questions, "ground_truth": ground_truths}


def run_full_evaluation(pipeline: RAGPipeline) -> dict:
    """
    Runs the complete RAGAS evaluation suite.

    Metrics:
      - context_recall: Did retrieval find all relevant info?
      - context_precision: Are top-ranked chunks actually relevant?
      - faithfulness: Is the answer grounded in context? (anti-hallucination)
      - answer_relevancy: Does the answer address the question?

    Returns a dict of metric scores.
    """
    settings = get_settings()
    logger.info("=" * 60)
    logger.info("STARTING RAGAS EVALUATION")
    logger.info("=" * 60)

    eval_results = []
    
    # Ingest directly to get chunks for synthetic generation
    chunks = ingest_directory()
    golden_dataset = generate_synthetic_testset(chunks, num_questions=5)

    if not golden_dataset["question"]:
        logger.error("Failed to generate synthetic dataset. Aborting evaluation.")
        return {}

    for question, ground_truth in zip(golden_dataset["question"], golden_dataset["ground_truth"]):
        result = pipeline.query(question)
        eval_results.append({
            "question": question,
            "answer": result["answer"],
            "contexts": [s.get("source", "") + ": " + str(s.get("page", "")) for s in result["sources"]],
            "ground_truth": ground_truth,
        })

    eval_dataset = Dataset.from_list(eval_results)

    # Use Gemini as the judge LLM for RAGAS
    judge_llm = get_gemini_model()

    logger.info("Evaluating with RAGAS (4 metrics)...")
    scores = evaluate(
        dataset=eval_dataset,
        metrics=[context_recall, context_precision, faithfulness, answer_relevancy],
        llm=judge_llm,
    )

    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    for metric_name, score in scores.items():
        if isinstance(score, (int, float)):
            logger.info("  %s: %.4f", metric_name, score)

    return dict(scores)


if __name__ == "__main__":
    pipeline = RAGPipeline()
    pipeline.build_knowledge_base()
    results = run_full_evaluation(pipeline)
    print("\nFinal Scores:", results)
