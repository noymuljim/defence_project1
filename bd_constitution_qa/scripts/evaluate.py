"""
scripts/evaluate.py
────────────────────
Comprehensive evaluation of the QA system:
  - Retrieval Recall@k (retrieval layer independently)
  - Exact Match and F1 (QA model)
  - Latency benchmarking

Usage:
    python scripts/evaluate.py \
        --corpus data/processed/corpus.json \
        --test   data/squad/test.json \
        --model  models/banglabert-qa
"""

import sys
import json
import time
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from loguru import logger
from collections import Counter


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    import re
    import string
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation + '।')
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def compute_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def evaluate_retrieval(retriever, test_pairs: list[dict], k_values=[1, 3, 5, 10]) -> dict:
    """Evaluate retrieval recall at different k values."""
    results = {f"recall@{k}": 0 for k in k_values}
    total = len(test_pairs)

    for pair in test_pairs:
        q = pair["question"]
        true_art = pair["article_number"]
        retrieved = retriever.retrieve(q, top_k=max(k_values))
        retrieved_arts = [r["article_number"] for r in retrieved]

        for k in k_values:
            if true_art in retrieved_arts[:k]:
                results[f"recall@{k}"] += 1

    for k in k_values:
        results[f"recall@{k}"] = round(results[f"recall@{k}"] / total, 4)

    return results


def evaluate_qa_model(pipeline, test_pairs: list[dict]) -> dict:
    """Evaluate QA model Exact Match and F1."""
    exact_matches = []
    f1_scores = []
    latencies = []

    for pair in test_pairs:
        q = pair["question"]
        gold = pair["answer"]

        t0 = time.time()
        result = pipeline.answer(q)
        latency = (time.time() - t0) * 1000
        latencies.append(latency)

        pred = result.answer
        exact_matches.append(compute_exact_match(pred, gold))
        f1_scores.append(compute_f1(pred, gold))

    return {
        "exact_match": round(np.mean(exact_matches) * 100, 2),
        "f1": round(np.mean(f1_scores) * 100, 2),
        "avg_latency_ms": round(np.mean(latencies), 1),
        "p95_latency_ms": round(np.percentile(latencies, 95), 1),
        "total_examples": len(test_pairs),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="data/processed/corpus.json")
    parser.add_argument("--test", default="data/squad/test.json")
    parser.add_argument("--model", default="models/banglabert-qa")
    parser.add_argument("--retrieval-only", action="store_true")
    args = parser.parse_args()

    logger.info("=== Evaluation ===")

    # Load test data
    with open(args.test, encoding="utf-8") as f:
        test_squad = json.load(f)

    test_pairs = []
    for article in test_squad["data"]:
        title = article["title"]
        art_num = title.split("—")[0].replace("Article", "").strip()
        for para in article["paragraphs"]:
            for qa in para["qas"]:
                test_pairs.append({
                    "question": qa["question"],
                    "answer": qa["answers"][0]["text"],
                    "article_number": art_num,
                })

    logger.info(f"Test pairs: {len(test_pairs)}")

    # Evaluate retrieval
    logger.info("\n--- Retrieval Evaluation ---")
    from src.retrieval.retriever import HybridRetriever
    retriever = HybridRetriever(args.corpus)
    retriever.build_index()

    recall_results = evaluate_retrieval(retriever, test_pairs)
    for k, v in recall_results.items():
        logger.info(f"  {k}: {v:.4f} ({v*100:.1f}%)")

    if not args.retrieval_only:
        # Evaluate QA model
        logger.info("\n--- QA Model Evaluation ---")
        from src.model.inference import ConstitutionQAPipeline
        pipeline = ConstitutionQAPipeline.load(
            model_path=args.model,
            corpus_path=args.corpus,
        )
        # Use already-built retriever
        pipeline.retriever = retriever

        qa_results = evaluate_qa_model(pipeline, test_pairs[:50])  # cap at 50 for speed
        logger.info(f"  Exact Match: {qa_results['exact_match']}%")
        logger.info(f"  F1 Score   : {qa_results['f1']}%")
        logger.info(f"  Avg latency: {qa_results['avg_latency_ms']} ms")
        logger.info(f"  P95 latency: {qa_results['p95_latency_ms']} ms")

    logger.info("\n=== Evaluation complete ===")


if __name__ == "__main__":
    main()
