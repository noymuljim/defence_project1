#!/usr/bin/env python3
"""
scripts/setup.py
─────────────────
One-command setup: build sample corpus + SQuAD dataset + retrieval index.
Run this first before training.

    python scripts/setup.py
"""
import os
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from src.preprocessing.corpus_builder import build_sample_corpus
from src.preprocessing.squad_builder import SQuADBuilder, BOOTSTRAP_QA_PAIRS


def main():
    logger.info("=== Bangladesh Constitution QA — Setup ===")

    # Create directories
    for d in [
        "data/raw", "data/processed", "data/squad",
        "data/index", "models", "logs",
    ]:
        Path(d).mkdir(parents=True, exist_ok=True)

    # Step 1: Build corpus
    corpus_path = "data/processed/corpus.json"
    if not Path(corpus_path).exists():
        logger.info("Step 1/3: Building sample corpus...")
        build_sample_corpus(corpus_path)
    else:
        logger.info(f"Step 1/3: Corpus already exists at {corpus_path}")

    with open(corpus_path) as f:
        corpus = json.load(f)
    logger.info(f"  Corpus: {len(corpus)} articles")

    # Step 2: Build SQuAD dataset
    squad_dir = "data/squad"
    if not Path(f"{squad_dir}/train.json").exists():
        logger.info("Step 2/3: Building SQuAD training data...")
        builder = SQuADBuilder(corpus_path)
        squad_data = builder.build_from_pairs(BOOTSTRAP_QA_PAIRS)
        paths = builder.split_and_save(squad_data, squad_dir)
        for split, path in paths.items():
            report = builder.validate(path)
            logger.info(f"  {split}: {report['total_qa_pairs']} pairs, quality {report['quality_score']}%")
    else:
        logger.info(f"Step 2/3: SQuAD data already exists at {squad_dir}/")

    # Step 3: Build retrieval index
    logger.info("Step 3/3: Building retrieval index...")
    try:
        from src.retrieval.retriever import HybridRetriever
        retriever = HybridRetriever(corpus_path, cache_dir="data/index")
        retriever.build_index(force_rebuild=False)
        logger.info("  Retrieval index built")
    except Exception as e:
        logger.warning(f"  Retrieval index build failed (install faiss/sentence-transformers): {e}")

    logger.info("\n=== Setup complete! ===")
    logger.info("Next steps:")
    logger.info("  Train model : python -m src.model.trainer --train data/squad/train.json --dev data/squad/dev.json")
    logger.info("  Start API   : uvicorn src.api.main:app --port 8000")
    logger.info("  Start UI    : streamlit run src/ui/app.py")
    logger.info("  Colab       : Open notebooks/train_colab.ipynb in Google Colab")


if __name__ == "__main__":
    main()
