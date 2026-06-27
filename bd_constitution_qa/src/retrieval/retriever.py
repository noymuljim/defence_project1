"""
src/retrieval/retriever.py
───────────────────────────
Hybrid retrieval pipeline combining:
  1. BM25 sparse retrieval  (keyword matching)
  2. Dense semantic retrieval via FAISS + sentence-transformers

The final ranked list is a weighted combination of both scores.

Usage:
    from src.retrieval.retriever import HybridRetriever
    retriever = HybridRetriever("data/processed/corpus.json")
    retriever.build_index()
    results = retriever.retrieve("বাংলাদেশের রাষ্ট্রধর্ম কী?", top_k=5)
"""

import os
import json
import pickle
import numpy as np
from pathlib import Path
from typing import Optional
from loguru import logger

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    raise ImportError("pip install rank-bm25")

try:
    import faiss
except ImportError:
    raise ImportError("pip install faiss-cpu")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("pip install sentence-transformers")

try:
    from normalizer import normalize as bn_normalize
    HAS_NORMALIZER = True
except ImportError:
    HAS_NORMALIZER = False


# ── Bangla tokenizer (simple whitespace + punctuation split) ─────────────────
def bangla_tokenize(text: str) -> list[str]:
    """
    Simple Bangla tokenizer.
    For production, replace with bnlp_toolkit BengaliTokenizer.
    """
    import re
    text = text.strip()
    # Remove punctuation except Bangla danda (।)
    text = re.sub(r'[^\u0980-\u09FF\u09E6-\u09EFa-zA-Z0-9\s।]', ' ', text)
    tokens = text.split()
    return [t for t in tokens if len(t) > 1]


def normalize_text(text: str) -> str:
    if HAS_NORMALIZER:
        try:
            return bn_normalize(text)
        except Exception:
            pass
    return text.strip()


class BM25Retriever:
    """Sparse BM25 retriever over article passages."""

    def __init__(self, articles: list[dict]):
        self.articles = articles
        self.bm25: Optional[BM25Okapi] = None
        self._tokenized_corpus: list[list[str]] = []

    def build(self) -> None:
        logger.info("Building BM25 index...")
        self._tokenized_corpus = [
            bangla_tokenize(normalize_text(a["passage"])) for a in self.articles
        ]
        self.bm25 = BM25Okapi(self._tokenized_corpus)
        logger.info(f"BM25 index built over {len(self.articles)} passages")

    def retrieve(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """Returns list of (article_index, bm25_score) sorted by score desc."""
        if self.bm25 is None:
            raise RuntimeError("Call build() first")
        tokens = bangla_tokenize(normalize_text(query))
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_indices]

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "tokenized": self._tokenized_corpus}, f)
        logger.info(f"BM25 index saved → {path}")

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self._tokenized_corpus = data["tokenized"]
        logger.info(f"BM25 index loaded ← {path}")


class DenseRetriever:
    """Dense semantic retriever using sentence-transformers + FAISS."""

    def __init__(
        self,
        articles: list[dict],
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ):
        self.articles = articles
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None
        self.index: Optional[faiss.IndexFlatIP] = None  # Inner product (cosine after normalization)
        self.embeddings: Optional[np.ndarray] = None

    def _load_model(self) -> None:
        if self.model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)

    def build(self, batch_size: int = 32) -> None:
        self._load_model()
        logger.info(f"Encoding {len(self.articles)} passages...")
        passages = [normalize_text(a["passage"]) for a in self.articles]

        embeddings = self.model.encode(
            passages,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2-normalize for cosine similarity via IP
        )

        self.embeddings = embeddings.astype(np.float32)
        dim = self.embeddings.shape[1]

        # Build FAISS flat inner product index
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)
        logger.info(f"FAISS index built: {self.index.ntotal} vectors, dim={dim}")

    def retrieve(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        if self.index is None:
            raise RuntimeError("Call build() first")
        self._load_model()

        query_embedding = self.model.encode(
            [normalize_text(query)],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        scores, indices = self.index.search(query_embedding, top_k)
        return [(int(i), float(s)) for i, s in zip(indices[0], scores[0]) if i >= 0]

    def save(self, index_path: str, embeddings_path: str) -> None:
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, index_path)
        np.save(embeddings_path, self.embeddings)
        logger.info(f"FAISS index saved → {index_path}")

    def load(self, index_path: str, embeddings_path: str) -> None:
        self.index = faiss.read_index(index_path)
        self.embeddings = np.load(embeddings_path)
        self._load_model()
        logger.info(f"FAISS index loaded ← {index_path} ({self.index.ntotal} vectors)")


class HybridRetriever:
    """
    Hybrid retriever combining BM25 and dense retrieval with score fusion.

    Score fusion: final_score = bm25_weight * norm_bm25 + dense_weight * dense_score
    """

    def __init__(
        self,
        corpus_path: str,
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        bm25_weight: float = 0.4,
        dense_weight: float = 0.6,
        cache_dir: str = "data/index",
    ):
        with open(corpus_path, encoding="utf-8") as f:
            self.articles = json.load(f)

        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.cache_dir = Path(cache_dir)

        self.bm25 = BM25Retriever(self.articles)
        self.dense = DenseRetriever(self.articles, model_name=embedding_model)

        self._bm25_cache = str(self.cache_dir / "bm25_cache.pkl")
        self._faiss_index = str(self.cache_dir / "faiss.index")
        self._faiss_embeddings = str(self.cache_dir / "embeddings.npy")

    def build_index(self, force_rebuild: bool = False) -> None:
        """Build or load BM25 + FAISS indices."""
        # BM25
        if not force_rebuild and Path(self._bm25_cache).exists():
            self.bm25.load(self._bm25_cache)
        else:
            self.bm25.build()
            self.bm25.save(self._bm25_cache)

        # Dense
        if not force_rebuild and Path(self._faiss_index).exists():
            self.dense.load(self._faiss_index, self._faiss_embeddings)
        else:
            self.dense.build()
            self.dense.save(self._faiss_index, self._faiss_embeddings)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        bm25_pool: int = 20,
        dense_pool: int = 20,
    ) -> list[dict]:
        """
        Hybrid retrieve: get top-k most relevant articles.
        Returns list of article dicts enriched with retrieval scores.
        """
        # Step 1: get candidates from both retrievers
        bm25_results = self.bm25.retrieve(query, top_k=bm25_pool)   # [(idx, score), ...]
        dense_results = self.dense.retrieve(query, top_k=dense_pool) # [(idx, score), ...]

        # Step 2: normalize BM25 scores to [0, 1]
        bm25_scores = {idx: score for idx, score in bm25_results}
        max_bm25 = max(bm25_scores.values(), default=1.0)
        if max_bm25 == 0:
            max_bm25 = 1.0
        bm25_norm = {idx: score / max_bm25 for idx, score in bm25_scores.items()}

        # Step 3: dense scores are already cosine similarities in [0, 1] after normalization
        dense_scores = {idx: max(0.0, score) for idx, score in dense_results}

        # Step 4: fuse scores over union of candidates
        all_indices = set(bm25_norm.keys()) | set(dense_scores.keys())
        fused = {}
        for idx in all_indices:
            b = bm25_norm.get(idx, 0.0)
            d = dense_scores.get(idx, 0.0)
            fused[idx] = self.bm25_weight * b + self.dense_weight * d

        # Step 5: sort by fused score and return top-k
        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for idx, score in ranked:
            article = dict(self.articles[idx])
            article["retrieval_score"] = round(score, 4)
            article["bm25_score"] = round(bm25_norm.get(idx, 0.0), 4)
            article["dense_score"] = round(dense_scores.get(idx, 0.0), 4)
            results.append(article)

        return results

    def evaluate_recall(self, qa_pairs: list[dict], top_k: int = 5) -> dict:
        """
        Evaluate Recall@k for the retrieval layer independently.
        qa_pairs: list of {"question": ..., "article_number": ...}
        """
        hits = 0
        total = len(qa_pairs)

        for pair in qa_pairs:
            results = self.retrieve(pair["question"], top_k=top_k)
            retrieved_articles = {r["article_number"] for r in results}
            if pair["article_number"] in retrieved_articles:
                hits += 1

        recall = hits / total if total > 0 else 0
        logger.info(f"Recall@{top_k}: {recall:.3f} ({hits}/{total})")
        return {"recall_at_k": recall, "k": top_k, "hits": hits, "total": total}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="data/processed/corpus.json")
    parser.add_argument("--query", default="বাংলাদেশের রাষ্ট্রধর্ম কী?")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    retriever = HybridRetriever(args.corpus)
    retriever.build_index(force_rebuild=args.rebuild)

    results = retriever.retrieve(args.query, top_k=args.top_k)
    print(f"\nQuery: {args.query}\n{'─'*60}")
    for i, r in enumerate(results, 1):
        print(f"{i}. Article {r['article_number']} — {r['title_en']}")
        print(f"   Score: {r['retrieval_score']:.4f} (BM25: {r['bm25_score']:.4f}, Dense: {r['dense_score']:.4f})")
        print(f"   {r['text'][:120]}...")
        print()
