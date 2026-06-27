"""
src/model/inference.py
───────────────────────
End-to-end QA inference pipeline.
Combines the HybridRetriever with the fine-tuned BanglaBERT QA model
to answer questions about the Bangladesh Constitution.

Usage:
    from src.model.inference import ConstitutionQAPipeline
    pipeline = ConstitutionQAPipeline.load(
        model_path="models/banglabert-qa",
        corpus_path="data/processed/corpus.json",
    )
    result = pipeline.answer("বাংলাদেশের রাষ্ট্রধর্ম কী?")
    print(result)
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional
from loguru import logger
from dataclasses import dataclass, field

from transformers import AutoTokenizer, AutoModelForQuestionAnswering


@dataclass
class QAResult:
    question: str
    answer: str
    confidence: float
    source_article_number: str
    source_article_title_en: str
    source_article_title_bn: str
    source_part: str
    context: str
    answer_start: int
    answer_end: int
    top_passages: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "confidence": round(self.confidence, 4),
            "source": {
                "article_number": self.source_article_number,
                "title_en": self.source_article_title_en,
                "title_bn": self.source_article_title_bn,
                "part": self.source_part,
            },
            "context": self.context,
            "answer_span": {"start": self.answer_start, "end": self.answer_end},
            "top_passages": self.top_passages,
        }


class ConstitutionQAPipeline:
    """
    Full QA pipeline: query → retrieval → BanglaBERT → answer with citation.
    """

    MODEL_NAME = "csebuetnlp/banglabert"

    def __init__(
        self,
        model_path: str,
        corpus_path: str,
        top_k_retrieval: int = 5,
        max_seq_length: int = 512,
        doc_stride: int = 128,
        max_answer_length: int = 150,
        n_best: int = 20,
        device: Optional[str] = None,
    ):
        self.model_path = model_path
        self.corpus_path = corpus_path
        self.top_k = top_k_retrieval
        self.max_seq_length = max_seq_length
        self.doc_stride = doc_stride
        self.max_answer_length = max_answer_length
        self.n_best = n_best

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.tokenizer = None
        self.model = None
        self.retriever = None

    @classmethod
    def load(
        cls,
        model_path: str,
        corpus_path: str,
        build_index: bool = True,
        **kwargs,
    ) -> "ConstitutionQAPipeline":
        """Factory method: load model + build retrieval index."""
        pipeline = cls(model_path=model_path, corpus_path=corpus_path, **kwargs)
        pipeline._load_model()
        pipeline._load_retriever(build_index=build_index)
        return pipeline

    def _load_model(self) -> None:
        """Load BanglaBERT QA model and tokenizer."""
        model_path = Path(self.model_path)

        # Use fine-tuned model if it exists, else fall back to base BanglaBERT
        load_path = str(model_path) if model_path.exists() else self.MODEL_NAME
        logger.info(f"Loading QA model from: {load_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(load_path)
        self.model = AutoModelForQuestionAnswering.from_pretrained(load_path)
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Model loaded on {self.device}")

    def _load_retriever(self, build_index: bool = True) -> None:
        """Load hybrid retriever."""
        # Import here to avoid circular imports
        from src.retrieval.retriever import HybridRetriever
        self.retriever = HybridRetriever(self.corpus_path)
        if build_index:
            self.retriever.build_index()

    def _get_answer_from_passage(
        self, question: str, context: str
    ) -> tuple[str, float, int, int]:
        """
        Run BanglaBERT on a single (question, context) pair.
        Returns (answer_text, score, start_char, end_char).
        """
        # Tokenize with sliding window
        inputs = self.tokenizer(
            question,
            context,
            max_length=self.max_seq_length,
            truncation="only_second",
            stride=self.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
            return_tensors="pt",
        )

        offset_mapping = inputs.pop("offset_mapping")
        overflow_to_sample = inputs.pop("overflow_to_sample_mapping", None)

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        start_logits = outputs.start_logits.cpu().numpy()  # (n_windows, seq_len)
        end_logits = outputs.end_logits.cpu().numpy()

        best_score = -1e9
        best_answer = ""
        best_start = 0
        best_end = 0

        for window_idx in range(start_logits.shape[0]):
            sl = start_logits[window_idx]
            el = end_logits[window_idx]
            offsets = offset_mapping[window_idx].numpy()

            sequence_ids = inputs.get("token_type_ids")
            if sequence_ids is not None:
                seq_ids = sequence_ids[window_idx].cpu().numpy()
                context_mask = seq_ids == 1
            else:
                # For models without token_type_ids (RoBERTa-style)
                context_mask = np.ones(len(sl), dtype=bool)
                context_mask[0] = False  # CLS

            # Get top start/end positions
            start_indices = np.where(context_mask)[0]
            end_indices = np.where(context_mask)[0]

            top_starts = start_indices[np.argsort(sl[start_indices])[-self.n_best:]]
            top_ends = end_indices[np.argsort(el[end_indices])[-self.n_best:]]

            for s_idx in top_starts:
                for e_idx in top_ends:
                    if e_idx < s_idx:
                        continue
                    if e_idx - s_idx + 1 > self.max_answer_length:
                        continue
                    if offsets[s_idx] is None or offsets[e_idx] is None:
                        continue
                    score = float(sl[s_idx] + el[e_idx])
                    if score > best_score:
                        best_score = score
                        best_start = int(offsets[s_idx][0])
                        best_end = int(offsets[e_idx][1])
                        best_answer = context[best_start:best_end]

        # Sigmoid-like normalization to get pseudo-probability
        confidence = float(1 / (1 + np.exp(-best_score / 10)))

        return best_answer, confidence, best_start, best_end

    def answer(self, question: str, top_k: Optional[int] = None) -> QAResult:
        """
        Answer a question about the Bangladesh Constitution.

        Process:
            1. Retrieve top-k relevant articles
            2. Run BanglaBERT on each retrieved passage
            3. Return the best answer with article citation
        """
        if self.retriever is None or self.model is None:
            raise RuntimeError("Call ConstitutionQAPipeline.load() first")

        k = top_k or self.top_k
        passages = self.retriever.retrieve(question, top_k=k)

        if not passages:
            return QAResult(
                question=question,
                answer="প্রাসঙ্গিক তথ্য পাওয়া যায়নি।",
                confidence=0.0,
                source_article_number="",
                source_article_title_en="",
                source_article_title_bn="",
                source_part="",
                context="",
                answer_start=0,
                answer_end=0,
            )

        # Try each passage; keep the best answer by confidence
        candidates = []
        for passage in passages:
            context = passage["text"]
            if not context.strip():
                continue

            answer_text, confidence, start, end = self._get_answer_from_passage(
                question, context
            )

            if answer_text.strip():
                candidates.append({
                    "answer": answer_text,
                    "confidence": confidence,
                    "start": start,
                    "end": end,
                    "passage": passage,
                })

        if not candidates:
            # Fallback: return the first retrieved passage snippet
            passage = passages[0]
            return QAResult(
                question=question,
                answer=passage["text"][:200],
                confidence=0.0,
                source_article_number=passage["article_number"],
                source_article_title_en=passage.get("title_en", ""),
                source_article_title_bn=passage.get("title_bn", ""),
                source_part=passage.get("part", ""),
                context=passage["text"],
                answer_start=0,
                answer_end=200,
                top_passages=passages,
            )

        # Select best candidate
        best = max(candidates, key=lambda x: x["confidence"])

        return QAResult(
            question=question,
            answer=best["answer"],
            confidence=best["confidence"],
            source_article_number=best["passage"]["article_number"],
            source_article_title_en=best["passage"].get("title_en", ""),
            source_article_title_bn=best["passage"].get("title_bn", ""),
            source_part=best["passage"].get("part", ""),
            context=best["passage"]["text"],
            answer_start=best["start"],
            answer_end=best["end"],
            top_passages=[
                {
                    "article_number": p["article_number"],
                    "title_en": p.get("title_en", ""),
                    "title_bn": p.get("title_bn", ""),
                    "text": p["text"][:300],
                    "retrieval_score": p.get("retrieval_score", 0),
                }
                for p in passages
            ],
        )

    def batch_answer(self, questions: list[str]) -> list[QAResult]:
        """Answer multiple questions."""
        return [self.answer(q) for q in questions]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/banglabert-qa")
    parser.add_argument("--corpus", default="data/processed/corpus.json")
    parser.add_argument("--question", default="বাংলাদেশের রাষ্ট্রধর্ম কী?")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    pipeline = ConstitutionQAPipeline.load(
        model_path=args.model,
        corpus_path=args.corpus,
        top_k_retrieval=args.top_k,
    )

    result = pipeline.answer(args.question)
    print(f"\nQuestion: {result.question}")
    print(f"Answer  : {result.answer}")
    print(f"Source  : Article {result.source_article_number} — {result.source_article_title_en}")
    print(f"Part    : {result.source_part}")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"\nContext : {result.context[:300]}...")
