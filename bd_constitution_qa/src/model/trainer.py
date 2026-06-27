"""
src/model/trainer.py
─────────────────────
Fine-tunes BanglaBERT (csebuetnlp/banglabert) on SQuAD-style
Bangladesh Constitution QA pairs using the HuggingFace Trainer API.

Usage (Colab / local with GPU):
    python -m src.model.trainer \
        --train data/squad/train.json \
        --dev   data/squad/dev.json \
        --output models/banglabert-qa \
        --epochs 4
"""

import os
import json
import argparse
import collections
import numpy as np
from pathlib import Path
from loguru import logger

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    default_data_collator,
    EarlyStoppingCallback,
)
from datasets import Dataset, DatasetDict
import evaluate


# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_NAME = "csebuetnlp/banglabert"
MAX_SEQ_LENGTH = 512
DOC_STRIDE = 128
MAX_QUERY_LENGTH = 64
MAX_ANSWER_LENGTH = 150
N_BEST_SIZE = 20


def load_squad_json(path: str) -> list[dict]:
    """Load SQuAD v1.1 JSON and flatten into a list of examples."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    examples = []
    for article in data["data"]:
        for para in article["paragraphs"]:
            context = para["context"]
            for qa in para["qas"]:
                if qa.get("is_impossible", False):
                    continue
                answer = qa["answers"][0]
                examples.append({
                    "id": qa["id"],
                    "question": qa["question"],
                    "context": context,
                    "answers": {
                        "text": [answer["text"]],
                        "answer_start": [answer["answer_start"]],
                    },
                })

    logger.info(f"Loaded {len(examples)} examples from {path}")
    return examples


def prepare_train_features(examples, tokenizer):
    """
    Tokenize train examples and compute start/end token positions for the answer.
    Handles long documents with sliding window (doc_stride).
    """
    tokenized = tokenizer(
        examples["question"],
        examples["context"],
        max_length=MAX_SEQ_LENGTH,
        truncation="only_second",
        stride=DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized.pop("offset_mapping")

    tokenized["start_positions"] = []
    tokenized["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        sequence_ids = tokenized.sequence_ids(i)
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]

        if len(answers["answer_start"]) == 0:
            tokenized["start_positions"].append(cls_index)
            tokenized["end_positions"].append(cls_index)
        else:
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Find first context token
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            # Check if answer is within this span
            if not (offsets[token_start_index][0] <= start_char and
                    offsets[token_end_index][1] >= end_char):
                tokenized["start_positions"].append(cls_index)
                tokenized["end_positions"].append(cls_index)
            else:
                while (token_start_index < len(offsets) and
                       offsets[token_start_index][0] <= start_char):
                    token_start_index += 1
                tokenized["start_positions"].append(token_start_index - 1)

                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized["end_positions"].append(token_end_index + 1)

    return tokenized


def prepare_validation_features(examples, tokenizer):
    """Tokenize validation examples (no start/end labels needed)."""
    tokenized = tokenizer(
        examples["question"],
        examples["context"],
        max_length=MAX_SEQ_LENGTH,
        truncation="only_second",
        stride=DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    sample_mapping = tokenized["overflow_to_sample_mapping"]
    tokenized["example_id"] = []

    for i in range(len(tokenized["input_ids"])):
        sequence_ids = tokenized.sequence_ids(i)
        sample_index = sample_mapping[i]
        tokenized["example_id"].append(examples["id"][sample_index])

        # Mark non-context tokens in offset_mapping as None
        tokenized["offset_mapping"][i] = [
            (o if sequence_ids[k] == 1 else None)
            for k, o in enumerate(tokenized["offset_mapping"][i])
        ]

    return tokenized


def postprocess_qa_predictions(
    examples,
    features,
    raw_predictions,
    tokenizer,
    n_best_size: int = N_BEST_SIZE,
    max_answer_length: int = MAX_ANSWER_LENGTH,
):
    """Convert model start/end logits to human-readable answer strings."""
    all_start_logits, all_end_logits = raw_predictions

    example_id_to_index = {ex["id"]: i for i, ex in enumerate(examples)}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    predictions = {}
    for example_index, example in enumerate(examples):
        feature_indices = features_per_example[example_index]
        min_null_score = None
        valid_answers = []

        context = example["context"]

        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            offset_mapping = features[feature_index]["offset_mapping"]

            cls_index = features[feature_index]["input_ids"].index(tokenizer.cls_token_id)
            feature_null_score = start_logits[cls_index] + end_logits[cls_index]
            if min_null_score is None or min_null_score < feature_null_score:
                min_null_score = feature_null_score

            start_indexes = np.argsort(start_logits)[-1:-n_best_size-1:-1].tolist()
            end_indexes = np.argsort(end_logits)[-1:-n_best_size-1:-1].tolist()

            for start_index in start_indexes:
                for end_index in end_indexes:
                    if (start_index >= len(offset_mapping) or
                            end_index >= len(offset_mapping) or
                            offset_mapping[start_index] is None or
                            offset_mapping[end_index] is None):
                        continue
                    if end_index < start_index:
                        continue
                    if end_index - start_index + 1 > max_answer_length:
                        continue
                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    valid_answers.append({
                        "score": start_logits[start_index] + end_logits[end_index],
                        "text": context[start_char:end_char],
                    })

        if valid_answers:
            best = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
            predictions[example["id"]] = best["text"]
        else:
            predictions[example["id"]] = ""

    return predictions


class BanglaQATrainer:
    """Wrapper around HuggingFace Trainer for BanglaBERT QA fine-tuning."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        output_dir: str = "models/banglabert-qa",
        num_epochs: int = 4,
        batch_size: int = 8,
        learning_rate: float = 2e-5,
        use_wandb: bool = False,
    ):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.use_wandb = use_wandb

        self.tokenizer = None
        self.model = None

    def load_model(self) -> None:
        logger.info(f"Loading tokenizer and model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(self.model_name)
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def prepare_datasets(self, train_path: str, dev_path: str) -> tuple:
        """Load and tokenize train/dev datasets."""
        train_examples = load_squad_json(train_path)
        dev_examples = load_squad_json(dev_path)

        train_dataset = Dataset.from_list(train_examples)
        dev_dataset = Dataset.from_list(dev_examples)

        tokenizer = self.tokenizer

        # Tokenize train set
        tokenized_train = train_dataset.map(
            lambda ex: prepare_train_features(ex, tokenizer),
            batched=True,
            remove_columns=train_dataset.column_names,
            desc="Tokenizing train set",
        )

        # Tokenize validation set
        tokenized_dev = dev_dataset.map(
            lambda ex: prepare_validation_features(ex, tokenizer),
            batched=True,
            remove_columns=dev_dataset.column_names,
            desc="Tokenizing dev set",
        )

        return tokenized_train, tokenized_dev, dev_dataset

    def compute_metrics_fn(self, dev_examples, dev_features):
        """Returns a compute_metrics function for the Trainer."""
        metric = evaluate.load("squad")

        def compute_metrics(eval_preds):
            predictions = postprocess_qa_predictions(
                dev_examples,
                dev_features,
                eval_preds.predictions,
                self.tokenizer,
            )
            formatted_preds = [
                {"id": k, "prediction_text": v} for k, v in predictions.items()
            ]
            references = [
                {"id": ex["id"], "answers": ex["answers"]} for ex in dev_examples
            ]

            result = metric.compute(predictions=formatted_preds, references=references)

            # Debug visibility: confirm what the metric actually returned.
            logger.info(f"DEBUG compute_metrics raw result: {result}")

            if not result:
                logger.warning(
                    "compute_metrics: evaluate.load('squad') returned empty/None. "
                    "Falling back to manual EM/F1 computation."
                )
                result = _manual_squad_metrics(formatted_preds, references)
                logger.info(f"DEBUG manual fallback result: {result}")

            # Normalize key names so metric_for_best_model='f1' always resolves.
            normalized = {}
            for k, v in result.items():
                key = k.lower().replace("-", "_")
                normalized[key] = v
            return normalized

        return compute_metrics

    def train(self, train_path: str, dev_path: str) -> None:
        if self.model is None:
            self.load_model()

        tokenized_train, tokenized_dev, dev_examples = self.prepare_datasets(train_path, dev_path)

        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size * 2,
            learning_rate=self.learning_rate,
            warmup_ratio=0.1,
            weight_decay=0.01,
            fp16=torch.cuda.is_available(),
            gradient_accumulation_steps=2 if self.batch_size < 16 else 1,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            logging_steps=50,
            report_to="wandb" if self.use_wandb else "none",
            seed=42,
            dataloader_num_workers=2,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_dev,
            processing_class=self.tokenizer,
            data_collator=default_data_collator,
            compute_metrics=self.compute_metrics_fn(dev_examples, list(tokenized_dev)),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )

        logger.info("Starting training...")
        trainer.train()

        # Save final model
        self.output_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(self.output_dir))
        self.tokenizer.save_pretrained(str(self.output_dir))
        logger.info(f"Model saved → {self.output_dir}")

        # Final evaluation
        metrics = trainer.evaluate()
        logger.info(f"Final dev metrics: {metrics}")
        return metrics

    def evaluate_on_test(self, test_path: str) -> dict:
        """Run evaluation on the held-out test set."""
        if self.model is None:
            # Load from saved checkpoint
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.output_dir))
            self.model = AutoModelForQuestionAnswering.from_pretrained(str(self.output_dir))

        test_examples = load_squad_json(test_path)
        test_dataset = Dataset.from_list(test_examples)
        tokenizer = self.tokenizer

        tokenized_test = test_dataset.map(
            lambda ex: prepare_validation_features(ex, tokenizer),
            batched=True,
            remove_columns=test_dataset.column_names,
            desc="Tokenizing test set",
        )

        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            per_device_eval_batch_size=16,
            report_to="none",
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            processing_class=self.tokenizer,
            data_collator=default_data_collator,
        )

        raw_preds = trainer.predict(tokenized_test)
        predictions = postprocess_qa_predictions(
            test_examples,
            list(tokenized_test),
            raw_preds.predictions,
            self.tokenizer,
        )

        metric = evaluate.load("squad")
        formatted_preds = [{"id": k, "prediction_text": v} for k, v in predictions.items()]
        references = [{"id": ex["id"], "answers": ex["answers"]} for ex in test_examples]
        results = metric.compute(predictions=formatted_preds, references=references)

        if not results:
            logger.warning("evaluate_on_test: squad metric returned empty, using manual fallback.")
            results = _manual_squad_metrics(formatted_preds, references)

        logger.info(f"Test set results: EM={results['exact_match']:.2f}, F1={results['f1']:.2f}")
        return results


# ── Manual EM/F1 fallback ──────────────────────────────────────────────────────
# Used only if evaluate.load("squad") fails to return scores (e.g. due to a
# version mismatch in the `evaluate`/`datasets` packages). Implements the
# standard SQuAD v1.1 normalization + EM/F1 scoring so training never
# crashes on metric_for_best_model lookups.

import re
import string


def _normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def _f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = _normalize_answer(prediction).split()
    gt_tokens = _normalize_answer(ground_truth).split()

    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return float(pred_tokens == gt_tokens)

    common = collections.Counter(pred_tokens) & collections.Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return (2 * precision * recall) / (precision + recall)


def _exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(_normalize_answer(prediction) == _normalize_answer(ground_truth))


def _manual_squad_metrics(formatted_preds: list[dict], references: list[dict]) -> dict:
    """Compute SQuAD-style EM/F1 without relying on the `evaluate` library."""
    preds_by_id = {p["id"]: p["prediction_text"] for p in formatted_preds}

    em_total = 0.0
    f1_total = 0.0
    count = 0

    for ref in references:
        qid = ref["id"]
        gold_answers = ref["answers"]["text"]
        if not gold_answers:
            continue
        prediction = preds_by_id.get(qid, "")

        em_total += max(_exact_match_score(prediction, gt) for gt in gold_answers)
        f1_total += max(_f1_score(prediction, gt) for gt in gold_answers)
        count += 1

    if count == 0:
        return {"exact_match": 0.0, "f1": 0.0}

    return {
        "exact_match": 100.0 * em_total / count,
        "f1": 100.0 * f1_total / count,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/squad/train.json")
    parser.add_argument("--dev", default="data/squad/dev.json")
    parser.add_argument("--test", default="data/squad/test.json")
    parser.add_argument("--output", default="models/banglabert-qa")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    trainer = BanglaQATrainer(
        model_name=args.model,
        output_dir=args.output,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        use_wandb=args.wandb,
    )

    if args.eval_only:
        results = trainer.evaluate_on_test(args.test)
        print(f"EM: {results['exact_match']:.2f} | F1: {results['f1']:.2f}")
    else:
        trainer.train(args.train, args.dev)
        if Path(args.test).exists():
            trainer.evaluate_on_test(args.test)