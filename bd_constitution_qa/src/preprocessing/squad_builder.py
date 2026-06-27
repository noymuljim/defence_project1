"""
src/preprocessing/squad_builder.py
────────────────────────────────────
Converts annotated QA pairs into HuggingFace-compatible SQuAD v1.1 format.
Also includes a dataset splitter and a validation utility.

Usage:
    # Use built-in QA pairs (quickest):
    python -m src.preprocessing.squad_builder \
        --bootstrap \
        --corpus data/processed/corpus.json \
        --output-dir data/squad/

    # Use your own annotated CSV:
    python -m src.preprocessing.squad_builder \
        --annotations data/annotations/qa_pairs.csv \
        --corpus data/processed/corpus.json \
        --output-dir data/squad/
"""

import json
import uuid
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from loguru import logger

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    from sklearn.model_selection import train_test_split
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ── Bootstrap QA pairs ────────────────────────────────────────────────────────
# Format: (question, answer, article_number, answer_start)
# answer_start = -1  →  auto-locate inside the article text

BOOTSTRAP_QA_PAIRS: List[Tuple] = [
    # Part I — The Republic
    ("বাংলাদেশের রাষ্ট্রধর্ম কী?",                      "ইসলাম",                              "2A",   -1),
    ("রাষ্ট্রধর্ম কোন অনুচ্ছেদে বর্ণিত?",               "অনুচ্ছেদ ২ক",                        "2A",   -1),
    ("প্রজাতন্ত্রের রাষ্ট্রভাষা কী?",                   "বাংলা",                              "3",    -1),
    ("বাংলাদেশের সরকারি ভাষা কোনটি?",                   "বাংলা",                              "3",    -1),
    ("সংবিধানের সর্বোচ্চ মর্যাদা কোন অনুচ্ছেদে?",       "সংবিধানের প্রাধান্য",                "7",    -1),
    ("প্রজাতন্ত্রের সকল ক্ষমতার মালিক কে?",             "জনগণ",                               "7",    -1),
    # Part II — Fundamental Principles
    ("রাষ্ট্রের মৌলিক নীতি কোন ভাগে আছে?",              "দ্বিতীয় ভাগ",                        "8",    -1),
    ("জাতীয়তাবাদ কোন অনুচ্ছেদে বর্ণিত?",               "অনুচ্ছেদ ৯",                         "9",    -1),
    # Part III — Fundamental Rights
    ("আইনের দৃষ্টিতে সমতার অধিকার কোন অনুচ্ছেদে?",      "অনুচ্ছেদ ২৭",                        "27",   -1),
    ("সকল নাগরিক কী অধিকারের অধিকারী?",                 "আইনের সমান আশ্রয় লাভের অধিকারী",   "27",   -1),
    ("জীবন ও ব্যক্তিস্বাধীনতার অধিকার কোন অনুচ্ছেদে?", "অনুচ্ছেদ ৩২",                        "32",   -1),
    ("কোনো ব্যক্তিকে জীবন হতে বঞ্চিত করার শর্ত কী?",   "আইনানুযায়ী ব্যতীত",                 "32",   -1),
    ("বাক্স্বাধীনতার অধিকার কোন অনুচ্ছেদে?",            "অনুচ্ছেদ ৩৯",                        "39",   -1),
    ("চিন্তা ও বিবেকের স্বাধীনতা কোন অনুচ্ছেদে নিশ্চিত?","অনুচ্ছেদ ৩৯",                      "39",   -1),
    ("ধর্মীয় স্বাধীনতার অধিকার কোন অনুচ্ছেদে?",        "অনুচ্ছেদ ৪১",                        "41",   -1),
    ("নাগরিকের ধর্ম পালনের অধিকার কোন অনুচ্ছেদে?",      "অনুচ্ছেদ ৪১",                        "41",   -1),
    # Part IV — The Executive
    ("রাষ্ট্রপতির বয়সসীমা কত?",                         "পঁয়ত্রিশ বৎসর",                     "48",   -1),
    ("রাষ্ট্রপতি কীভাবে নির্বাচিত হন?",                 "সংসদ-সদস্যদের দ্বারা নির্বাচিত",    "48",   -1),
    ("প্রধানমন্ত্রী কে নিয়োগ করেন?",                    "রাষ্ট্রপতি",                          "56",   -1),
    ("মন্ত্রিসভা গঠনের নিয়ম কোন অনুচ্ছেদে?",           "অনুচ্ছেদ ৫৬",                        "56",   -1),
    # Part V — The Legislature
    ("সংসদের মোট আসন সংখ্যা কত?",                        "তিনশত",                              "65",   -1),
    ("মহিলাদের জন্য সংরক্ষিত আসন কতটি?",                "পঞ্চাশটি",                           "65",   -1),
    ("জাতীয় সংসদ কোন অনুচ্ছেদে প্রতিষ্ঠিত?",           "অনুচ্ছেদ ৬৫",                        "65",   -1),
    # Part VI — The Judiciary
    ("সুপ্রীম কোর্ট কয়টি বিভাগ নিয়ে গঠিত?",            "দুইটি বিভাগ",                        "94",   -1),
    ("বাংলাদেশের সর্বোচ্চ আদালতের নাম কী?",             "বাংলাদেশ সুপ্রীম কোর্ট",            "94",   -1),
    ("বিচার বিভাগের স্বাধীনতা কোন অনুচ্ছেদে?",          "অনুচ্ছেদ ১১৬ক",                      "116A", -1),
    ("বিচারকগণ কার অধীন থাকেন?",                        "সংবিধান ও আইনের অধীন",               "116A", -1),
    # Part VII — Elections
    ("নির্বাচন কমিশন কে নিয়োগ করেন?",                   "রাষ্ট্রপতি",                          "118",  -1),
    ("প্রধান নির্বাচন কমিশনার কে নিয়োগ করেন?",          "রাষ্ট্রপতি",                          "118",  -1),
    # Part X — Amendment
    ("সংবিধান সংশোধনে কত ভাগ ভোটের দরকার?",             "দুই-তৃতীয়াংশ",                      "142",  -1),
    ("সংবিধান সংশোধনের পদ্ধতি কোন অনুচ্ছেদে?",          "অনুচ্ছেদ ১৪২",                       "142",  -1),
]


# ── Manual train/dev/test split (no sklearn needed) ───────────────────────────

def _manual_split(
    items: list,
    train_ratio: float,
    dev_ratio: float,
    seed: int,
) -> Tuple[list, list, list]:
    """Simple deterministic split without sklearn."""
    import random
    rng = random.Random(seed)
    shuffled = items[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    train_end = int(n * train_ratio)
    dev_end   = int(n * (train_ratio + dev_ratio))
    return shuffled[:train_end], shuffled[train_end:dev_end], shuffled[dev_end:]


# ── Main class ────────────────────────────────────────────────────────────────

class SQuADBuilder:
    """
    Converts QA pairs + corpus into SQuAD v1.1 JSON format.

    SQuAD v1.1 structure:
    {
      "version": "v1.1",
      "data": [
        {
          "title": "Article 7 — Supremacy of the Constitution",
          "paragraphs": [
            {
              "context": "<full article text>",
              "qas": [
                {
                  "id": "<uuid>",
                  "question": "<question in Bangla>",
                  "answers": [{"text": "<answer>", "answer_start": <int>}],
                  "is_impossible": false
                }
              ]
            }
          ]
        }
      ]
    }
    """

    def __init__(self, corpus_path: str):
        path = Path(corpus_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Corpus not found: {corpus_path}\n"
                f"Run corpus_builder.py first:\n"
                f"  python -m src.preprocessing.corpus_builder --sample"
            )
        with open(path, encoding="utf-8") as f:
            articles = json.load(f)

        self.article_map: Dict[str, Dict] = {
            a["article_number"]: a for a in articles
        }
        logger.info(f"Loaded corpus with {len(self.article_map)} articles")

    # ── Answer location ───────────────────────────────────────────────────────

    def find_answer_start(self, context: str, answer: str) -> int:
        """
        Find character offset of answer inside context.
        Tries exact match first, then case-insensitive fallback.
        Returns -1 if not found.
        """
        idx = context.find(answer)
        if idx == -1:
            idx = context.lower().find(answer.lower())
        return idx

    # ── Build from list of tuples ─────────────────────────────────────────────

    def build_from_pairs(self, qa_pairs: List[Tuple]) -> Dict:
        """
        Build a SQuAD dict from (question, answer, article_number, answer_start) tuples.
        Pass answer_start=-1 to auto-locate the answer inside the article text.
        """
        article_qa: Dict[str, List] = {}
        skipped = 0
        bad_offsets = 0

        for q, a, art_num, ans_start in qa_pairs:

            # ── Guard: article must exist in corpus ───────────────────────────
            if art_num not in self.article_map:
                logger.warning(
                    f"Article '{art_num}' not in corpus — skipping: {q[:50]}"
                )
                skipped += 1
                continue

            context = self.article_map[art_num]["text"]

            # ── Auto-locate answer if offset not given ────────────────────────
            if ans_start == -1:
                ans_start = self.find_answer_start(context, a)

            if ans_start == -1:
                logger.warning(
                    f"Answer not found in Article {art_num}: "
                    f"'{a[:30]}' | Q: {q[:50]}"
                )
                ans_start = 0   # use 0 as fallback (degraded quality)
                bad_offsets += 1

            qa_entry = {
                "id":           str(uuid.uuid4()),
                "question":     q,
                "answers":      [{"text": a, "answer_start": ans_start}],
                "is_impossible": False,
            }
            article_qa.setdefault(art_num, []).append(qa_entry)

        # ── Assemble SQuAD structure ──────────────────────────────────────────
        data = []
        for art_num, qas in article_qa.items():
            article = self.article_map[art_num]
            data.append({
                "title": f"Article {art_num} — {article.get('title_en', '')}",
                "paragraphs": [
                    {
                        "context": article["text"],
                        "qas":     qas,
                    }
                ],
            })

        total = sum(len(d["paragraphs"][0]["qas"]) for d in data)
        logger.info(
            f"Built SQuAD dataset: {total} QA pairs "
            f"| skipped: {skipped} | bad offsets: {bad_offsets}"
        )
        return {"version": "v1.1", "data": data}

    # ── Build from CSV (Label Studio export) ──────────────────────────────────

    def build_from_csv(self, csv_path: str) -> Dict:
        """
        Build from a CSV file with columns:
            question, answer, article_number, answer_start (optional)

        This is the format exported by Label Studio.
        """
        if not HAS_PANDAS:
            raise ImportError(
                "pandas is required to read CSV files.\n"
                "Install with: pip install pandas"
            )

        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        df = pd.read_csv(path, encoding="utf-8")

        required = {"question", "answer", "article_number"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"CSV is missing columns: {missing}\n"
                f"Found columns: {list(df.columns)}"
            )

        if "answer_start" not in df.columns:
            df["answer_start"] = -1

        # Drop rows with empty question or answer
        before = len(df)
        df = df.dropna(subset=["question", "answer", "article_number"])
        if len(df) < before:
            logger.warning(f"Dropped {before - len(df)} rows with missing values")

        pairs = list(
            df[["question", "answer", "article_number", "answer_start"]]
            .itertuples(index=False, name=None)
        )
        return self.build_from_pairs(pairs)

    # ── Split and save ────────────────────────────────────────────────────────

    def split_and_save(
        self,
        squad_data: Dict,
        output_dir: str,
        train_ratio: float = 0.8,
        dev_ratio:   float = 0.1,
        seed:        int   = 42,
    ) -> Dict[str, str]:
        """
        Split SQuAD data into train / dev / test sets and write to disk.
        Returns a dict of {split_name: file_path}.
        """
        # Validate ratios
        if not (0 < train_ratio < 1 and 0 < dev_ratio < 1):
            raise ValueError("train_ratio and dev_ratio must be between 0 and 1")
        if train_ratio + dev_ratio >= 1.0:
            raise ValueError("train_ratio + dev_ratio must be less than 1.0")

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # ── Flatten all QA pairs ──────────────────────────────────────────────
        all_pairs = []
        for article_block in squad_data["data"]:
            title   = article_block["title"]
            for para in article_block["paragraphs"]:
                context = para["context"]
                for qa in para["qas"]:
                    all_pairs.append({
                        "title":   title,
                        "context": context,
                        "qa":      qa,
                    })

        if len(all_pairs) < 3:
            raise ValueError(
                f"Need at least 3 QA pairs to split into train/dev/test. "
                f"Got {len(all_pairs)}."
            )

        # ── Split ─────────────────────────────────────────────────────────────
        if HAS_SKLEARN:
            test_size  = 1 - train_ratio - dev_ratio
            train_dev, test = train_test_split(
                all_pairs, test_size=test_size, random_state=seed
            )
            relative_dev = dev_ratio / (train_ratio + dev_ratio)
            train, dev = train_test_split(
                train_dev, test_size=relative_dev, random_state=seed
            )
        else:
            logger.warning(
                "scikit-learn not found — using simple random split. "
                "Install with: pip install scikit-learn"
            )
            train, dev, test = _manual_split(
                all_pairs, train_ratio, dev_ratio, seed
            )

        # ── Reassemble SQuAD format and write ────────────────────────────────
        def _pairs_to_squad(pairs: list, version: str = "v1.1") -> Dict:
            grouped: Dict = defaultdict(list)
            ctx_map: Dict = {}
            for p in pairs:
                key = (p["title"], p["context"])
                grouped[key].append(p["qa"])
                ctx_map[key] = p["context"]
            data = [
                {
                    "title": title,
                    "paragraphs": [{"context": ctx_map[(title, ctx)], "qas": qas}],
                }
                for (title, ctx), qas in grouped.items()
            ]
            return {"version": version, "data": data}

        splits = {"train": train, "dev": dev, "test": test}
        paths: Dict[str, str] = {}

        for split_name, split_data in splits.items():
            squad_split = _pairs_to_squad(split_data)
            out_path    = str(Path(output_dir) / f"{split_name}.json")

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(squad_split, f, ensure_ascii=False, indent=2)

            n = sum(len(p["paragraphs"][0]["qas"]) for p in squad_split["data"])
            logger.info(f"  {split_name}: {n} QA pairs → {out_path}")
            paths[split_name] = out_path

        return paths

    # ── Validation ────────────────────────────────────────────────────────────

    def validate(self, squad_path: str) -> Dict:
        """
        Check a saved SQuAD file for:
        - answer_start offset correctness
        - empty answer strings
        Returns a report dict with a quality_score (0-100).
        """
        path = Path(squad_path)
        if not path.exists():
            raise FileNotFoundError(f"SQuAD file not found: {squad_path}")

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        total        = 0
        bad_starts   = 0
        empty_answers = 0

        for article in data["data"]:
            for para in article["paragraphs"]:
                ctx = para["context"]
                for qa in para["qas"]:
                    total += 1
                    for ans in qa["answers"]:
                        start = ans["answer_start"]
                        text  = ans["text"]

                        # Check offset is in range
                        if start < 0 or start >= len(ctx):
                            bad_starts += 1
                        # Check offset actually points to the answer text
                        elif ctx[start : start + len(text)] != text:
                            bad_starts += 1

                        if not text.strip():
                            empty_answers += 1

        quality = round(
            (total - bad_starts - empty_answers) / max(total, 1) * 100, 1
        )
        report = {
            "total_qa_pairs":  total,
            "bad_answer_starts": bad_starts,
            "empty_answers":   empty_answers,
            "quality_score":   quality,
        }
        logger.info(f"Validation [{Path(squad_path).name}]: {report}")
        return report


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build SQuAD-format training data for the Constitution QA system."
    )
    parser.add_argument(
        "--corpus",
        default="data/processed/corpus.json",
        help="Path to corpus JSON (output of corpus_builder.py)",
    )
    parser.add_argument(
        "--annotations",
        default=None,
        help="Path to annotated CSV file (from Label Studio)",
    )
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Use the built-in 31 QA pairs (no CSV needed)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/squad/",
        help="Directory to write train.json / dev.json / test.json",
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.8,
        help="Fraction of data for training (default: 0.8)",
    )
    parser.add_argument(
        "--dev-ratio", type=float, default=0.1,
        help="Fraction of data for validation (default: 0.1)",
    )
    args = parser.parse_args()

    # ── Validate arguments ────────────────────────────────────────────────────
    if not args.bootstrap and not args.annotations:
        parser.print_help()
        logger.error(
            "\nNo input specified. Use one of:\n"
            "  --bootstrap                          (use built-in 31 QA pairs)\n"
            "  --annotations path/to/qa_pairs.csv  (use your own annotations)"
        )
        raise SystemExit(1)

    # ── Run ───────────────────────────────────────────────────────────────────
    builder = SQuADBuilder(args.corpus)

    if args.bootstrap:
        squad = builder.build_from_pairs(BOOTSTRAP_QA_PAIRS)
    else:
        squad = builder.build_from_csv(args.annotations)

    paths = builder.split_and_save(
        squad,
        args.output_dir,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
    )

    # Validate every split
    logger.info("\n── Validation ──")
    for split, path in paths.items():
        builder.validate(path)