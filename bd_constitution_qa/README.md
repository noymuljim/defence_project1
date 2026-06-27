# Bangladesh Constitution QA System
**বাংলাদেশ সংবিধান প্রশ্নোত্তর ব্যবস্থা**

An end-to-end NLP system for answering questions about the Bangladesh Constitution using BanglaBERT and hybrid retrieval-augmented generation.

---

## Architecture

```
User Query (Bangla/English)
        │
        ▼
┌─────────────────┐
│ Preprocessing   │  Unicode normalization, tokenization (bnlp-toolkit)
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
┌───▼───┐ ┌──▼────────┐
│ BM25  │ │   FAISS   │   Hybrid Retrieval (BM25 0.4 + Dense 0.6)
└───┬───┘ └──┬────────┘
    └────┬───┘
         │  Top-5 passages
         ▼
┌─────────────────┐
│   BanglaBERT    │  Fine-tuned QA head (span extraction)
│  QA Model       │
└────────┬────────┘
         │
         ▼
  Answer + Article Citation
```

## Project Structure

```
bd_constitution_qa/
├── configs/
│   └── config.yaml              # Central config
├── data/
│   ├── raw/                     # Constitution PDFs
│   ├── processed/corpus.json    # Structured article corpus
│   ├── squad/                   # SQuAD train/dev/test
│   └── index/                   # BM25 + FAISS indices
├── models/
│   └── banglabert-qa/           # Fine-tuned model checkpoint
├── notebooks/
│   └── train_colab.ipynb        # Google Colab training notebook
├── scripts/
│   ├── setup.py                 # One-command project setup
│   └── evaluate.py              # Full evaluation suite
├── src/
│   ├── preprocessing/
│   │   ├── corpus_builder.py    # PDF → article JSON
│   │   └── squad_builder.py     # QA pairs → SQuAD format
│   ├── retrieval/
│   │   └── retriever.py         # HybridRetriever (BM25 + FAISS)
│   ├── model/
│   │   ├── trainer.py           # BanglaBERT fine-tuning
│   │   └── inference.py         # QA pipeline
│   ├── api/
│   │   └── main.py              # FastAPI backend
│   └── ui/
│       └── app.py               # Streamlit frontend
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
pip install git+https://github.com/csebuetnlp/normalizer
```

### 2. Setup (builds sample corpus + training data + index)
```bash
python scripts/setup.py
```

### 3. Train on Google Colab (recommended)
Open `notebooks/train_colab.ipynb` in [Google Colab](https://colab.research.google.com/).
Select Runtime → T4 or A100 GPU.

### 3b. Train locally (requires CUDA GPU)
```bash
python -m src.model.trainer \
    --train data/squad/train.json \
    --dev   data/squad/dev.json \
    --output models/banglabert-qa \
    --epochs 4
```

### 4. Start the API
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### 5. Start the UI
```bash
streamlit run src/ui/app.py
```

### 6. Evaluate
```bash
python scripts/evaluate.py --test data/squad/test.json
```

---

## Using the API

### Ask a question
```bash
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "বাংলাদেশের রাষ্ট্রধর্ম কী?", "top_k": 5}'
```

Response:
```json
{
  "question": "বাংলাদেশের রাষ্ট্রধর্ম কী?",
  "answer": "ইসলাম",
  "confidence": 0.92,
  "confidence_label": "high",
  "source": {
    "article_number": "2A",
    "title_en": "State religion",
    "title_bn": "রাষ্ট্রধর্ম",
    "part": "I"
  },
  "top_passages": [...],
  "latency_ms": 145.3
}
```

### Search articles
```bash
curl "http://localhost:8000/api/search?query=মৌলিক+অধিকার&top_k=5"
```

### Get a specific article
```bash
curl "http://localhost:8000/api/article/7"
```

---

## Constitution Corpus

The corpus should be the official Bangla version of the Bangladesh Constitution
from [bdlaws.minlaw.gov.bd](http://bdlaws.minlaw.gov.bd/).

Place the PDF at `data/raw/constitution_bn.pdf`, then run:
```bash
python -m src.preprocessing.corpus_builder \
    --pdf data/raw/constitution_bn.pdf \
    --output data/processed/corpus.json \
    --lang bn
```

If the PDF is scanned (image-based), extract text with OCR first:
```bash
# Using Tesseract with Bengali language pack
tesseract constitution.pdf output_text -l ben
```

---

## Annotation Guide

Use [Label Studio](https://labelstud.io/) for annotating QA pairs.

1. Install: `pip install label-studio`
2. Start: `label-studio start`
3. Import corpus articles as documents
4. Use the **Question Answering** template
5. Annotate with 2+ annotators, target Cohen's Kappa > 0.7
6. Export as CSV and run:
   ```bash
   python -m src.preprocessing.squad_builder \
       --annotations data/annotations/qa_pairs.csv \
       --corpus data/processed/corpus.json \
       --output-dir data/squad/
   ```

---

## Model Details

| Component | Value |
|-----------|-------|
| Base model | `csebuetnlp/banglabert` |
| Task | Extractive QA (span prediction) |
| Training framework | HuggingFace Transformers |
| Max sequence length | 512 tokens |
| Doc stride | 128 tokens |
| Embedding model | `paraphrase-multilingual-MiniLM-L12-v2` |
| Retrieval | BM25 (40%) + Dense FAISS (60%) |

## Target Metrics

| Metric | Baseline (mBERT zero-shot) | Target (BanglaBERT fine-tuned) |
|--------|--------------------------|-------------------------------|
| Exact Match | ~34% | 70%+ |
| F1 Score | ~41% | 80%+ |
| Recall@5 | ~62% | 90%+ |

---

## Tech Stack

- **Model**: `csebuetnlp/banglabert` (BanglaBERT)
- **Training**: HuggingFace Transformers + Trainer API
- **Retrieval**: rank-bm25 + FAISS + sentence-transformers
- **Backend**: FastAPI + Uvicorn
- **Frontend**: Streamlit
- **Bangla NLP**: bnlp-toolkit, csebuetnlp/normalizer
- **Tracking**: Weights & Biases
- **GPU**: Google Colab Pro (A100)

---

## References

- Bhattacharjee et al. (2022). [BanglaBERT: Language Model Pretraining and Benchmarks for Low-Resource Language Understanding Evaluation in Bangla](https://arxiv.org/abs/2101.00204)
- Rajpurkar et al. (2016). [SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/abs/1606.05250)
- Bangladesh Government. [Constitution of Bangladesh](http://bdlaws.minlaw.gov.bd/act-367.html)
