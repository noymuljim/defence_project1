"""
src/api/main.py
────────────────
FastAPI backend for the Bangladesh Constitution QA System.

Endpoints:
    POST /api/ask          — answer a question
    POST /api/ask/batch    — batch answer multiple questions
    GET  /api/search       — retrieve relevant articles without full QA
    GET  /api/article/{id} — get a specific article
    GET  /api/health       — health check
    GET  /api/stats        — corpus + model stats

Run:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import time
import json
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager
from loguru import logger

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ── Request/Response models ───────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500,
                          example="বাংলাদেশের রাষ্ট্রধর্ম কী?")
    top_k: int = Field(default=5, ge=1, le=10,
                       description="Number of passages to retrieve")
    language: Optional[str] = Field(default="bn", description="'bn' or 'en'")


class BatchAskRequest(BaseModel):
    questions: list[str] = Field(..., min_items=1, max_items=20)
    top_k: int = Field(default=5, ge=1, le=10)


class AnswerSource(BaseModel):
    article_number: str
    title_en: str
    title_bn: str
    part: str


class Passage(BaseModel):
    article_number: str
    title_en: str
    title_bn: str
    text: str
    retrieval_score: float


class AskResponse(BaseModel):
    question: str
    answer: str
    confidence: float
    confidence_label: str  # "high" | "medium" | "low"
    source: AnswerSource
    top_passages: list[Passage]
    latency_ms: float


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=20)


class ArticleResponse(BaseModel):
    article_number: str
    article_number_bn: Optional[str]
    part: str
    part_title: str
    title_en: str
    title_bn: str
    text: str


# ── App state ─────────────────────────────────────────────────────────────────

class AppState:
    pipeline = None
    articles: list[dict] = []
    article_map: dict = {}
    start_time: float = 0


state = AppState()

CORPUS_PATH = os.getenv("CORPUS_PATH", "data/processed/corpus.json")
MODEL_PATH = os.getenv("MODEL_PATH", "models/banglabert-qa")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and corpus at startup."""
    state.start_time = time.time()
    logger.info("Starting Bangladesh Constitution QA API...")

    # Load corpus
    if Path(CORPUS_PATH).exists():
        with open(CORPUS_PATH, encoding="utf-8") as f:
            state.articles = json.load(f)
        state.article_map = {a["article_number"]: a for a in state.articles}
        logger.info(f"Corpus loaded: {len(state.articles)} articles")
    else:
        logger.warning(f"Corpus not found at {CORPUS_PATH} — run corpus_builder.py first")

    # Load QA pipeline
    try:
        from src.model.inference import ConstitutionQAPipeline
        state.pipeline = ConstitutionQAPipeline.load(
            model_path=MODEL_PATH,
            corpus_path=CORPUS_PATH,
        )
        logger.info("QA pipeline loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load QA pipeline: {e}")
        logger.warning("API will run in retrieval-only mode")

    yield

    logger.info("Shutting down...")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Bangladesh Constitution QA API",
    description="Ask questions about the Bangladesh Constitution in Bangla or English",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def confidence_label(score: float) -> str:
    if score >= 0.7:
        return "high"
    elif score >= 0.4:
        return "medium"
    return "low"


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    """Health check."""
    return {
        "status": "ok",
        "uptime_seconds": round(time.time() - state.start_time, 1),
        "corpus_loaded": len(state.articles) > 0,
        "model_loaded": state.pipeline is not None,
        "articles": len(state.articles),
    }


@app.get("/api/stats")
async def stats():
    """Corpus and model statistics."""
    if not state.articles:
        raise HTTPException(status_code=503, detail="Corpus not loaded")

    from collections import Counter
    parts = Counter(a.get("part", "?") for a in state.articles)
    return {
        "total_articles": len(state.articles),
        "articles_by_part": dict(parts),
        "model_path": MODEL_PATH,
        "corpus_path": CORPUS_PATH,
    }


@app.post("/api/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    """
    Answer a question about the Bangladesh Constitution.
    Returns the extracted answer span with article citation.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    t0 = time.time()

    if state.pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="QA model not loaded. Run: python -m src.model.trainer first"
        )

    try:
        result = state.pipeline.answer(request.question, top_k=request.top_k)
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    latency = round((time.time() - t0) * 1000, 1)

    return AskResponse(
        question=result.question,
        answer=result.answer,
        confidence=round(result.confidence, 4),
        confidence_label=confidence_label(result.confidence),
        source=AnswerSource(
            article_number=result.source_article_number,
            title_en=result.source_article_title_en,
            title_bn=result.source_article_title_bn,
            part=result.source_part,
        ),
        top_passages=[
            Passage(
                article_number=p["article_number"],
                title_en=p.get("title_en", ""),
                title_bn=p.get("title_bn", ""),
                text=p["text"],
                retrieval_score=p.get("retrieval_score", 0),
            )
            for p in result.top_passages
        ],
        latency_ms=latency,
    )


@app.post("/api/ask/batch")
async def batch_ask(request: BatchAskRequest):
    """Answer multiple questions in one request."""
    if state.pipeline is None:
        raise HTTPException(status_code=503, detail="QA model not loaded")

    results = []
    for q in request.questions:
        try:
            r = state.pipeline.answer(q, top_k=request.top_k)
            results.append({
                "question": r.question,
                "answer": r.answer,
                "confidence": round(r.confidence, 4),
                "source_article": r.source_article_number,
            })
        except Exception as e:
            results.append({"question": q, "error": str(e)})

    return {"results": results, "count": len(results)}


@app.get("/api/search")
async def search(query: str, top_k: int = 5):
    """
    Retrieve relevant articles without full QA inference.
    Useful for browsing related constitutional provisions.
    """
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    if state.pipeline is None or state.pipeline.retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not loaded")

    try:
        passages = state.pipeline.retriever.retrieve(query, top_k=min(top_k, 20))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "query": query,
        "results": [
            {
                "article_number": p["article_number"],
                "title_en": p.get("title_en", ""),
                "title_bn": p.get("title_bn", ""),
                "part": p.get("part", ""),
                "text": p["text"][:400],
                "retrieval_score": p.get("retrieval_score", 0),
            }
            for p in passages
        ],
    }


@app.get("/api/article/{article_number}", response_model=ArticleResponse)
async def get_article(article_number: str):
    """Retrieve a specific constitutional article by its number."""
    if not state.article_map:
        raise HTTPException(status_code=503, detail="Corpus not loaded")

    article = state.article_map.get(article_number)
    if article is None:
        # Try uppercase variant
        article = state.article_map.get(article_number.upper())

    if article is None:
        raise HTTPException(
            status_code=404,
            detail=f"Article {article_number} not found"
        )

    return ArticleResponse(
        article_number=article["article_number"],
        article_number_bn=article.get("article_number_bn"),
        part=article.get("part", ""),
        part_title=article.get("part_title", ""),
        title_en=article.get("title_en", ""),
        title_bn=article.get("title_bn", ""),
        text=article["text"],
    )


@app.get("/api/articles")
async def list_articles(part: Optional[str] = None, limit: int = 50, offset: int = 0):
    """List all articles, optionally filtered by part."""
    articles = state.articles
    if part:
        articles = [a for a in articles if a.get("part", "").upper() == part.upper()]
    total = len(articles)
    page = articles[offset:offset + limit]
    return {
        "total": total,
        "articles": [
            {
                "article_number": a["article_number"],
                "title_en": a.get("title_en", ""),
                "title_bn": a.get("title_bn", ""),
                "part": a.get("part", ""),
            }
            for a in page
        ],
    }


# ── Error handlers ────────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
