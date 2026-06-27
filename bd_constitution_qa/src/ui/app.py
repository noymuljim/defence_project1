"""
src/ui/app.py
──────────────
Streamlit web interface for the Bangladesh Constitution QA System.

Run:
    streamlit run src/ui/app.py

Requires either:
  (a) The FastAPI backend running at localhost:8000, or
  (b) The QA pipeline loaded directly (set USE_API=False below)
"""

import os
import time
import json
import requests
import streamlit as st
from pathlib import Path
from typing import Optional

# ── Configuration ─────────────────────────────────────────────────────────────
API_URL = os.getenv("API_URL", "http://localhost:8000")
USE_API = os.getenv("USE_API", "true").lower() == "true"
CORPUS_PATH = os.getenv("CORPUS_PATH", "data/processed/corpus.json")
MODEL_PATH = os.getenv("MODEL_PATH", "models/banglabert-qa")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="বাংলাদেশ সংবিধান QA",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Serif+Bengali:wght@400;600&display=swap');

    .bangla-text {
        font-family: 'Noto Serif Bengali', serif;
        font-size: 1.1rem;
        line-height: 1.8;
    }
    .bangla-answer {
        font-family: 'Noto Serif Bengali', serif;
        font-size: 1.3rem;
        font-weight: 600;
        color: #1a4a8a;
        background: #e8f0fb;
        padding: 12px 16px;
        border-left: 4px solid #1a4a8a;
        border-radius: 4px;
        margin: 8px 0;
    }
    .article-badge {
        display: inline-block;
        background: #dbeafe;
        color: #1e40af;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 8px;
    }
    .confidence-high { color: #166534; font-weight: 600; }
    .confidence-medium { color: #854d0e; font-weight: 600; }
    .confidence-low { color: #991b1b; font-weight: 600; }
    .passage-card {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px;
        margin: 6px 0;
        background: #f8fafc;
    }
    .header-bn {
        font-family: 'Noto Serif Bengali', serif;
        font-size: 1.4rem;
        color: #374151;
    }
</style>
""", unsafe_allow_html=True)


# ── Pipeline loader (direct mode) ─────────────────────────────────────────────
@st.cache_resource
def load_pipeline():
    """Load QA pipeline directly (bypasses API)."""
    try:
        from src.model.inference import ConstitutionQAPipeline
        pipeline = ConstitutionQAPipeline.load(
            model_path=MODEL_PATH,
            corpus_path=CORPUS_PATH,
        )
        return pipeline, None
    except Exception as e:
        return None, str(e)


@st.cache_data
def load_corpus():
    if Path(CORPUS_PATH).exists():
        with open(CORPUS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return []


# ── API helpers ───────────────────────────────────────────────────────────────
def api_ask(question: str, top_k: int = 5) -> Optional[dict]:
    try:
        resp = requests.post(
            f"{API_URL}/api/ask",
            json={"question": question, "top_k": top_k},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def api_search(query: str, top_k: int = 5) -> Optional[dict]:
    try:
        resp = requests.get(
            f"{API_URL}/api/search",
            params={"query": query, "top_k": top_k},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def api_health() -> bool:
    try:
        resp = requests.get(f"{API_URL}/api/health", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


# ── UI helpers ────────────────────────────────────────────────────────────────
def render_confidence(label: str, score: float):
    css_class = f"confidence-{label}"
    emoji = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(label, "⚪")
    st.markdown(
        f'<span class="{css_class}">{emoji} {label.capitalize()} confidence ({score:.0%})</span>',
        unsafe_allow_html=True,
    )


def render_answer(result: dict):
    """Render a QA result in a structured format."""
    answer = result.get("answer", "")
    source = result.get("source", {})
    confidence = result.get("confidence", 0)
    conf_label = result.get("confidence_label", "low")
    latency = result.get("latency_ms", 0)

    st.markdown("### উত্তর / Answer")
    st.markdown(f'<div class="bangla-answer">{answer}</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        art_num = source.get("article_number", "")
        art_title = source.get("title_en", "")
        st.markdown(f'<span class="article-badge">Article {art_num}</span>', unsafe_allow_html=True)
        if art_title:
            st.caption(art_title)
    with col2:
        render_confidence(conf_label, confidence)
    with col3:
        st.caption(f"⚡ {latency:.0f} ms")

    # Source context
    with st.expander(f"📄 Source: Article {art_num} — {source.get('title_bn', '')}"):
        passages = result.get("top_passages", [])
        if passages:
            top = passages[0]
            st.markdown(
                f'<p class="bangla-text">{top.get("text", "")}</p>',
                unsafe_allow_html=True,
            )
        st.caption(f"Part {source.get('part', '')} — {source.get('title_en', '')}")

    # Retrieved passages
    with st.expander(f"🔍 Retrieved passages ({len(result.get('top_passages', []))})"):
        for i, p in enumerate(result.get("top_passages", []), 1):
            score = p.get("retrieval_score", 0)
            bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
            st.markdown(f"""
            <div class="passage-card">
              <strong>#{i} Article {p.get('article_number')} — {p.get('title_en', '')}</strong><br>
              <small style="color:#6b7280">{bar} {score:.3f}</small><br>
              <span class="bangla-text">{p.get('text', '')[:250]}...</span>
            </div>
            """, unsafe_allow_html=True)


# ── Main UI ───────────────────────────────────────────────────────────────────
def main():
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚖️ বাংলাদেশ সংবিধান")
        st.markdown('<p class="header-bn">Constitution QA System</p>', unsafe_allow_html=True)
        st.divider()

        mode = st.radio("Mode", ["Ask a Question", "Browse Articles", "Search & Retrieve"])
        st.divider()

        top_k = st.slider("Retrieved passages", min_value=1, max_value=10, value=5)

        st.divider()
        # API status
        if USE_API:
            healthy = api_health()
            status = "🟢 API connected" if healthy else "🔴 API offline"
            st.caption(status)
        else:
            st.caption("🔵 Direct mode")

        st.markdown("---")
        st.caption("Built with BanglaBERT + HuggingFace")
        st.caption("Model: csebuetnlp/banglabert")

    # Header
    st.title("⚖️ Bangladesh Constitution QA")
    st.markdown(
        '<p class="header-bn">বাংলাদেশের সংবিধান সম্পর্কে যেকোনো প্রশ্ন করুন</p>',
        unsafe_allow_html=True,
    )

    # ── Mode: Ask a Question ──────────────────────────────────────────────────
    if mode == "Ask a Question":
        # Example questions
        examples = [
            "বাংলাদেশের রাষ্ট্রধর্ম কী?",
            "মৌলিক অধিকার কোন ভাগে বর্ণিত?",
            "সংসদের মোট আসন সংখ্যা কত?",
            "রাষ্ট্রপতির বয়সসীমা কত?",
            "সংবিধান সংশোধনে কত ভাগ ভোটের দরকার?",
            "বিচার বিভাগের স্বাধীনতা কোন অনুচ্ছেদে?",
            "What is the state language of Bangladesh?",
            "Who appoints the Prime Minister?",
        ]

        st.markdown("**Example questions:**")
        cols = st.columns(4)
        for i, ex in enumerate(examples):
            with cols[i % 4]:
                if st.button(ex[:30] + ("..." if len(ex) > 30 else ""), key=f"ex_{i}"):
                    st.session_state["current_question"] = ex

        st.divider()
        question = st.text_input(
            "Ask your question (Bangla or English):",
            value=st.session_state.get("current_question", ""),
            placeholder="যেমন: বাংলাদেশের রাষ্ট্রধর্ম কী?",
            key="question_input",
        )

        col1, col2 = st.columns([1, 5])
        with col1:
            ask_btn = st.button("🔍 Ask", type="primary", use_container_width=True)

        if ask_btn and question.strip():
            with st.spinner("Searching the Constitution..."):
                if USE_API:
                    result = api_ask(question, top_k=top_k)
                    if result is None:
                        st.error("API not available. Start the backend: uvicorn src.api.main:app")
                        st.stop()
                else:
                    pipeline, err = load_pipeline()
                    if err or pipeline is None:
                        st.error(f"Failed to load pipeline: {err}")
                        st.stop()
                    t0 = time.time()
                    qa_result = pipeline.answer(question, top_k=top_k)
                    result = qa_result.to_dict()
                    result["latency_ms"] = round((time.time() - t0) * 1000, 1)
                    result["confidence_label"] = (
                        "high" if result["confidence"] >= 0.7
                        else "medium" if result["confidence"] >= 0.4
                        else "low"
                    )

            render_answer(result)

    # ── Mode: Browse Articles ─────────────────────────────────────────────────
    elif mode == "Browse Articles":
        articles = load_corpus()
        if not articles:
            st.error(f"Corpus not found at {CORPUS_PATH}. Run corpus_builder.py first.")
            st.stop()

        parts = sorted(set(a.get("part", "?") for a in articles))
        selected_part = st.selectbox("Filter by Part", ["All"] + parts)

        filtered = articles if selected_part == "All" else [
            a for a in articles if a.get("part") == selected_part
        ]
        st.caption(f"Showing {len(filtered)} articles")

        for art in filtered:
            with st.expander(f"Article {art['article_number']} — {art.get('title_en', '')}"):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(
                        f'<p class="bangla-text">{art.get("text", "")}</p>',
                        unsafe_allow_html=True,
                    )
                with col2:
                    st.markdown(f"**Part:** {art.get('part', '')}")
                    st.markdown(f"**Bangla title:** {art.get('title_bn', '')}")
                    if st.button("Ask about this", key=f"ask_{art['article_number']}"):
                        st.session_state["current_question"] = f"অনুচ্ছেদ {art.get('article_number_bn', art['article_number'])} সম্পর্কে বলুন"

    # ── Mode: Search & Retrieve ───────────────────────────────────────────────
    elif mode == "Search & Retrieve":
        st.markdown("### 🔍 Semantic Search (Retrieval Only)")
        st.caption("Find relevant articles without running the full QA model")

        query = st.text_input(
            "Search query:",
            placeholder="যেমন: মৌলিক অধিকার",
        )

        if st.button("Search", type="primary") and query.strip():
            with st.spinner("Searching..."):
                if USE_API:
                    result = api_search(query, top_k=top_k)
                else:
                    pipeline, err = load_pipeline()
                    if err or pipeline is None:
                        st.error(str(err))
                        st.stop()
                    passages = pipeline.retriever.retrieve(query, top_k=top_k)
                    result = {"query": query, "results": passages}

            if result:
                st.markdown(f"**Results for:** _{result['query']}_")
                for i, r in enumerate(result.get("results", []), 1):
                    score = r.get("retrieval_score", 0)
                    with st.expander(
                        f"#{i} Article {r.get('article_number')} — {r.get('title_en', '')} "
                        f"(score: {score:.3f})"
                    ):
                        st.markdown(
                            f'<p class="bangla-text">{r.get("text", "")}</p>',
                            unsafe_allow_html=True,
                        )
                        st.caption(f"Part {r.get('part', '')} | BM25: {r.get('bm25_score', 0):.3f} | Dense: {r.get('dense_score', 0):.3f}")


if __name__ == "__main__":
    main()
