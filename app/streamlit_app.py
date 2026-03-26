"""
📚 AI Book Recommender — Production Streamlit App
──────────────────────────────────────────────────
Loads model from MLflow registry (with pickle fallback),
supports monitoring, and displays model metadata.
"""

import sys
import time
import urllib.parse
from pathlib import Path

import streamlit as st

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import BookRecommender
from src.monitoring import MonitoringService

# ─── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="AI Book Recommender",
    page_icon="📚",
    layout="wide",
)

# ─── Load Model & Monitoring (cached) ────────────────────────
@st.cache_resource
def load_recommender():
    """Load the recommendation engine (cached for performance)."""
    try:
        return BookRecommender(use_mlflow=True)
    except Exception:
        return BookRecommender(use_mlflow=False)


@st.cache_resource
def load_monitor():
    """Load the monitoring service."""
    return MonitoringService()


recommender = load_recommender()
monitor = load_monitor()


def get_amazon_link(book_title):
    """Generate Amazon search link for a book."""
    encoded = urllib.parse.quote_plus(book_title)
    return f"https://www.amazon.in/s?k={encoded}"


# ─── Custom CSS ───────────────────────────────────────────────
st.markdown("""
    <style>
    html, body, [class*="css"]  {
        background-color: #101010;
        color: #f2f2f2;
        font-family: 'Segoe UI', sans-serif;
    }
    h1, h3, h5 {
        font-family: 'Segoe UI', sans-serif;
    }
    .book-container {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 15px;
        text-align: center;
        transition: 0.3s ease;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    }
    .book-container:hover {
        transform: scale(1.03);
        border: 1px solid #0cf;
        box-shadow: 0 0 15px #0cf;
    }
    .amazon-button {
        background-color: #FF9900;
        color: white;
        padding: 8px 14px;
        border-radius: 10px;
        font-weight: bold;
        text-decoration: none;
        display: inline-block;
        margin-top: 10px;
    }
    .amazon-button:hover {
        background-color: #e88b00;
    }
    .model-info {
        background: rgba(0, 204, 255, 0.05);
        border: 1px solid rgba(0, 204, 255, 0.2);
        border-radius: 10px;
        padding: 12px 16px;
        margin: 10px 0;
        font-size: 13px;
    }
    .stat-card {
        background: rgba(255, 215, 0, 0.05);
        border: 1px solid rgba(255, 215, 0, 0.2);
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)


# ─── Sidebar: Model Info & Monitoring ─────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Model Info")
    model_info = recommender.get_model_info()
    st.markdown(f"""
    <div class="model-info">
        <b>Version:</b> {model_info['version']}<br>
        <b>Books:</b> {model_info['n_books']}<br>
        <b>Pivot:</b> {model_info['pivot_shape']}
    </div>
    """, unsafe_allow_html=True)

    if model_info.get("metrics"):
        st.markdown("### 📊 Model Metrics")
        for key, val in model_info["metrics"].items():
            if isinstance(val, float):
                st.metric(key, f"{val:.4f}")

    st.markdown("---")
    st.markdown("### 📈 Monitoring")
    try:
        stats = monitor.get_stats()
        col1, col2 = st.columns(2)
        col1.metric("Total Predictions", stats["total_predictions"])
        col2.metric("Today", stats["predictions_today"])

        if stats["avg_response_time_ms"] > 0:
            st.metric("Avg Response", f"{stats['avg_response_time_ms']}ms")

        if stats["top_queried_books"]:
            st.markdown("**🔥 Top Queried:**")
            for book, count in stats["top_queried_books"][:5]:
                st.text(f"  {book[:30]}... ({count})")
    except Exception:
        st.info("No monitoring data yet.")

    st.markdown("---")
    st.markdown(
        "<p style='text-align:center; font-size:12px; color:#666;'>"
        "Built with MLOps ❤️</p>",
        unsafe_allow_html=True
    )


# ─── Main Content ────────────────────────────────────────────
st.markdown(
    "<h1 style='text-align:center; color:#FFD700;'>📚 AI Book Recommender</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;'>Your personalized reading assistant powered by AI 🔍</p>",
    unsafe_allow_html=True
)

# ─── Book Selection ──────────────────────────────────────────
selected_book = st.selectbox(
    "Choose a book you like 👇",
    recommender.book_names,
    key="select_book"
)

# ─── Recommend Button ────────────────────────────────────────
if st.button("🔍 Recommend Books"):
    start = time.time()

    try:
        results = recommender.recommend(selected_book)
        elapsed_ms = (time.time() - start) * 1000

        # Log to monitoring
        rec_titles = [r["title"] for r in results]
        monitor.log_prediction(
            query_book=selected_book,
            recommended_books=rec_titles,
            model_version=model_info["version"],
            response_time_ms=elapsed_ms
        )

        st.markdown("---")
        st.markdown(
            "<h3 style='color:#00FFFF;'>📘 Recommended for You:</h3>",
            unsafe_allow_html=True
        )

        cols = st.columns(5)
        for idx, col in enumerate(cols):
            if idx < len(results):
                rec = results[idx]
                amazon_link = get_amazon_link(rec["title"])

                with col:
                    st.markdown(f"""
                        <div class="book-container">
                            <img src="{rec['image_url']}" width="120" style="border-radius: 8px;" />
                            <h5 style="color:#ffffff; font-size:14px;">{rec['title']}</h5>
                            <p style="font-size:11px; color:#888;">
                                Distance: {rec['distance']:.3f}
                            </p>
                            <a href="{amazon_link}" class="amazon-button" target="_blank">
                                🛒 View on Amazon
                            </a>
                        </div>
                    """, unsafe_allow_html=True)

        st.markdown(
            f"<p style='text-align:right; color:#555; font-size:12px;'>"
            f"⏱️ {elapsed_ms:.0f}ms | Model: {model_info['version']}</p>",
            unsafe_allow_html=True
        )

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
