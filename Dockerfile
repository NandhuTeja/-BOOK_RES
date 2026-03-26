# ─── Base Image ───────────────────────────────────────────────
FROM python:3.10-slim

# ─── Metadata ─────────────────────────────────────────────────
LABEL maintainer="Mallikarjuna Reddy Gali"
LABEL description="AI Book Recommender — Production ML System"
LABEL version="2.0"

# ─── System Dependencies ─────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ─── Working Directory ───────────────────────────────────────
WORKDIR /app

# ─── Install Python Dependencies ─────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ─── Copy Project Files ──────────────────────────────────────
COPY src/ ./src/
COPY app/ ./app/
COPY data/ ./data/
COPY models/ ./models/

# ─── Environment Variables ───────────────────────────────────
ENV PYTHONUNBUFFERED=1
ENV MLFLOW_TRACKING_URI=sqlite:///mlflow.db

# ─── Expose Streamlit Port ───────────────────────────────────
EXPOSE 8501

# ─── Health Check ─────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# ─── Run Streamlit ───────────────────────────────────────────
ENTRYPOINT ["streamlit", "run", "app/streamlit_app.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.headless=true"]
