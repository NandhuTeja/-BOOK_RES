# 📚 AI Book Recommender — Production ML System

> A production-grade Book Recommendation System built with **MLOps best practices**, featuring modular ML pipelines, experiment tracking with **MLflow**, model versioning, automated training, monitoring, and **Docker** containerization.

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.33+-red?logo=streamlit)
![MLflow](https://img.shields.io/badge/MLflow-2.10+-blue?logo=mlflow)
![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🏗️ Architecture

```
Raw Data (BX-Books, BX-Ratings, BX-Users)
   ↓
Data Validation (missing values, duplicates, ranges)
   ↓
Preprocessing (merge, filter active users/popular books)
   ↓
Feature Engineering (pivot table, sparse matrix)
   ↓
Train/Test Split + Cross-Validation
   ↓
Model Training (KNN Collaborative Filtering)
   ↓
Evaluation (Precision@K, Recall@K, NDCG@K, Hit Rate)
   ↓
MLflow Tracking + Model Registry
   ↓
Dockerized Streamlit App
   ↓
Monitoring (SQLite logging)
```

---

## 📂 Project Structure

```
book-recommender/
│
├── data/
│   ├── raw/                    # Original CSV datasets
│   │   ├── BX-Books.csv
│   │   ├── BX-Book-Ratings.csv
│   │   └── BX-Users.csv
│   └── processed/              # Cleaned & engineered data
│
├── src/
│   ├── config.py               # Central configuration
│   ├── data_validation.py      # Data quality checks
│   ├── data_preprocessing.py   # Data cleaning pipeline
│   ├── feature_engineering.py  # Pivot table & sparse matrix
│   ├── train.py                # Full training pipeline + MLflow
│   ├── evaluate.py             # Evaluation metrics
│   ├── predict.py              # Prediction engine
│   └── monitoring.py           # SQLite-based monitoring
│
├── models/                     # Trained model artifacts
├── app/
│   └── streamlit_app.py        # Production Streamlit app
│
├── .github/workflows/
│   └── ci.yml                  # CI/CD pipeline
│
├── Dockerfile                  # Container definition
├── docker-compose.yml          # Multi-service setup
├── .dockerignore
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🚀 Quick Start

### Option 1: Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/book-recommender.git
cd book-recommender

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the training pipeline
python src/train.py

# 5. Launch the app
streamlit run app/streamlit_app.py
```

### Option 2: Docker

```bash
# Build and run
docker build -t book-recommender .
docker run -p 8501:8501 book-recommender

# Or with Docker Compose (includes MLflow)
docker-compose up --build
```

---

## 🏋️ Training Pipeline

Run the full end-to-end training:

```bash
# Full pipeline (data validation + preprocessing + training + evaluation + MLflow)
python src/train.py

# Skip MLflow tracking
python src/train.py --no-mlflow

# Custom hyperparameters
python src/train.py --n-neighbors 10 --metric euclidean --algorithm ball_tree

# Quick training (skip validation and cross-validation)
python src/train.py --no-validation --no-cv
```

### Pipeline Steps:
1. **Data Validation** — Checks data quality before training
2. **Preprocessing** — Merges datasets, filters active users and popular books
3. **Feature Engineering** — Creates user-book pivot table and sparse matrix
4. **Cross-Validation** — 5-fold CV for model robustness
5. **Model Training** — KNN with configurable hyperparameters
6. **Evaluation** — Precision@K, Recall@K, NDCG@K, Hit Rate, Coverage
7. **MLflow Logging** — Tracks parameters, metrics, and model artifacts

---

## 📊 MLflow Experiment Tracking

Access the MLflow UI:

```bash
# Start MLflow server
mlflow ui --port 5000

# Or with Docker Compose (automatic)
docker-compose up
```

Visit **http://localhost:5000** to view:
- Experiment runs with hyperparameters
- Evaluation metrics comparison
- Model versions in the registry
- Production model promotion

---

## 📈 Evaluation Metrics

| Metric | Description |
|--------|------------|
| **Precision@K** | Fraction of relevant items in top-K |
| **Recall@K** | Fraction of relevant items retrieved |
| **NDCG@K** | Ranking quality with position discounting |
| **Hit Rate** | At least one relevant item recommended |
| **Coverage** | Fraction of catalog recommended |

---

## 🐳 Docker Commands

```bash
# Build image
docker build -t book-recommender .

# Run container
docker run -p 8501:8501 book-recommender

# Docker Compose (Streamlit + MLflow)
docker-compose up --build

# Stop services
docker-compose down
```

---

## 🔄 CI/CD Pipeline

GitHub Actions workflow (`.github/workflows/ci.yml`):

1. **Validate** — Run data validation checks
2. **Train** — Execute training pipeline
3. **Docker** — Build and verify Docker image

Triggered on push/PR to `main` branch.

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| ML Model | scikit-learn (NearestNeighbors) |
| Experiment Tracking | MLflow |
| Frontend | Streamlit |
| Containerization | Docker + Docker Compose |
| CI/CD | GitHub Actions |
| Monitoring | SQLite |
| Data Processing | Pandas, NumPy, SciPy |
| Language | Python 3.10 |

---

## 📊 Dataset

**Book-Crossing Dataset** containing:
- **271,379** books with metadata
- **1,149,780** ratings from **278,858** users
- Rating scale: 1–10 (explicit), 0 (implicit)

---

## 📝 License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built with MLOps best practices ❤️*
