"""
Centralized configuration for the Book Recommender MLOps pipeline.
All paths, hyperparameters, and settings are defined here.
"""

import os
from pathlib import Path

# ─── Project Paths ───────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# ─── Raw Data Files ──────────────────────────────────────────────
BOOKS_CSV = DATA_RAW_DIR / "BX-Books.csv"
RATINGS_CSV = DATA_RAW_DIR / "BX-Book-Ratings.csv"
USERS_CSV = DATA_RAW_DIR / "BX-Users.csv"

# ─── Processed Data Files ────────────────────────────────────────
PROCESSED_RATINGS = DATA_PROCESSED_DIR / "filtered_ratings.pkl"
PROCESSED_BOOKS = DATA_PROCESSED_DIR / "filtered_books.pkl"
BOOK_PIVOT_PATH = DATA_PROCESSED_DIR / "book_pivot.pkl"
BOOK_NAMES_PATH = DATA_PROCESSED_DIR / "book_names.pkl"
FINAL_RATING_PATH = DATA_PROCESSED_DIR / "final_rating.pkl"

# ─── Model Paths ─────────────────────────────────────────────────
MODEL_PATH = MODELS_DIR / "model.pkl"

# ─── Data Filtering Thresholds ───────────────────────────────────
MIN_USER_RATINGS = 200      # Users must have rated at least this many books
MIN_BOOK_RATINGS = 50       # Books must have at least this many ratings
RATING_MIN = 1
RATING_MAX = 10

# ─── Model Hyperparameters ───────────────────────────────────────
KNN_N_NEIGHBORS = 6         # Number of neighbors for KNN
KNN_ALGORITHM = "brute"     # Algorithm: 'auto', 'ball_tree', 'kd_tree', 'brute'
KNN_METRIC = "cosine"       # Distance metric: 'cosine', 'euclidean', 'manhattan'
N_RECOMMENDATIONS = 5       # Number of recommendations to show

# ─── Train/Test Split ────────────────────────────────────────────
TEST_SIZE = 0.2
RANDOM_SEED = 42
CV_FOLDS = 5

# ─── MLflow Configuration ────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MLFLOW_EXPERIMENT_NAME = "book-recommender"

# ─── Monitoring Database ─────────────────────────────────────────
MONITORING_DB = PROJECT_ROOT / "monitoring.db"

# ─── Evaluation ──────────────────────────────────────────────────
EVAL_K_VALUES = [5, 10]     # K values for Precision@K, Recall@K, NDCG@K
