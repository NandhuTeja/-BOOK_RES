"""
Feature Engineering Module
──────────────────────────
Creates the user-book pivot table and sparse matrix
used for collaborative filtering.
"""

import sys
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    FINAL_RATING_PATH, BOOK_PIVOT_PATH, BOOK_NAMES_PATH, DATA_PROCESSED_DIR
)


def load_processed_data():
    """Load preprocessed final rating data."""
    print("📂 Loading processed data...")
    final_rating = pickle.load(open(FINAL_RATING_PATH, "rb"))
    print(f"   Loaded {final_rating.shape[0]:,} ratings")
    return final_rating


def create_pivot_table(final_rating: pd.DataFrame) -> pd.DataFrame:
    """Create user-book rating pivot table."""
    print("📊 Creating pivot table...")
    book_pivot = final_rating.pivot_table(
        columns="User-ID",
        index="Book-Title",
        values="Book-Rating"
    )
    book_pivot.fillna(0, inplace=True)
    print(f"   Pivot shape: {book_pivot.shape[0]} books × {book_pivot.shape[1]} users")
    return book_pivot


def create_sparse_matrix(book_pivot: pd.DataFrame) -> csr_matrix:
    """Convert pivot table to sparse matrix for efficient computation."""
    print("🔢 Creating sparse matrix...")
    sparse = csr_matrix(book_pivot.values)
    density = sparse.nnz / (sparse.shape[0] * sparse.shape[1]) * 100
    print(f"   Sparse matrix: {sparse.shape}, density: {density:.2f}%")
    return sparse


def extract_book_names(book_pivot: pd.DataFrame) -> list:
    """Extract list of book names from pivot table index."""
    book_names = list(book_pivot.index)
    print(f"📖 Extracted {len(book_names)} book names")
    return book_names


def engineer_features(save=True):
    """Run the full feature engineering pipeline."""
    print("=" * 60)
    print("🛠️  FEATURE ENGINEERING PIPELINE")
    print("=" * 60)

    # Load
    final_rating = load_processed_data()

    # Create pivot table
    book_pivot = create_pivot_table(final_rating)

    # Create sparse matrix
    sparse_matrix = create_sparse_matrix(book_pivot)

    # Extract book names
    book_names = extract_book_names(book_pivot)

    if save:
        DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        pickle.dump(book_pivot, open(BOOK_PIVOT_PATH, "wb"))
        pickle.dump(book_names, open(BOOK_NAMES_PATH, "wb"))
        print(f"\n💾 Saved features to {DATA_PROCESSED_DIR}")

    print("=" * 60)
    return book_pivot, sparse_matrix, book_names


if __name__ == "__main__":
    engineer_features()
