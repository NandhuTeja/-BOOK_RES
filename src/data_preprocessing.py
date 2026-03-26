"""
Data Preprocessing Module
─────────────────────────
Loads raw data, merges datasets, filters users/books
by activity thresholds, and saves processed data.
"""

import sys
import pickle
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    BOOKS_CSV, RATINGS_CSV,
    DATA_PROCESSED_DIR,
    PROCESSED_RATINGS, PROCESSED_BOOKS, FINAL_RATING_PATH,
    MIN_USER_RATINGS, MIN_BOOK_RATINGS
)


def load_raw_data():
    """Load raw CSV files."""
    print("📂 Loading raw data...")
    books = pd.read_csv(
        BOOKS_CSV, sep=";", on_bad_lines="skip", encoding="latin-1", low_memory=False
    )
    ratings = pd.read_csv(
        RATINGS_CSV, sep=";", on_bad_lines="skip", encoding="latin-1"
    )
    print(f"   Loaded {books.shape[0]:,} books and {ratings.shape[0]:,} ratings")
    return books, ratings


def clean_books(books: pd.DataFrame) -> pd.DataFrame:
    """Select relevant columns and clean the books dataframe."""
    print("🧹 Cleaning books data...")
    books = books[["ISBN", "Book-Title", "Book-Author", "Year-Of-Publication", "Publisher", "Image-URL-L"]]
    books.drop_duplicates(subset="ISBN", inplace=True)
    print(f"   {books.shape[0]:,} unique books after deduplication")
    return books


def filter_active_users(ratings: pd.DataFrame) -> pd.DataFrame:
    """Filter to keep only users with >= MIN_USER_RATINGS ratings."""
    print(f"👤 Filtering users with ≥{MIN_USER_RATINGS} ratings...")
    user_counts = ratings.groupby("User-ID").count()["Book-Rating"]
    active_users = user_counts[user_counts >= MIN_USER_RATINGS].index
    filtered = ratings[ratings["User-ID"].isin(active_users)]
    print(f"   {len(active_users):,} active users → {filtered.shape[0]:,} ratings")
    return filtered


def filter_popular_books(ratings: pd.DataFrame) -> pd.DataFrame:
    """Filter to keep only books with >= MIN_BOOK_RATINGS ratings."""
    print(f"📚 Filtering books with ≥{MIN_BOOK_RATINGS} ratings...")
    book_counts = ratings.groupby("Book-Title").count()["Book-Rating"]
    popular_books = book_counts[book_counts >= MIN_BOOK_RATINGS].index
    filtered = ratings[ratings["Book-Title"].isin(popular_books)]
    print(f"   {len(popular_books):,} popular books → {filtered.shape[0]:,} ratings")
    return filtered


def preprocess(save=True):
    """Run the full preprocessing pipeline."""
    print("=" * 60)
    print("🔧 DATA PREPROCESSING PIPELINE")
    print("=" * 60)

    # Load
    books, ratings = load_raw_data()

    # Clean books
    books = clean_books(books)

    # Merge ratings with book info
    print("🔗 Merging ratings with book data...")
    ratings_with_books = ratings.merge(books, on="ISBN")
    print(f"   Merged dataset: {ratings_with_books.shape[0]:,} rows")

    # Filter active users
    filtered = filter_active_users(ratings_with_books)

    # Filter popular books
    final_rating = filter_popular_books(filtered)

    # Remove duplicates (user-book pairs)
    final_rating.drop_duplicates(subset=["User-ID", "Book-Title"], inplace=True)
    print(f"\n✅ Final dataset: {final_rating.shape[0]:,} ratings")
    print(f"   Unique users: {final_rating['User-ID'].nunique():,}")
    print(f"   Unique books: {final_rating['Book-Title'].nunique():,}")

    if save:
        DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        pickle.dump(final_rating, open(FINAL_RATING_PATH, "wb"))
        pickle.dump(books, open(PROCESSED_BOOKS, "wb"))
        print(f"\n💾 Saved processed data to {DATA_PROCESSED_DIR}")

    print("=" * 60)
    return final_rating, books


if __name__ == "__main__":
    preprocess()
