"""
Prediction / Recommendation Module
───────────────────────────────────
Handles loading the trained model and generating recommendations.
Supports loading from MLflow registry or local pickle.
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    MODEL_PATH, BOOK_PIVOT_PATH, BOOK_NAMES_PATH,
    PROCESSED_BOOKS, FINAL_RATING_PATH,
    BOOKS_CSV, N_RECOMMENDATIONS,
    MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME
)


class BookRecommender:
    """Production-ready book recommendation engine."""

    def __init__(self, use_mlflow=True):
        self.model = None
        self.book_pivot = None
        self.book_names = None
        self.books_df = None
        self.model_version = "unknown"
        self.model_metrics = {}
        self._load_model(use_mlflow)
        self._load_data()

    def _load_model(self, use_mlflow):
        """Load model from MLflow registry or fallback to pickle."""
        if use_mlflow:
            try:
                import mlflow
                mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
                client = mlflow.tracking.MlflowClient()

                model_name = "book-recommender-knn"

                # Try to load model by 'production' alias (MLflow 3.x)
                try:
                    model_uri = f"models:/{model_name}@production"
                    self.model = mlflow.sklearn.load_model(model_uri)
                    mv = client.get_model_version_by_alias(model_name, "production")
                    self.model_version = f"v{mv.version} (production)"
                    run = client.get_run(mv.run_id)
                    self.model_metrics = run.data.metrics
                    print(f"✅ Loaded model from MLflow: {self.model_version}")
                    return
                except Exception:
                    pass

                # Try loading latest version by searching model versions
                try:
                    versions = client.search_model_versions(f"name='{model_name}'")
                    if versions:
                        latest = sorted(versions, key=lambda v: int(v.version), reverse=True)[0]
                        model_uri = f"models:/{model_name}/{latest.version}"
                        self.model = mlflow.sklearn.load_model(model_uri)
                        self.model_version = f"v{latest.version}"
                        run = client.get_run(latest.run_id)
                        self.model_metrics = run.data.metrics
                        print(f"✅ Loaded model from MLflow: {self.model_version}")
                        return
                except Exception:
                    pass

            except ImportError:
                print("⚠️  MLflow not available, falling back to pickle")
            except Exception as e:
                print(f"⚠️  MLflow load failed ({e}), falling back to pickle")

        # Fallback: load from pickle
        self._load_from_pickle()

    def _load_from_pickle(self):
        """Load model from local pickle files."""
        print("📦 Loading model from pickle...")

        # Try new location first, then legacy location
        model_path = MODEL_PATH if MODEL_PATH.exists() else Path("model.pkl")
        if model_path.exists():
            self.model = pickle.load(open(model_path, "rb"))
            self.model_version = "pickle (legacy)"
            print(f"   Loaded from {model_path}")
        else:
            raise FileNotFoundError(f"No model found at {MODEL_PATH} or model.pkl")

    def _load_data(self):
        """Load pivot table, book names, and books dataframe."""
        # Try new paths first, then legacy
        pivot_path = BOOK_PIVOT_PATH if BOOK_PIVOT_PATH.exists() else Path("book_pivot.pkl")
        names_path = BOOK_NAMES_PATH if BOOK_NAMES_PATH.exists() else Path("book_names.pkl")

        self.book_pivot = pickle.load(open(pivot_path, "rb"))
        self.book_names = pickle.load(open(names_path, "rb"))

        # Load books dataframe for images
        if PROCESSED_BOOKS.exists():
            self.books_df = pickle.load(open(PROCESSED_BOOKS, "rb"))
        else:
            self.books_df = pd.read_csv(
                BOOKS_CSV, sep=";", on_bad_lines="skip", encoding="latin-1"
            )

        if "Image-URL-L" not in self.books_df.columns and "Image-URL-L" in self.books_df.columns:
            pass  # already has the column
        self.books_df = self.books_df[["Book-Title", "Image-URL-L"]].drop_duplicates(subset="Book-Title")
        print(f"   Loaded {len(self.book_names)} books, pivot shape: {self.book_pivot.shape}")

    def recommend(self, book_title: str, n: int = N_RECOMMENDATIONS) -> list:
        """
        Get book recommendations.

        Args:
            book_title: The title of the book to base recommendations on.
            n: Number of recommendations to return.

        Returns:
            List of dicts with 'title', 'image_url', 'distance'.
        """
        if book_title not in self.book_pivot.index:
            raise ValueError(f"Book '{book_title}' not found in the dataset.")

        book_index = np.where(self.book_pivot.index == book_title)[0][0]
        distances, suggestions = self.model.kneighbors(
            self.book_pivot.iloc[book_index, :].values.reshape(1, -1),
            n_neighbors=n + 1  # +1 because the book itself is included
        )

        recommendations = []
        for idx in range(1, len(suggestions[0])):  # Skip first (self)
            rec_title = self.book_pivot.index[suggestions[0][idx]]
            rec_distance = distances[0][idx]
            image_url = self._get_image(rec_title)

            recommendations.append({
                "title": rec_title,
                "image_url": image_url,
                "distance": float(rec_distance),
            })

        return recommendations

    def _get_image(self, book_title: str) -> str:
        """Get book cover image URL."""
        try:
            return self.books_df[
                self.books_df["Book-Title"] == book_title
            ]["Image-URL-L"].values[0]
        except (IndexError, KeyError):
            return "https://via.placeholder.com/150"

    def get_model_info(self) -> dict:
        """Return model metadata for display."""
        info = {
            "version": self.model_version,
            "n_books": len(self.book_names),
            "pivot_shape": f"{self.book_pivot.shape[0]} × {self.book_pivot.shape[1]}",
        }
        if self.model_metrics:
            info["metrics"] = self.model_metrics
        return info


if __name__ == "__main__":
    rec = BookRecommender(use_mlflow=False)
    info = rec.get_model_info()
    print(f"\nModel Info: {info}")

    test_book = rec.book_names[0]
    print(f"\nRecommendations for '{test_book}':")
    results = rec.recommend(test_book)
    for r in results:
        print(f"  📖 {r['title']} (distance: {r['distance']:.4f})")
