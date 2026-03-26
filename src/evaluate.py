"""
Evaluation Module
─────────────────
Provides evaluation metrics for the recommendation system:
Precision@K, Recall@K, NDCG@K, Hit Rate, and Coverage.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import EVAL_K_VALUES


def precision_at_k(recommended: list, relevant: set, k: int) -> float:
    """
    Precision@K: What fraction of the top-K recommendations are relevant?

    Args:
        recommended: List of recommended item titles (ordered).
        relevant: Set of ground-truth relevant item titles.
        k: Number of top recommendations to consider.
    """
    top_k = recommended[:k]
    hits = len(set(top_k) & relevant)
    return hits / k if k > 0 else 0.0


def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    """
    Recall@K: What fraction of relevant items appear in the top-K?

    Args:
        recommended: List of recommended item titles (ordered).
        relevant: Set of ground-truth relevant item titles.
        k: Number of top recommendations to consider.
    """
    top_k = recommended[:k]
    hits = len(set(top_k) & relevant)
    return hits / len(relevant) if len(relevant) > 0 else 0.0


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    """
    Normalized Discounted Cumulative Gain @ K.

    Args:
        recommended: List of recommended item titles (ordered).
        relevant: Set of ground-truth relevant item titles.
        k: Number of top recommendations to consider.
    """
    top_k = recommended[:k]
    dcg = sum(
        1.0 / np.log2(i + 2) for i, item in enumerate(top_k) if item in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def hit_rate(recommended: list, relevant: set) -> float:
    """
    Hit Rate: Is at least one relevant item in the recommendations?

    Returns 1.0 if there's a hit, 0.0 otherwise.
    """
    return 1.0 if len(set(recommended) & relevant) > 0 else 0.0


def coverage(all_recommendations: list, total_items: int) -> float:
    """
    Coverage: What fraction of the catalogue is being recommended?

    Args:
        all_recommendations: Flat list of all recommended items across all queries.
        total_items: Total number of unique items in the catalogue.
    """
    unique_recommended = len(set(all_recommendations))
    return unique_recommended / total_items if total_items > 0 else 0.0


def evaluate_model(model, book_pivot, test_pairs, k_values=None):
    """
    Evaluate the model on test data using multiple metrics.

    Args:
        model: Trained KNN model.
        book_pivot: User-book pivot table.
        test_pairs: List of (book_title, set_of_relevant_books) tuples.
        k_values: List of K values to evaluate at.

    Returns:
        Dictionary of aggregated metrics.
    """
    if k_values is None:
        k_values = EVAL_K_VALUES

    results = {f"precision@{k}": [] for k in k_values}
    results.update({f"recall@{k}": [] for k in k_values})
    results.update({f"ndcg@{k}": [] for k in k_values})
    results["hit_rate"] = []
    all_recs = []

    book_titles = list(book_pivot.index)

    for query_book, relevant_books in test_pairs:
        if query_book not in book_titles:
            continue

        # Get recommendations
        book_idx = book_titles.index(query_book)
        try:
            distances, indices = model.kneighbors(
                book_pivot.iloc[book_idx, :].values.reshape(1, -1),
                n_neighbors=max(k_values) + 1
            )
        except Exception:
            continue

        # Skip the query book itself (first result)
        rec_titles = [book_titles[i] for i in indices[0][1:]]
        all_recs.extend(rec_titles)

        for k in k_values:
            results[f"precision@{k}"].append(precision_at_k(rec_titles, relevant_books, k))
            results[f"recall@{k}"].append(recall_at_k(rec_titles, relevant_books, k))
            results[f"ndcg@{k}"].append(ndcg_at_k(rec_titles, relevant_books, k))

        results["hit_rate"].append(hit_rate(rec_titles, relevant_books))

    # Aggregate metrics
    metrics = {}
    for key, values in results.items():
        if values:
            metrics[key] = float(np.mean(values))
        else:
            metrics[key] = 0.0

    metrics["coverage"] = coverage(all_recs, len(book_titles))
    metrics["num_test_queries"] = len(test_pairs)

    return metrics


def print_metrics(metrics: dict):
    """Pretty-print evaluation metrics."""
    print("\n📊 EVALUATION METRICS")
    print("─" * 40)
    for key, value in sorted(metrics.items()):
        if isinstance(value, float):
            print(f"   {key:20s}: {value:.4f}")
        else:
            print(f"   {key:20s}: {value}")
    print("─" * 40)
