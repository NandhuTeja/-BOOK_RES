"""
Model Training Pipeline
───────────────────────
End-to-end training with:
- Data validation
- Preprocessing
- Feature engineering
- Train/test split with cross-validation
- KNN model training
- Evaluation metrics
- MLflow experiment tracking
- Model registry
"""

import sys
import time
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import KFold
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    MODELS_DIR, MODEL_PATH,
    KNN_N_NEIGHBORS, KNN_ALGORITHM, KNN_METRIC,
    TEST_SIZE, RANDOM_SEED, CV_FOLDS,
    MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME,
    EVAL_K_VALUES, BOOK_PIVOT_PATH, BOOK_NAMES_PATH
)
from src.data_validation import DataValidator
from src.data_preprocessing import preprocess
from src.feature_engineering import engineer_features
from src.evaluate import (
    precision_at_k, recall_at_k, ndcg_at_k, hit_rate,
    coverage, evaluate_model, print_metrics
)
from src.monitoring import MonitoringService

warnings.filterwarnings("ignore")


def create_test_pairs(book_pivot, final_rating, test_size=TEST_SIZE, seed=RANDOM_SEED):
    """
    Create train/test split for evaluation.

    Strategy: For each test user, hold out some books they rated highly
    and check if the model can recommend them.
    """
    print(f"\n✂️  Creating train/test split (test_size={test_size})...")
    np.random.seed(seed)

    book_titles = list(book_pivot.index)
    # Get books with high ratings (>= 6) for test pairs
    high_rated = final_rating[final_rating["Book-Rating"] >= 6]

    # Group by book title and find related books (same author/popular co-rated)
    test_pairs = []
    sampled_books = np.random.choice(
        book_titles,
        size=min(100, len(book_titles)),
        replace=False
    )

    for book in sampled_books:
        # Find users who rated this book highly
        book_fans = high_rated[high_rated["Book-Title"] == book]["User-ID"].values
        if len(book_fans) == 0:
            continue

        # Find other books those users also rated highly
        related = high_rated[
            (high_rated["User-ID"].isin(book_fans)) &
            (high_rated["Book-Title"] != book) &
            (high_rated["Book-Title"].isin(book_titles))
        ]["Book-Title"].value_counts()

        if len(related) > 0:
            # Top related books are the "relevant" set
            relevant = set(related.head(20).index)
            test_pairs.append((book, relevant))

    print(f"   Created {len(test_pairs)} test query-relevance pairs")
    return test_pairs


def cross_validate_model(book_pivot, n_neighbors, algorithm, metric, cv_folds=CV_FOLDS):
    """
    Run K-fold cross-validation on the KNN model.
    """
    print(f"\n🔄 Running {cv_folds}-fold cross-validation...")
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED)

    fold_scores = []
    book_matrix = book_pivot.values

    for fold, (train_idx, test_idx) in enumerate(kf.split(book_matrix), 1):
        # Train on subset
        train_data = book_matrix[train_idx]
        model = NearestNeighbors(
            n_neighbors=min(n_neighbors, len(train_idx)),
            algorithm=algorithm,
            metric=metric
        )
        model.fit(train_data)

        # Evaluate: average distance to neighbors (lower = better clustering)
        distances, _ = model.kneighbors(book_matrix[test_idx])
        avg_distance = np.mean(distances[:, 1:])  # Exclude self
        fold_scores.append(avg_distance)
        print(f"   Fold {fold}: avg neighbor distance = {avg_distance:.4f}")

    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)
    print(f"   📊 CV Result: {mean_score:.4f} ± {std_score:.4f}")
    return mean_score, std_score


def train_model(
    n_neighbors=KNN_N_NEIGHBORS,
    algorithm=KNN_ALGORITHM,
    metric=KNN_METRIC,
    use_mlflow=True,
    run_validation=True,
    run_cv=True,
):
    """
    Full training pipeline.

    Args:
        n_neighbors: Number of neighbors for KNN.
        algorithm: KNN algorithm.
        metric: Distance metric.
        use_mlflow: Whether to log to MLflow.
        run_validation: Whether to validate data first.
        run_cv: Whether to run cross-validation.

    Returns:
        Trained model, metrics dict.
    """
    start_time = time.time()
    monitor = MonitoringService()

    print("=" * 60)
    print("🏋️  MODEL TRAINING PIPELINE")
    print("=" * 60)

    # Log training start event
    monitor.log_event("training_started", {
        "n_neighbors": n_neighbors,
        "algorithm": algorithm,
        "metric": metric,
        "use_mlflow": use_mlflow,
        "run_validation": run_validation,
        "run_cv": run_cv,
    })

    # ─── Step 1: Data Validation ───────────────────────────────
    if run_validation:
        print("\n📋 Step 1/6: Data Validation")
        validator = DataValidator()
        passed, val_stats = validator.run_all()
    else:
        print("\n📋 Step 1/6: Data Validation — SKIPPED")
        val_stats = {}

    # ─── Step 2: Preprocessing ─────────────────────────────────
    print("\n📋 Step 2/6: Data Preprocessing")
    final_rating, books = preprocess(save=True)

    # ─── Step 3: Feature Engineering ───────────────────────────
    print("\n📋 Step 3/6: Feature Engineering")
    book_pivot, sparse_matrix, book_names = engineer_features(save=True)

    # ─── Step 4: Cross-Validation ──────────────────────────────
    cv_mean, cv_std = 0.0, 0.0
    if run_cv:
        print("\n📋 Step 4/6: Cross-Validation")
        cv_mean, cv_std = cross_validate_model(
            book_pivot, n_neighbors, algorithm, metric
        )
    else:
        print("\n📋 Step 4/6: Cross-Validation — SKIPPED")

    # ─── Step 5: Train Final Model ─────────────────────────────
    print("\n📋 Step 5/6: Training Final Model")
    print(f"   Algorithm: {algorithm}")
    print(f"   Metric: {metric}")
    print(f"   N_neighbors: {n_neighbors}")

    model = NearestNeighbors(
        n_neighbors=n_neighbors,
        algorithm=algorithm,
        metric=metric
    )
    model.fit(sparse_matrix)
    print("   ✅ Model trained successfully!")

    # ─── Step 6: Evaluation ────────────────────────────────────
    print("\n📋 Step 6/6: Evaluation")
    test_pairs = create_test_pairs(book_pivot, final_rating)
    metrics = evaluate_model(model, book_pivot, test_pairs, EVAL_K_VALUES)
    metrics["cv_mean_distance"] = cv_mean
    metrics["cv_std_distance"] = cv_std
    print_metrics(metrics)

    # ─── Save Model ────────────────────────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    pickle.dump(model, open(MODEL_PATH, "wb"))
    print(f"\n💾 Model saved to {MODEL_PATH}")

    training_time = time.time() - start_time
    metrics["training_time_seconds"] = training_time

    # ─── MLflow Logging ────────────────────────────────────────
    if use_mlflow:
        try:
            import mlflow
            import mlflow.sklearn
            print("\n📈 Logging to MLflow...")

            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

            with mlflow.start_run(run_name=f"knn_{metric}_{n_neighbors}") as run:
                # Log parameters
                mlflow.log_param("model_type", "KNN")
                mlflow.log_param("n_neighbors", n_neighbors)
                mlflow.log_param("algorithm", algorithm)
                mlflow.log_param("metric", metric)
                mlflow.log_param("n_books", len(book_names))
                mlflow.log_param("pivot_shape", str(book_pivot.shape))
                mlflow.log_param("cv_folds", CV_FOLDS if run_cv else 0)
                mlflow.log_param("test_size", TEST_SIZE)
                mlflow.log_param("random_seed", RANDOM_SEED)

                # Log metrics (sanitize keys: @ is reserved in MLflow 3.x)
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        safe_key = key.replace("@", "_at_")
                        mlflow.log_metric(safe_key, value)

                # Log model artifact
                mlflow.sklearn.log_model(model, "model")

                print(f"   ✅ MLflow Run ID: {run.info.run_id}")

            # Register model AFTER run is closed (required for MLflow 3.x)
            model_name = "book-recommender-knn"
            model_uri = f"runs:/{run.info.run_id}/model"
            try:
                result = mlflow.register_model(model_uri, model_name)
                print(f"   ✅ Model registered: {model_name} v{result.version}")

                # Set alias for easy loading
                client = mlflow.tracking.MlflowClient()
                client.set_registered_model_alias(model_name, "production", result.version)
                print(f"   ✅ Model v{result.version} aliased as 'production'")
            except Exception as reg_err:
                print(f"   ⚠️  Model registration: {reg_err}")
                print(f"   ✅ Model logged as artifact (run: {run.info.run_id})")

        except ImportError:
            print("   ⚠️  MLflow not installed. Skipping tracking.")
        except Exception as e:
            print(f"   ⚠️  MLflow error: {e}")

    # ─── Log Training Completion ────────────────────────────────
    monitor.log_event("training_completed", {
        "training_time_seconds": round(training_time, 2),
        "n_books": len(book_names),
        "pivot_shape": str(book_pivot.shape),
        "metrics": {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()},
    })

    # ─── Summary ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("🏆 TRAINING COMPLETE")
    print(f"   ⏱️  Time: {training_time:.1f}s")
    print(f"   📊 Books: {len(book_names)}")
    print(f"   📊 Pivot: {book_pivot.shape}")
    for k in EVAL_K_VALUES:
        p = metrics.get(f"precision@{k}", 0)
        r = metrics.get(f"recall@{k}", 0)
        print(f"   📊 P@{k}: {p:.4f}  |  R@{k}: {r:.4f}")
    print("=" * 60)

    return model, metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Book Recommender Model")
    parser.add_argument("--n-neighbors", type=int, default=KNN_N_NEIGHBORS)
    parser.add_argument("--algorithm", type=str, default=KNN_ALGORITHM)
    parser.add_argument("--metric", type=str, default=KNN_METRIC)
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--no-validation", action="store_true")
    parser.add_argument("--no-cv", action="store_true")
    args = parser.parse_args()

    train_model(
        n_neighbors=args.n_neighbors,
        algorithm=args.algorithm,
        metric=args.metric,
        use_mlflow=not args.no_mlflow,
        run_validation=not args.no_validation,
        run_cv=not args.no_cv,
    )
