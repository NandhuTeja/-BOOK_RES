"""
Model Monitoring Module
───────────────────────
Tracks predictions, popular recommendations, and usage stats.
Stores logs in a SQLite database.
"""

import sys
import sqlite3
import json
from datetime import datetime
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import MONITORING_DB


class MonitoringService:
    """Lightweight monitoring for the recommendation system."""

    def __init__(self, db_path=None):
        self.db_path = str(db_path or MONITORING_DB)
        self._init_db()

    def _init_db(self):
        """Initialize the SQLite monitoring database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                query_book TEXT NOT NULL,
                recommended_books TEXT NOT NULL,
                model_version TEXT,
                response_time_ms REAL
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                details TEXT
            )
        """)

        conn.commit()
        conn.close()

    def log_prediction(self, query_book, recommended_books, model_version="unknown", response_time_ms=0):
        """Log a prediction event."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO predictions (timestamp, query_book, recommended_books, model_version, response_time_ms) VALUES (?, ?, ?, ?, ?)",
            (
                datetime.now().isoformat(),
                query_book,
                json.dumps(recommended_books),
                model_version,
                response_time_ms
            )
        )
        conn.commit()
        conn.close()

    def log_event(self, event_type, details=None):
        """Log a model lifecycle event (training, deployment, etc.)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO model_events (timestamp, event_type, details) VALUES (?, ?, ?)",
            (datetime.now().isoformat(), event_type, json.dumps(details) if details else None)
        )
        conn.commit()
        conn.close()

    def get_stats(self):
        """Get monitoring statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Total predictions
        cursor.execute("SELECT COUNT(*) FROM predictions")
        total_predictions = cursor.fetchone()[0]

        # Predictions today
        today = datetime.now().strftime("%Y-%m-%d")
        cursor.execute("SELECT COUNT(*) FROM predictions WHERE timestamp LIKE ?", (f"{today}%",))
        predictions_today = cursor.fetchone()[0]

        # Most queried books
        cursor.execute("SELECT query_book, COUNT(*) as cnt FROM predictions GROUP BY query_book ORDER BY cnt DESC LIMIT 10")
        top_queries = cursor.fetchall()

        # Most recommended books
        cursor.execute("SELECT recommended_books FROM predictions")
        all_recs = cursor.fetchall()
        rec_counter = Counter()
        for row in all_recs:
            books = json.loads(row[0])
            rec_counter.update(books)
        top_recommended = rec_counter.most_common(10)

        # Average response time
        cursor.execute("SELECT AVG(response_time_ms) FROM predictions WHERE response_time_ms > 0")
        avg_response = cursor.fetchone()[0] or 0

        # Recent predictions
        cursor.execute(
            "SELECT timestamp, query_book, model_version FROM predictions ORDER BY id DESC LIMIT 5"
        )
        recent = cursor.fetchall()

        conn.close()

        return {
            "total_predictions": total_predictions,
            "predictions_today": predictions_today,
            "top_queried_books": top_queries,
            "top_recommended_books": top_recommended,
            "avg_response_time_ms": round(avg_response, 2),
            "recent_predictions": recent,
        }


if __name__ == "__main__":
    monitor = MonitoringService()
    stats = monitor.get_stats()
    print("📊 Monitoring Stats:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
