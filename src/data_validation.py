"""
Data Validation Module
─────────────────────
Validates raw data quality before it enters the pipeline.
Checks for: missing values, duplicates, rating ranges, dataset shape.
"""

import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import BOOKS_CSV, RATINGS_CSV, USERS_CSV, RATING_MIN, RATING_MAX


class DataValidator:
    """Validates the Book-Crossing dataset before training."""

    def __init__(self):
        self.issues = []
        self.stats = {}

    def load_datasets(self):
        """Load all three raw CSV files."""
        print("📂 Loading raw datasets...")
        self.books = pd.read_csv(
            BOOKS_CSV, sep=";", on_bad_lines="skip", encoding="latin-1", low_memory=False
        )
        self.ratings = pd.read_csv(
            RATINGS_CSV, sep=";", on_bad_lines="skip", encoding="latin-1"
        )
        self.users = pd.read_csv(
            USERS_CSV, sep=";", on_bad_lines="skip", encoding="latin-1"
        )
        print(f"   Books:   {self.books.shape[0]:,} rows × {self.books.shape[1]} cols")
        print(f"   Ratings: {self.ratings.shape[0]:,} rows × {self.ratings.shape[1]} cols")
        print(f"   Users:   {self.users.shape[0]:,} rows × {self.users.shape[1]} cols")

    def check_missing_values(self):
        """Check for missing values in all datasets."""
        print("\n🔍 Checking missing values...")
        for name, df in [("Books", self.books), ("Ratings", self.ratings), ("Users", self.users)]:
            missing = df.isnull().sum()
            total_missing = missing.sum()
            if total_missing > 0:
                msg = f"   ⚠️  {name}: {total_missing} missing values"
                print(msg)
                self.issues.append(msg)
                for col, count in missing[missing > 0].items():
                    print(f"      - {col}: {count} missing ({count/len(df)*100:.2f}%)")
            else:
                print(f"   ✅ {name}: No missing values")
            self.stats[f"{name}_missing"] = total_missing

    def check_duplicate_isbns(self):
        """Check for duplicate ISBNs in the books dataset."""
        print("\n🔍 Checking duplicate ISBNs...")
        if "ISBN" in self.books.columns:
            duplicates = self.books["ISBN"].duplicated().sum()
            if duplicates > 0:
                msg = f"   ⚠️  {duplicates} duplicate ISBNs found"
                print(msg)
                self.issues.append(msg)
            else:
                print("   ✅ No duplicate ISBNs")
            self.stats["duplicate_isbns"] = duplicates
        else:
            print("   ❌ ISBN column not found")
            self.issues.append("ISBN column missing from Books dataset")

    def check_rating_ranges(self):
        """Validate that ratings are within expected range."""
        print("\n🔍 Checking rating ranges...")
        if "Book-Rating" in self.ratings.columns:
            ratings = self.ratings["Book-Rating"]
            min_r, max_r = ratings.min(), ratings.max()
            out_of_range = ((ratings < RATING_MIN) & (ratings != 0)).sum()

            # Note: 0 means implicit (not rated), valid range is 1-10 for explicit
            explicit_ratings = ratings[ratings > 0]
            invalid = explicit_ratings[(explicit_ratings < RATING_MIN) | (explicit_ratings > RATING_MAX)]

            print(f"   Rating range: [{min_r}, {max_r}]")
            print(f"   Explicit ratings (1-10): {len(explicit_ratings):,}")
            print(f"   Implicit ratings (0): {(ratings == 0).sum():,}")

            if len(invalid) > 0:
                msg = f"   ⚠️  {len(invalid)} ratings outside valid range [{RATING_MIN}-{RATING_MAX}]"
                print(msg)
                self.issues.append(msg)
            else:
                print(f"   ✅ All explicit ratings within [{RATING_MIN}-{RATING_MAX}]")

            self.stats["total_ratings"] = len(ratings)
            self.stats["explicit_ratings"] = len(explicit_ratings)
            self.stats["implicit_ratings"] = (ratings == 0).sum()
        else:
            print("   ❌ Book-Rating column not found")
            self.issues.append("Book-Rating column missing from Ratings dataset")

    def check_dataset_shapes(self):
        """Verify dataset shapes are reasonable."""
        print("\n🔍 Checking dataset shapes...")
        checks = [
            ("Books", self.books, 100000),
            ("Ratings", self.ratings, 100000),
            ("Users", self.users, 50000),
        ]
        for name, df, min_rows in checks:
            if len(df) < min_rows:
                msg = f"   ⚠️  {name} has only {len(df):,} rows (expected ≥{min_rows:,})"
                print(msg)
                self.issues.append(msg)
            else:
                print(f"   ✅ {name}: {len(df):,} rows — OK")

    def validate_referential_integrity(self):
        """Check that ISBNs and User-IDs in ratings exist in books/users."""
        print("\n🔍 Checking referential integrity...")
        if "ISBN" in self.ratings.columns and "ISBN" in self.books.columns:
            book_isbns = set(self.books["ISBN"])
            rating_isbns = set(self.ratings["ISBN"])
            orphan_isbns = rating_isbns - book_isbns
            if orphan_isbns:
                msg = f"   ⚠️  {len(orphan_isbns):,} ISBNs in ratings not found in books"
                print(msg)
                self.issues.append(msg)
            else:
                print("   ✅ All rating ISBNs exist in books")

        if "User-ID" in self.ratings.columns and "User-ID" in self.users.columns:
            user_ids = set(self.users["User-ID"])
            rating_user_ids = set(self.ratings["User-ID"])
            orphan_users = rating_user_ids - user_ids
            if orphan_users:
                msg = f"   ⚠️  {len(orphan_users):,} User-IDs in ratings not found in users"
                print(msg)
                self.issues.append(msg)
            else:
                print("   ✅ All rating User-IDs exist in users")

    def run_all(self):
        """Run all validation checks."""
        print("=" * 60)
        print("🛡️  DATA VALIDATION REPORT")
        print("=" * 60)

        self.load_datasets()
        self.check_missing_values()
        self.check_duplicate_isbns()
        self.check_rating_ranges()
        self.check_dataset_shapes()
        self.validate_referential_integrity()

        print("\n" + "=" * 60)
        if self.issues:
            print(f"⚠️  VALIDATION COMPLETE — {len(self.issues)} issue(s) found:")
            for i, issue in enumerate(self.issues, 1):
                print(f"   {i}. {issue.strip()}")
        else:
            print("✅ VALIDATION COMPLETE — All checks passed!")
        print("=" * 60)

        return len(self.issues) == 0, self.stats


if __name__ == "__main__":
    validator = DataValidator()
    passed, stats = validator.run_all()
    if not passed:
        print("\n⚠️  Data has issues but pipeline can continue with warnings.")
    sys.exit(0)
