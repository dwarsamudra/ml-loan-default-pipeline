"""
tests/test_pipeline.py — Unit Tests
Author: Anvesh Dubey
Run with: pytest tests/ -v
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.train import generate_sample_data, feature_engineering, build_preprocessor


# ─────────────────────────────────────────────
# Data Generation Tests
# ─────────────────────────────────────────────

class TestDataGeneration:
    def test_shape(self):
        df = generate_sample_data(n=100)
        assert df.shape[0] == 100, "Row count mismatch"

    def test_columns_exist(self):
        df = generate_sample_data(n=100)
        expected = ["age", "income", "loan_amount", "credit_score",
                    "employment_status", "education", "loan_purpose",
                    "num_prev_loans", "months_employed", "default"]
        for col in expected:
            assert col in df.columns, f"Missing column: {col}"

    def test_target_is_binary(self):
        df = generate_sample_data(n=500)
        assert set(df["default"].unique()).issubset({0, 1}), "Target not binary"

    def test_age_range(self):
        df = generate_sample_data(n=500)
        assert df["age"].between(22, 65).all(), "Age out of range"

    def test_has_missing_values(self):
        df = generate_sample_data(n=1000)
        assert df.isnull().any().any(), "Expected some missing values"


# ─────────────────────────────────────────────
# Feature Engineering Tests
# ─────────────────────────────────────────────

class TestFeatureEngineering:
    def test_new_columns_added(self):
        df = generate_sample_data(n=100)
        df_eng = feature_engineering(df)
        assert "loan_to_income_ratio" in df_eng.columns
        assert "is_high_loan" in df_eng.columns
        assert "employment_years" in df_eng.columns

    def test_loan_to_income_positive(self):
        df = generate_sample_data(n=100)
        df_eng = feature_engineering(df)
        # Drop NaN rows (from injected missing values) before asserting
        col = df_eng["loan_to_income_ratio"].dropna()
        assert (col >= 0).all()

    def test_is_high_loan_binary(self):
        df = generate_sample_data(n=100)
        df_eng = feature_engineering(df)
        assert set(df_eng["is_high_loan"].unique()).issubset({0, 1})

    def test_employment_years_non_negative(self):
        df = generate_sample_data(n=100)
        df_eng = feature_engineering(df)
        col = df_eng["employment_years"].dropna()
        assert (col >= 0).all()


# ─────────────────────────────────────────────
# Preprocessor Tests
# ─────────────────────────────────────────────

class TestPreprocessor:
    def setup_method(self):
        df = generate_sample_data(n=200)
        self.df = feature_engineering(df)
        self.num_cols = ["age", "income", "loan_amount", "credit_score",
                         "num_prev_loans", "months_employed",
                         "loan_to_income_ratio", "is_high_loan", "employment_years"]
        self.cat_cols = ["employment_status", "education", "loan_purpose"]

    def test_preprocessor_fits(self):
        preprocessor = build_preprocessor(self.num_cols, self.cat_cols)
        X = self.df[self.num_cols + self.cat_cols]
        preprocessor.fit(X)  # should not raise

    def test_preprocessor_transforms(self):
        preprocessor = build_preprocessor(self.num_cols, self.cat_cols)
        X = self.df[self.num_cols + self.cat_cols]
        X_transformed = preprocessor.fit_transform(X)
        assert X_transformed.shape[0] == len(self.df)
        assert not np.isnan(X_transformed).any(), "NaNs remain after preprocessing"


# ─────────────────────────────────────────────
# API Schema Tests (no server needed)
# ─────────────────────────────────────────────

class TestAPISchema:
    def test_valid_input(self):
        """Import Pydantic schema and validate a correct input."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))
        from app import LoanApplication
        app = LoanApplication(
            age=30, income=600000, loan_amount=200000,
            credit_score=700, num_prev_loans=1,
            months_employed=36, employment_status="employed",
            education="graduate", loan_purpose="home"
        )
        assert app.age == 30

    def test_invalid_credit_score(self):
        from pydantic import ValidationError
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))
        from app import LoanApplication
        with pytest.raises(ValidationError):
            LoanApplication(
                age=30, income=600000, loan_amount=200000,
                credit_score=9999,   # invalid: > 850
                num_prev_loans=1, months_employed=36,
                employment_status="employed",
                education="graduate", loan_purpose="home"
            )

    def test_invalid_employment_status(self):
        from pydantic import ValidationError
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))
        from app import LoanApplication
        with pytest.raises(ValidationError):
            LoanApplication(
                age=30, income=600000, loan_amount=200000,
                credit_score=700, num_prev_loans=1, months_employed=36,
                employment_status="astronaut",  # invalid
                education="graduate", loan_purpose="home"
            )
