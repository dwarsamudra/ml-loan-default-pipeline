"""
train.py - End-to-End ML Pipeline: Training Script
Author: Anvesh Dubey
Description: Trains a loan default prediction model with full preprocessing,
             feature engineering, model selection, evaluation and artifact saving.
"""

import os
import pandas as pd
import numpy as np
import joblib
import json
import warnings
from datetime import datetime

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, f1_score
)

import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# STEP 1: Generate / Load Data
# ─────────────────────────────────────────────

def generate_sample_data(n=2000, seed=42):
    """Generate synthetic loan default dataset."""
    np.random.seed(seed)
    ages = np.random.randint(22, 65, n)
    incomes = np.random.randint(20000, 150000, n)
    loan_amounts = np.random.randint(5000, 80000, n)
    credit_scores = np.random.randint(300, 850, n)
    employment = np.random.choice(["employed", "self-employed", "unemployed"], n, p=[0.65, 0.25, 0.10])
    education = np.random.choice(["graduate", "undergraduate", "high_school"], n, p=[0.4, 0.4, 0.2])
    loan_purpose = np.random.choice(["home", "car", "education", "personal"], n)
    num_prev_loans = np.random.randint(0, 6, n)
    months_employed = np.random.randint(0, 300, n)

    # Default probability influenced by features
    default_prob = (
        0.10
        + 0.15 * (loan_amounts / incomes > 0.8).astype(float)
        + 0.20 * (credit_scores < 500).astype(float)
        - 0.10 * (credit_scores > 750).astype(float)
        + 0.15 * (employment == "unemployed").astype(float)
        + 0.05 * (num_prev_loans > 3).astype(float)
    )
    default_prob = np.clip(default_prob, 0.02, 0.95)
    default = np.random.binomial(1, default_prob)

    df = pd.DataFrame({
        "age": ages,
        "income": incomes,
        "loan_amount": loan_amounts,
        "credit_score": credit_scores,
        "employment_status": employment,
        "education": education,
        "loan_purpose": loan_purpose,
        "num_prev_loans": num_prev_loans,
        "months_employed": months_employed,
        "default": default
    })

    # Inject some missing values to simulate real data
    for col in ["income", "credit_score", "months_employed"]:
        df.loc[df.sample(frac=0.03, random_state=seed).index, col] = np.nan

    return df


# ─────────────────────────────────────────────
# STEP 2: Feature Engineering
# ─────────────────────────────────────────────

def feature_engineering(df):
    """Add derived features to improve model performance."""
    df = df.copy()
    df["loan_to_income_ratio"] = df["loan_amount"] / (df["income"] + 1)
    df["credit_score_bucket"] = pd.cut(
        df["credit_score"], bins=[0, 500, 650, 750, 1000],
        labels=["poor", "fair", "good", "excellent"]
    )
    df["is_high_loan"] = (df["loan_to_income_ratio"] > 0.5).astype(int)
    df["employment_years"] = df["months_employed"] / 12
    return df


# ─────────────────────────────────────────────
# STEP 3: Build Preprocessing Pipeline
# ─────────────────────────────────────────────

def build_preprocessor(num_features, cat_features):
    """Create sklearn ColumnTransformer for numeric and categorical features."""
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, num_features),
        ("cat", categorical_transformer, cat_features)
    ])
    return preprocessor


# ─────────────────────────────────────────────
# STEP 4: Train & Evaluate Models
# ─────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Print evaluation metrics for a trained model."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"\n{'='*50}")
    print(f"  {model_name}")
    print(f"{'='*50}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['No Default', 'Default'])}")

    return {"accuracy": acc, "f1": f1, "roc_auc": auc}


def plot_confusion_matrix(model, X_test, y_test, model_name, save_path):
    """Save confusion matrix plot."""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Default", "Default"],
                yticklabels=["No Default", "Default"])
    plt.title(f"Confusion Matrix — {model_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"  Saved confusion matrix → {save_path}")


def plot_feature_importance(model, feature_names, save_path):
    """Save feature importance plot (for tree-based models)."""
    clf = model.named_steps["classifier"]
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
        indices = np.argsort(importances)[::-1][:15]
        top_features = [feature_names[i] for i in indices]
        top_importances = importances[indices]

        plt.figure(figsize=(8, 5))
        sns.barplot(x=top_importances, y=top_features, palette="Blues_r")
        plt.title("Top 15 Feature Importances")
        plt.xlabel("Importance Score")
        plt.tight_layout()
        plt.savefig(save_path, dpi=120)
        plt.close()
        print(f"  Saved feature importance → {save_path}")


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  END-TO-END ML PIPELINE: Loan Default Prediction")
    print(f"  Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # ── 1. Data
    print("\n[1/6] Generating dataset...")
    os.makedirs("data", exist_ok=True)
    df = generate_sample_data(n=2000)
    df.to_csv("data/loan_data.csv", index=False)
    print(f"  Rows: {len(df)} | Default rate: {df['default'].mean():.2%}")

    # ── 2. Feature Engineering
    print("\n[2/6] Feature engineering...")
    df = feature_engineering(df)

    # ── 3. Split
    print("\n[3/6] Splitting data...")
    TARGET = "default"
    DROP_COLS = [TARGET, "credit_score_bucket"]  # drop target + ordinal bucket (encoded below)
    X = df.drop(columns=DROP_COLS)
    y = df[TARGET]

    NUM_FEATURES = ["age", "income", "loan_amount", "credit_score",
                    "num_prev_loans", "months_employed",
                    "loan_to_income_ratio", "is_high_loan", "employment_years"]
    CAT_FEATURES = ["employment_status", "education", "loan_purpose"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

    # ── 4. Preprocessor
    print("\n[4/6] Building preprocessor & training models...")
    preprocessor = build_preprocessor(NUM_FEATURES, CAT_FEATURES)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    results = {}
    trained_models = {}

    for name, clf in models.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ])
        pipeline.fit(X_train, y_train)
        metrics = evaluate_model(pipeline, X_test, y_test, name)
        results[name] = metrics
        trained_models[name] = pipeline

    # ── 5. Select Best Model
    print("\n[5/6] Selecting best model by ROC-AUC...")
    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    best_model = trained_models[best_name]
    print(f"  Best model: {best_name} (AUC = {results[best_name]['roc_auc']:.4f})")

    # ── 6. Save Artifacts
    print("\n[6/6] Saving model artifacts...")
    os.makedirs("models", exist_ok=True)
    os.makedirs("static", exist_ok=True)

    joblib.dump(best_model, "models/best_model.pkl")
    print("  Saved → models/best_model.pkl")

    # Save feature list for API
    feature_meta = {
        "num_features": NUM_FEATURES,
        "cat_features": CAT_FEATURES,
        "all_features": NUM_FEATURES + CAT_FEATURES,
        "best_model": best_name,
        "metrics": results[best_name],
        "trained_at": datetime.now().isoformat()
    }
    with open("models/feature_meta.json", "w") as f:
        json.dump(feature_meta, f, indent=2)
    print("  Saved → models/feature_meta.json")

    # Save plots
    ohe_features = list(
        best_model.named_steps["preprocessor"]
        .named_transformers_["cat"]
        .named_steps["onehot"]
        .get_feature_names_out(CAT_FEATURES)
    )
    all_feature_names = NUM_FEATURES + list(ohe_features)

    plot_confusion_matrix(best_model, X_test, y_test, best_name, "static/confusion_matrix.png")
    plot_feature_importance(best_model, all_feature_names, "static/feature_importance.png")

    # Save results summary
    with open("models/results_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n✅ Pipeline complete! All artifacts saved.")
    print(f"   Model: models/best_model.pkl")
    print(f"   Meta:  models/feature_meta.json")
    print(f"   Plots: static/\n")


if __name__ == "__main__":
    main()
