# 🏦 End-to-End ML Pipeline — Loan Default Prediction

> **Author:** Anvesh Dubey
> **Tech Stack:** Python · scikit-learn · FastAPI · Docker · GitHub Actions
> **Target Roles:** ML Engineer · Data Scientist · Python Developer (6–12 LPA)

---

## 📌 Project Overview

A **production-grade, end-to-end machine learning pipeline** that predicts whether a loan applicant will default. This project demonstrates the full ML lifecycle — from raw data ingestion and EDA to model training, evaluation, REST API deployment, containerisation and CI/CD.

This is the kind of project interviewers at 6–12 LPA companies **actually want to see** on your GitHub.

---

## 🎯 What This Project Covers (Skills Demonstrated)

| Skill Area | What's Shown |
|---|---|
| **Data Engineering** | Data generation, cleaning, missing-value imputation |
| **EDA** | Distribution plots, correlation heatmap, categorical analysis |
| **Feature Engineering** | Derived features (loan-to-income ratio, risk buckets) |
| **ML Pipeline** | `sklearn.Pipeline` + `ColumnTransformer` for clean, leak-free preprocessing |
| **Model Comparison** | Logistic Regression vs Random Forest vs Gradient Boosting |
| **Evaluation** | Accuracy, F1, ROC-AUC, Confusion Matrix, Feature Importance |
| **Model Serialization** | `joblib` for saving/loading artifacts |
| **REST API** | FastAPI with Pydantic validation, batch endpoint, health check |
| **Containerization** | Dockerfile — build and run in one command |
| **Testing** | `pytest` unit tests for data, features, preprocessor, and API schema |
| **CI/CD** | GitHub Actions — auto-test on every push, Docker build on `main` |

---

## 📁 Project Structure

```
ml-loan-default-pipeline/
│
├── src/
│   └── train.py              # Full training pipeline (run this first)
│
├── api/
│   └── app.py                # FastAPI REST API
│
├── notebooks/
│   └── eda.py                # Exploratory Data Analysis script
│
├── tests/
│   └── test_pipeline.py      # pytest unit tests
│
├── models/                   # Generated after training
│   ├── best_model.pkl
│   ├── feature_meta.json
│   └── results_summary.json
│
├── data/                     # Generated after training
│   └── loan_data.csv
│
├── static/                   # Plots (generated)
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   └── eda_*.png
│
├── .github/
│   └── workflows/
│       └── ci.yml            # GitHub Actions CI/CD
│
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/YOUR_USERNAME/ml-loan-default-pipeline.git
cd ml-loan-default-pipeline

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python src/train.py
```

**Output:**
```
[1/6] Generating dataset...   Rows: 2000 | Default rate: 22.45%
[2/6] Feature engineering...
[3/6] Splitting data...        Train: 1600 | Test: 400
[4/6] Building preprocessor & training models...

==================================================
  Random Forest
==================================================
  Accuracy : 0.8575
  F1 Score : 0.6842
  ROC-AUC  : 0.9103

[5/6] Selecting best model by ROC-AUC...
  Best model: Random Forest (AUC = 0.9103)
[6/6] Saving model artifacts...
  Saved → models/best_model.pkl
  Saved → models/feature_meta.json
✅ Pipeline complete!
```

### 3. Run the EDA

```bash
python notebooks/eda.py
```

Plots saved to `static/`.

### 4. Start the API

```bash
uvicorn api.app:app --reload --port 8000
```

Open → **http://localhost:8000/docs** for the interactive Swagger UI.

### 5. Test the API

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 32,
    "income": 750000,
    "loan_amount": 300000,
    "credit_score": 680,
    "num_prev_loans": 1,
    "months_employed": 48,
    "employment_status": "employed",
    "education": "graduate",
    "loan_purpose": "home"
  }'
```

**Response:**
```json
{
  "prediction": 0,
  "prediction_label": "No Default",
  "default_probability": 0.1423,
  "risk_category": "Low Risk",
  "model_used": "Random Forest"
}
```

### 6. Run Tests

```bash
pytest tests/ -v
```

---

## 🐳 Docker

```bash
# Build
docker build -t loan-predictor .

# Run
docker run -p 8000:8000 loan-predictor

# API is now live at http://localhost:8000
```

---

## 📊 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/` | API info |
| `GET`  | `/health` | Health check + model info |
| `POST` | `/predict` | Single prediction |
| `POST` | `/predict/batch` | Batch predictions |
| `GET`  | `/model/info` | Model metadata & metrics |
| `GET`  | `/docs` | Interactive Swagger UI |

---

## 🧠 Model Details

### Features Used

| Feature | Type | Description |
|---------|------|-------------|
| `age` | Numeric | Applicant age |
| `income` | Numeric | Annual income |
| `loan_amount` | Numeric | Requested loan |
| `credit_score` | Numeric | Credit score (300–850) |
| `num_prev_loans` | Numeric | Number of past loans |
| `months_employed` | Numeric | Total employment duration |
| `loan_to_income_ratio` | Engineered | loan / income |
| `is_high_loan` | Engineered | ratio > 0.5 → 1 |
| `employment_years` | Engineered | months / 12 |
| `employment_status` | Categorical | employed / self-employed / unemployed |
| `education` | Categorical | graduate / undergraduate / high_school |
| `loan_purpose` | Categorical | home / car / education / personal |

### Models Compared

| Model | Accuracy | F1 | ROC-AUC |
|-------|----------|-----|---------|
| Logistic Regression | ~0.82 | ~0.60 | ~0.87 |
| Random Forest | ~0.86 | ~0.68 | ~0.91 |
| Gradient Boosting | ~0.85 | ~0.67 | ~0.90 |

*Best model selected automatically by ROC-AUC.*

---

## 📈 Results

### Confusion Matrix
![Confusion Matrix](static/confusion_matrix.png)

### Feature Importance
![Feature Importance](static/feature_importance.png)

---

## 🛠️ Tech Stack

```
Python 3.11
├── scikit-learn   — ML pipeline, preprocessing, models
├── pandas         — data manipulation
├── numpy          — numerical operations
├── matplotlib     — plotting
├── seaborn        — statistical visualization
├── joblib         — model serialization
├── FastAPI        — REST API framework
├── Pydantic       — input validation
├── uvicorn        — ASGI server
├── pytest         — unit testing
└── Docker         — containerization
```

---

## 🤝 Interview Talking Points

When interviewers ask about this project, highlight:

1. **Why Pipeline?** — Prevents data leakage; scaler fitted only on train split
2. **Why ROC-AUC over Accuracy?** — Imbalanced classes; AUC is threshold-independent
3. **Feature Engineering reasoning** — loan-to-income ratio directly encodes risk
4. **Pydantic validation** — catches bad inputs before they hit the model
5. **How to improve?** — Add SHAP explainability, A/B model versioning with MLflow, Kubernetes deploy

---

## 📄 License

MIT License — free to use, fork, and modify.

---

*Built by **Anvesh Dubey** as part of a Python portfolio for 6–12 LPA ML/Data roles.*
