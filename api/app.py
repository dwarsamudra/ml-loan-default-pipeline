"""
api/app.py - FastAPI REST API for Loan Default Prediction
Author: Anvesh Dubey
Description: Serves the trained ML model via REST endpoints with
             input validation (Pydantic), JSON responses and a /health check.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Literal
import uvicorn

# ─────────────────────────────────────────────
# Load Model & Metadata
# ─────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")
META_PATH  = os.path.join(BASE_DIR, "models", "feature_meta.json")

model = None
feature_meta = None

def load_model():
    global model, feature_meta
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Please run `python src/train.py` first."
        )
    model = joblib.load(MODEL_PATH)
    with open(META_PATH) as f:
        feature_meta = json.load(f)
    print(f"✅ Model loaded: {feature_meta['best_model']}")
    print(f"   AUC: {feature_meta['metrics']['roc_auc']:.4f}")


# ─────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────

app = FastAPI(
    title="Loan Default Predictor API",
    description=(
        "End-to-End ML Pipeline: predicts whether a loan applicant will default. "
        "Built with scikit-learn + FastAPI. Author: Anvesh Dubey."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    load_model()


# ─────────────────────────────────────────────
# Pydantic Schemas
# ─────────────────────────────────────────────

class LoanApplication(BaseModel):
    age:               int   = Field(..., ge=18, le=80,       description="Applicant age in years")
    income:            float = Field(..., gt=0,               description="Annual income in INR/USD")
    loan_amount:       float = Field(..., gt=0,               description="Requested loan amount")
    credit_score:      float = Field(..., ge=300, le=850,     description="Credit score (300–850)")
    num_prev_loans:    int   = Field(..., ge=0, le=20,        description="Number of previous loans")
    months_employed:   float = Field(..., ge=0,               description="Total months in employment")
    employment_status: Literal["employed", "self-employed", "unemployed"] = "employed"
    education:         Literal["graduate", "undergraduate", "high_school"] = "graduate"
    loan_purpose:      Literal["home", "car", "education", "personal"] = "personal"

    class Config:
        schema_extra = {
            "example": {
                "age": 32,
                "income": 750000,
                "loan_amount": 300000,
                "credit_score": 680,
                "num_prev_loans": 1,
                "months_employed": 48,
                "employment_status": "employed",
                "education": "graduate",
                "loan_purpose": "home"
            }
        }


class PredictionResponse(BaseModel):
    prediction:        int
    prediction_label:  str
    default_probability: float
    risk_category:     str
    model_used:        str
    input_received:    dict


class BatchRequest(BaseModel):
    applications: list[LoanApplication]


# ─────────────────────────────────────────────
# Helper: preprocess input → DataFrame
# ─────────────────────────────────────────────

def application_to_df(app: LoanApplication) -> pd.DataFrame:
    data = app.dict()
    df = pd.DataFrame([data])

    # Feature engineering (must mirror train.py)
    df["loan_to_income_ratio"] = df["loan_amount"] / (df["income"] + 1)
    df["is_high_loan"]         = (df["loan_to_income_ratio"] > 0.5).astype(int)
    df["employment_years"]     = df["months_employed"] / 12

    return df


def get_risk_category(prob: float) -> str:
    if prob < 0.20:
        return "Low Risk"
    elif prob < 0.45:
        return "Moderate Risk"
    elif prob < 0.70:
        return "High Risk"
    else:
        return "Very High Risk"


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.get("/", tags=["Info"])
def root():
    return {
        "message": "Loan Default Predictor API is running 🚀",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict"
    }


@app.get("/health", tags=["Info"])
def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "healthy",
        "model": feature_meta["best_model"],
        "roc_auc": feature_meta["metrics"]["roc_auc"],
        "trained_at": feature_meta["trained_at"]
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(application: LoanApplication):
    """
    Predict whether a loan applicant will default.

    Returns:
    - **prediction**: 0 (no default) or 1 (default)
    - **default_probability**: probability of default (0–1)
    - **risk_category**: Low / Moderate / High / Very High
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    df = application_to_df(application)

    try:
        prediction   = int(model.predict(df)[0])
        default_prob = float(model.predict_proba(df)[0][1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    return PredictionResponse(
        prediction        = prediction,
        prediction_label  = "Default" if prediction == 1 else "No Default",
        default_probability = round(default_prob, 4),
        risk_category     = get_risk_category(default_prob),
        model_used        = feature_meta["best_model"],
        input_received    = application.dict()
    )


@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(batch: BatchRequest):
    """Predict for multiple applicants in one call."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    results = []
    for i, app in enumerate(batch.applications):
        df = application_to_df(app)
        prediction   = int(model.predict(df)[0])
        default_prob = float(model.predict_proba(df)[0][1])
        results.append({
            "index":               i,
            "prediction":          prediction,
            "prediction_label":    "Default" if prediction == 1 else "No Default",
            "default_probability": round(default_prob, 4),
            "risk_category":       get_risk_category(default_prob)
        })

    return {"count": len(results), "predictions": results}


@app.get("/model/info", tags=["Model"])
def model_info():
    """Get information about the currently loaded model."""
    if feature_meta is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return feature_meta


# ─────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
