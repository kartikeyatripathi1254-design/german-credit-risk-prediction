"""
api.py - Production-ready FastAPI scoring service for German Credit Risk Prediction.
Run with: uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""
from typing import List, Literal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
import os

app = FastAPI(
    title="German Credit Risk Scoring API",
    description="Cost-sensitive real-time credit default risk prediction API.",
    version="1.0.0"
)

# Applicant Input Schema
class Applicant(BaseModel):
    status: str = Field(..., example="A11", description="Status of existing checking account")
    duration: int = Field(..., example=24, description="Duration in months")
    credit_history: str = Field(..., example="A34", description="Credit history")
    purpose: str = Field(..., example="A40", description="Purpose of the loan")
    credit_amount: float = Field(..., example=3000.0, description="Credit amount in DM")
    savings: str = Field(..., example="A61", description="Savings account/bonds")
    employment_duration: str = Field(..., example="A73", description="Present employment since")
    installment_rate: int = Field(..., example=4, description="Installment rate in % of disposable income")
    personal_status_sex: str = Field(..., example="A93", description="Personal status and sex")
    other_debtors: str = Field(..., example="A101", description="Other debtors / guarantors")
    present_residence: int = Field(..., example=4, description="Present residence since")
    property: str = Field(..., example="A121", description="Property")
    age: int = Field(..., example=35, description="Age in years")
    other_installment_plans: str = Field(..., example="A143", description="Other installment plans")
    housing: str = Field(..., example="A152", description="Housing status")
    number_credits: int = Field(..., example=1, description="Number of existing credits at this bank")
    job: str = Field(..., example="A173", description="Job classification")
    people_liable: int = Field(..., example=1, description="Number of people liable to maintain")
    telephone: str = Field(..., example="A192", description="Telephone registered")
    foreign_worker: str = Field(..., example="A201", description="Foreign worker status")

class PredictionResponse(BaseModel):
    default_probability: float
    risk_level: Literal["Low Risk (Approve)", "High Risk (Reject)"]
    optimal_threshold_used: float

# In-memory artifact storage
MODEL_ARTIFACT = None
OPTIMAL_THRESHOLD = 0.35  # Cost-optimal threshold derived from 5:1 penalty

@app.on_event("startup")
def load_model():
    global MODEL_ARTIFACT
    model_path = os.getenv("MODEL_PATH", "models/best_model.joblib")
    if os.path.exists(model_path):
        MODEL_ARTIFACT = joblib.load(model_path)
    else:
        # Fallback to credit_model pipeline if joblib not yet serialized
        from credit_model import build_and_train_pipeline
        MODEL_ARTIFACT = build_and_train_pipeline()

@app.get("/health", tags=["Monitoring"])
def health_check():
    return {"status": "healthy", "model_loaded": MODEL_ARTIFACT is not None}

@app.post("/predict", response_model=PredictionResponse, tags=["Scoring"])
def predict_applicant(applicant: Applicant):
    if MODEL_ARTIFACT is None:
        raise HTTPException(status_code=503, detail="Model is not initialized.")

    input_df = pd.DataFrame([applicant.model_dump()])
    probability = float(MODEL_ARTIFACT.predict_proba(input_df)[:, 1][0])
    
    decision = "High Risk (Reject)" if probability >= OPTIMAL_THRESHOLD else "Low Risk (Approve)"

    return PredictionResponse(
        default_probability=round(probability, 4),
        risk_level=decision,
        optimal_threshold_used=OPTIMAL_THRESHOLD
    )

@app.post("/predict/batch", response_model=List[PredictionResponse], tags=["Scoring"])
def predict_batch(applicants: List[Applicant]):
    if MODEL_ARTIFACT is None:
        raise HTTPException(status_code=503, detail="Model is not initialized.")

    input_df = pd.DataFrame([a.model_dump() for a in applicants])
    probabilities = MODEL_ARTIFACT.predict_proba(input_df)[:, 1]

    results = []
    for prob in probabilities:
        decision = "High Risk (Reject)" if prob >= OPTIMAL_THRESHOLD else "Low Risk (Approve)"
        results.append(
            PredictionResponse(
                default_probability=round(float(prob), 4),
                risk_level=decision,
                optimal_threshold_used=OPTIMAL_THRESHOLD
            )
        )
    return results
