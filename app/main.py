import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from app.schemas import FEATURES, HeartInput, PredictionOutput

MODEL_PATH = Path(os.getenv("MODEL_PATH", Path(__file__).resolve().parent.parent / "model" / "heart_model.joblib"))
state = {"model": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if MODEL_PATH.exists():
        state["model"] = joblib.load(MODEL_PATH)
    yield


app = FastAPI(
    title="Heart Disease Prediction API",
    description="Serves predictions from a classifier trained on the Heart Disease dataset.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {"message": "Heart Disease Prediction API. See /docs for Swagger UI."}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": state["model"] is not None}


@app.get("/info")
def info():
    model = state["model"]
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    estimator = model.steps[-1][1] if hasattr(model, "steps") else model
    return {
        "model_type": type(estimator).__name__,
        "features": FEATURES,
        "n_features": len(FEATURES),
        "target": "heart_disease (True = presence, False = absence)",
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(payload: HeartInput):
    model = state["model"]
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    row = pd.DataFrame([payload.model_dump()], columns=FEATURES)
    pred = int(model.predict(row)[0])
    proba = float(model.predict_proba(row)[0][1])
    return PredictionOutput(heart_disease=bool(pred), probability=round(proba, 4))
