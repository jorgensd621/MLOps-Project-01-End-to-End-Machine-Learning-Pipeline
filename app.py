import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict

app = FastAPI(
    title="Diabetes Risk Classifier",
    description="MLOps End-to-End Pipeline - Predict diabetes risk",
    version="1.0"
)

# Load model once at startup
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

class DiabetesInput(BaseModel):
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

@app.get("/health")
def health_check() -> Dict:
    """Health check endpoint for monitoring and Render"""
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict")
def predict(input_data: DiabetesInput) -> Dict:
    """Predict diabetes risk (1 = high risk, 0 = low risk)"""
    df = pd.DataFrame([input_data.dict()])
    
    prediction = int(model.predict(df)[0])
    probability = float(model.predict_proba(df)[0][1])
    
    return {
        "prediction": prediction,
        "probability_diabetes": probability,
        "message": "1 = Diabetes risk, 0 = No diabetes"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
