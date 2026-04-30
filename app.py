import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict
from fastapi_mcp import FastApiMCP   # MCP support

app = FastAPI(
    title="Diabetes Risk Classifier",
    description="MLOps End-to-End Pipeline with MCP support",
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
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict")
def predict(input_data: DiabetesInput) -> Dict:
    df = pd.DataFrame([input_data.dict()])
    prediction = int(model.predict(df)[0])
    probability = float(model.predict_proba(df)[0][1])
    return {
        "prediction": prediction,
        "probability_diabetes": probability,
        "message": "1 = Diabetes risk, 0 = No diabetes"
    }

# ─────────────────────────────────────────────────────────────
# MCP Server (Model Context Protocol) - AI agents can now use this
# ─────────────────────────────────────────────────────────────
mcp = FastApiMCP(app)          # ← Correct positional usage
mcp.mount()                    # ← Mounts MCP server at /mcp

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
