from fastapi import FastAPI
import pickle
import pandas as pd
from pydantic import BaseModel
from typing import Dict

app = FastAPI(title="Diabetes Classifier API")

# Load the model produced by DVC
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

@app.post("/predict")
def predict(input_data: DiabetesInput) -> Dict:
    df = pd.DataFrame([input_data.dict()])
    prediction = int(model.predict(df)[0])
    probability = float(model.predict_proba(df)[0][1])

    return {
        "prediction": prediction,           # 0 = no diabetes, 1 = diabetes
        "probability_diabetes": probability,
        "message": "1 = Diabetes risk, 0 = No diabetes"
    }

# Run locally with: uvicorn app:app --reload --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
