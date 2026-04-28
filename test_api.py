import pickle
import pandas as pd

def test_model_prediction():
    """Simple test that directly uses the model (works reliably in CI)"""
    # Load the model produced by DVC
    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)

    # Test input
    test_input = {
        "Pregnancies": 6,
        "Glucose": 148,
        "BloodPressure": 72,
        "SkinThickness": 35,
        "Insulin": 0,
        "BMI": 33.6,
        "DiabetesPedigreeFunction": 0.627,
        "Age": 50
    }

    df = pd.DataFrame([test_input])
    prediction = int(model.predict(df)[0])
    probability = float(model.predict_proba(df)[0][1])

    print(f"✅ Prediction: {prediction} (0 = no diabetes, 1 = diabetes)")
    print(f"✅ Probability of diabetes: {probability:.4f}")

    assert prediction in [0, 1]
    print("✅ Model prediction test passed!")

if __name__ == "__main__":
    test_model_prediction()
