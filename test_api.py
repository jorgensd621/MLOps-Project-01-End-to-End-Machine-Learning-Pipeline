import subprocess
import time
import requests
import sys

print("Starting API test...")

# Start the API in background
proc = subprocess.Popen([
    "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"
], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

time.sleep(5)  # Wait for server to start

try:
    response = requests.post(
        "http://localhost:8000/predict",
        json={
            "Pregnancies": 6,
            "Glucose": 148,
            "BloodPressure": 72,
            "SkinThickness": 35,
            "Insulin": 0,
            "BMI": 33.6,
            "DiabetesPedigreeFunction": 0.627,
            "Age": 50
        },
        timeout=10
    )
    print("API Response:", response.json())
    assert response.status_code == 200
    print("✅ API test passed!")
except Exception as e:
    print("❌ API test failed:", e)
    sys.exit(1)
finally:
    proc.terminate()
