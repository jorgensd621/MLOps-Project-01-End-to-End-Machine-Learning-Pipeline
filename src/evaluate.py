from dotenv import load_dotenv
load_dotenv()

import os
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score
import mlflow

# MLflow optional (full locally, skips in CI)
if os.getenv("MLFLOW_TRACKING_URI"):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

def evaluate(input_path, model_path):
    data = pd.read_csv(input_path)
    X = data.drop(columns=['Outcome'])
    y = data['Outcome']

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    print(f"INFO: Evaluated Model Accuracy: {accuracy}")

    # Log to MLflow only if configured
    if os.getenv("MLFLOW_TRACKING_URI"):
        with mlflow.start_run():
            mlflow.log_metric("accuracy", accuracy)

if __name__ == "__main__":
    import yaml
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    evaluate(params['train']['input'], params['train']['output'])
