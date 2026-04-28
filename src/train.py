from dotenv import load_dotenv
load_dotenv()

import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score

# Explicit MLflow setup
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("diabetes-classifier")

def train(input_path, output_path, random_state, n_estimators, max_depth):
    data = pd.read_csv(input_path)
    X = data.drop(columns=['Outcome'])
    y = data['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    rf = RandomForestClassifier(random_state=random_state)
    grid_search = GridSearchCV(rf, param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"INFO: Best Model Accuracy: {accuracy}")

    with mlflow.start_run() as run:
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(best_model, "model")
        print(f"🏃 View run {run.info.run_name} at: {mlflow.get_tracking_uri()}/#/experiments/0/runs/{run.info.run_id}")

    with open(output_path, "wb") as f:
        pickle.dump(best_model, f)
    print(f"INFO: Model saved to {output_path}")

if __name__ == "__main__":
    import yaml
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(
        params['train']['input'],
        params['train']['output'],
        params['train']['random_state'],
        params['train']['n_estimators'],
        params['train']['max_depth']
    )
