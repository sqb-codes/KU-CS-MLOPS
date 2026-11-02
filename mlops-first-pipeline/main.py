from pathlib import Path
import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field
import mlflow
import mlflow.sklearn
import os

app = FastAPI(title="Iris Model API")

# MLFlow Configuration
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI", "http://127.0.0.1:6001"
)

# Model URI: models:/iris-classifer@champion
MODEL_URI = "models:/iris-classifier@champion"

# Tell the MLFlow client where the model registry lives
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# model_path = Path("models/iris_model.joblib")
# model = joblib.load(model_path)

try:
    model = mlflow.sklearn.load_model(MODEL_URI)
    client = mlflow.MlflowClient()
    champion_version = client.get_model_version_by_alias(
        name="iris-classifier",
        alias="champion"
    )
    model_version = champion_version.version
    print(f"Loaded model: iris-classifier:{model_version}")
except Exception as ex:
    raise RuntimeError(
        f"Could not load champion model from MLFlow: {ex}"
    )


class IrisRequest(BaseModel):
    sepal_length: float = Field(gt=0)
    sepal_width: float = Field(gt=0)
    petal_length: float = Field(gt=0)
    petal_width: float = Field(gt=0)

@app.get("/")
def health_check():
    return {"message": "API is ready to serve..."}

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_name": "iris-classifier",
        "model_version": model_version
        }


@app.post("/predict")
def predict(data: IrisRequest):
    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    prediction = int(model.predict(features)[0])

    return {
        "prediction": prediction,
        "class_name": ["setosa", "versicolor", "virginica"][prediction]
    }

