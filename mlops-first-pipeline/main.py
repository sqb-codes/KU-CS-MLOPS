from pathlib import Path
import joblib
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
import mlflow
import mlflow.sklearn
import os
from contextlib import asynccontextmanager
from typing import Callable

# app = FastAPI(title="Iris Model API")

# MLFlow Configuration
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI", "http://127.0.0.1:6001"
)

# Model URI: models:/iris-classifer@champion
MODEL_URI = "models:/iris-classifier@champion"


def load_mlflow_model():
    # Tell the MLFlow client where the model registry lives
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return mlflow.sklearn.load_model(MODEL_URI)


class IrisRequest(BaseModel):
    sepal_length: float = Field(gt=0)
    sepal_width: float = Field(gt=0)
    petal_length: float = Field(gt=0)
    petal_width: float = Field(gt=0)


# Application Factory
def create_app(model_loader: Callable=load_mlflow_model, 
               version_loader: Callable=None):
    
    if version_loader is None:
        def version_loader():
            client = mlflow.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
            champion = client.get_model_version_by_alias(
                        name="iris-classifier",
                        alias="champion"
                    )
            return champion.version
    
    # lifespan function executes when the application starts...
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Load the model once during startup
        # Injected loader can be real MLFlow or a test model
        try:
            app.state.model = model_loader()
        except Exception as ex:
            raise RuntimeError(
                f"Failed to load MLFlow champion model..\n{ex}"
            )

        client = mlflow.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        champion = client.get_model_version_by_alias(
            name="iris-classifier",
            alias="champion"
        )
        app.state.model_version = version_loader()
        print(f"Loaded iris-classfier version {app.state.model_version}")
        yield

    app = FastAPI(
        title="Iris Classification API",
        description="Iris prediction API powered by MLFlow",
        version="3.0.0",
        lifespan=lifespan
    )

    @app.get("/")
    def health_check():
        return {"message": "API is ready to serve..."}

    @app.get("/health")
    def health_check():
        return {
            "status": "healthy",
            "model_name": "iris-classifier",
            "model_version": app.state.model_version
            }


    @app.post("/predict")
    def predict(data: IrisRequest, request: Request):
        model = request.app.state.model
        
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

app = create_app()
