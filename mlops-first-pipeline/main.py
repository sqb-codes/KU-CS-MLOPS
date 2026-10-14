from pathlib import Path
import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Iris Model API")

model_path = Path("models/iris_model.joblib")
model = joblib.load(model_path)

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
    return {"status": "healthy"}


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

