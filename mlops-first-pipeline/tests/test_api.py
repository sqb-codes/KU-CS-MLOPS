# Import TestClient so we can send simulated HTTP requests
from fastapi.testclient import TestClient

# Import application object defined in main.py
from main import create_app
import pytest


class FakeIrisModel:
    def predict(self, features):
        # Always return class 0
        return [0]


def fake_model_loader():
    return FakeIrisModel()

def fake_version_loader():
    return "test-version"


@pytest.fixture
def client():
    app = create_app(
        model_loader=fake_model_loader,
        version_loader=fake_version_loader
    )

    with TestClient(app) as test_client:
        yield test_client

def test_health_endpoint():
    """
    Verify that the application is ruuning and can respond to a health-check request
    """

    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}
    assert response.json()["model_version"] == "test-version"

def test_valid_iris_prediction():
    """
    Verify that we are getting valid predictions
    """
    payload = {
        "sepal_length": 3.4,
        "sepal_width": 1.4,
        "petal_length": 3.4,
        "petal_width": 0.4
    }

    # Send a simulated POST request with JSON input
    response = client.post("/predict", json=payload)

    # Endpoint should accept the valid input
    assert response.status_code == 200

    result = response.json()
    assert "prediction" in result
    assert "class_name" in result

    assert result["class_name"] in ["setosa", "versicolor", "virginica"]
    assert result["prediction"] in [0, 1, 2]
