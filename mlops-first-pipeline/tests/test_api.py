# Import TestClient so we can send simulated HTTP requests
from fastapi.testclient import TestClient

# Import application object defined in main.py
from main import create_app


class FakeIrisModel:
    def predict(self, features):
        # Always return class 0
        return [0]


def fake_model_loader():
    return FakeIrisModel()


app = create_app(model_loader=fake_model_loader)