from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_read_model_metadata():
    response = client.get("/model_metadata")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert data["model_name"] == "Text Classifier"

def test_predict_unauthorized():
    response = client.post("/predict", json={"text": "sample text"})
    assert response.status_code == 401

