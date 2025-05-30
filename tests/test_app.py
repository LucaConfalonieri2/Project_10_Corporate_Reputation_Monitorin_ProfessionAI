from fastapi.testclient import TestClient
from spaces.app import app
client = TestClient(app)

# Testa l'applicazione
def test_prediction():
    response = client.post("http://127.0.0.1:8000/predict", json={"text": "Great service!"})
    assert response.status_code == 200
    data = response.json()
    assert "label" in data
    assert "score" in data
    assert data["score"]>=0 and data["score"]<=1
    assert data["label"] in ["positive", "neutral", "negative"]




