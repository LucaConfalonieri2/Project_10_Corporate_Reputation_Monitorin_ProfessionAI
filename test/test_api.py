import requests

def test_prediction():
    response = requests.post("http://127.0.0.1:8000/predict", json={"text": "Great service!"})
    assert response.status_code == 200
    data = response.json()
    assert "label" in data
    assert "prob" in data
    assert isinstance(data["prob"], list)
    assert len(data["prob"]) == 3
