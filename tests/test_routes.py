import pytest

from app import create_app
from app import routes


class FakeClassifier:
    metadata = {
        "model_name": "TestModel",
        "metrics": {"f1_weighted": 0.75},
    }
    label_map = {0: "INFORMATION-TECHNOLOGY", 1: "ENGINEERING"}

    def predict(self, resume_text):
        return {
            "error": False,
            "category": "INFORMATION-TECHNOLOGY",
            "confidence": 0.8,
            "top_k": [
                {"category": "INFORMATION-TECHNOLOGY", "confidence": 0.8}
            ],
        }


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(routes, "_classifier", FakeClassifier())
    app = create_app()
    app.config.update(TESTING=True)
    return app.test_client()


def test_health_reports_ready_model(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.get_json() == {"status": "ok", "model_ready": True}


@pytest.mark.parametrize("payload", [None, {}, {"other": "value"}])
def test_predict_requires_resume_text(client, payload):
    response = client.post("/predict", json=payload)

    assert response.status_code == 400
    assert response.get_json()["success"] is False


@pytest.mark.parametrize("value", [123, [], {}, None])
def test_predict_rejects_non_string_resume_text(client, value):
    response = client.post("/predict", json={"resume_text": value})

    assert response.status_code == 400
    assert response.get_json() == {
        "success": False,
        "error": "resume_text 必须是字符串。",
    }


def test_predict_rejects_short_resume(client):
    response = client.post("/predict", json={"resume_text": "Python developer"})

    assert response.status_code == 400


def test_predict_returns_classification(client):
    response = client.post(
        "/predict",
        json={"resume_text": "Python backend engineer " * 10},
    )

    assert response.status_code == 200
    assert response.get_json()["category"] == "INFORMATION-TECHNOLOGY"
