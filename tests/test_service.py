import pytest
pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from cns_tinker.service import create_app

TOKEN = "test-only-secret-never-deploy-00000000"


def test_fail_closed_and_capabilities():
    with pytest.raises(RuntimeError):
        create_app("short")
    client = TestClient(create_app(TOKEN))
    assert client.get("/healthz").status_code == 200
    assert client.get("/v1/capabilities").status_code == 401
    result = client.get("/v1/capabilities", headers={"Authorization": f"Bearer {TOKEN}"})
    assert result.json()["training"] is False
    assert result.json()["malecns_mounted"] is False


def test_run_executes_and_preserves_truth_and_time():
    client = TestClient(create_app(TOKEN))
    headers = {"Authorization": f"Bearer {TOKEN}"}
    assert client.post("/v1/runs", json={"recipe": "../../private"}, headers=headers).status_code == 422
    assert client.post("/v1/runs", json={"recipe": "loom", "code": "x"}, headers=headers).status_code == 422
    result = client.post("/v1/runs", json={"recipe": "loom"}, headers=headers)
    assert result.status_code == 200
    data = result.json()
    assert data["mode"] == "executed-synthetic-preview"
    assert "placeholder" in data["manifest"]["connectome"]["connectome_dataset"]
    signal = data["signal"]
    assert len(signal["t_ms"]) == len(signal["activity"]) == len(data["actions"])
    assert all(b > a for a, b in zip(signal["t_ms"], signal["t_ms"][1:]))
    assert len(signal["activity"][0]) == 512
    assert data["recipe"] == "loom"
    assert len(data["stimuli"]) == len(signal["t_ms"])
    assert [float(s["t_ms"]) for s in data["stimuli"]] == signal["t_ms"]
    assert max(float(s["expansion_rate"]) for s in data["stimuli"]) > 0.9


def test_timeout_releases_runner_and_hides_process_details(monkeypatch):
    import subprocess
    import cns_tinker.service as service
    client = TestClient(create_app(TOKEN))
    headers = {"Authorization": f"Bearer {TOKEN}"}
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("private-path", 45)
    monkeypatch.setattr(service.subprocess, "run", timeout)
    for _ in range(2):
        response = client.post("/v1/runs", json={"recipe": "loom"}, headers=headers)
        assert response.status_code == 504
        assert "private-path" not in response.text
