from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from distillery.cli import app as cli_app
from distillery.config import Settings
from distillery.serve import create_app

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def client(tmp_path_factory: pytest.TempPathFactory) -> TestClient:
    tmp_path = tmp_path_factory.mktemp("serve")
    config = yaml.safe_load((ROOT / "configs/demo.yaml").read_text())
    config["output_dir"] = str(tmp_path / "run")
    config["generate"]["num_tickets"] = 60
    config["eval"]["gold_path"] = str(ROOT / "data/gold/gold.jsonl")
    config_path = tmp_path / "demo.yaml"
    config_path.write_text(yaml.safe_dump(config))
    result = CliRunner().invoke(cli_app, ["run-all", "-c", str(config_path)])
    assert result.exit_code == 0, result.output
    settings = Settings(_env_file=None, distillery_config=config_path)  # type: ignore[call-arg]
    return TestClient(create_app(settings))


def test_health(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["student_backend"] == "fake"
    assert body["teacher"] == "fake/fake-teacher-a"
    assert 0 <= body["threshold"] <= 1.01


def test_triage_returns_validated_answer(client: TestClient) -> None:
    response = client.post("/triage", json={"text": "I was charged twice for order BL-503318, please fix it"})
    assert response.status_code == 200
    body = response.json()
    assert body["source"] in {"student", "teacher"}
    assert body["triage"]["order_id"] == "BL-503318"
    assert set(body["student_field_confidence"]) == {"intent", "urgency", "sentiment", "product_area", "order_id"}


def test_triage_validates_input(client: TestClient) -> None:
    assert client.post("/triage", json={"text": ""}).status_code == 422


def test_dashboard_pages(client: TestClient) -> None:
    overview = client.get("/")
    assert overview.status_code == 200
    assert "Gold-set comparison" in overview.text and "Filter funnel" in overview.text
    playground = client.get("/playground")
    assert playground.status_code == 200 and "Triage with both" in playground.text
    assert client.get("/api/report").json()["gold_items"] == 80


def test_playground_htmx_partial_and_full_page(client: TestClient) -> None:
    text = "My Sentry camera is offline and my house is unmonitored"
    partial = client.post("/playground", data={"text": text}, headers={"HX-Request": "true"})
    assert partial.status_code == 200
    assert "Router decision" in partial.text and "<html" not in partial.text
    full = client.post("/playground", data={"text": text})
    assert "<html" in full.text and "Router decision" in full.text
