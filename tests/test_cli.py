from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from distillery.cli import app
from distillery.io import iter_jsonl, read_json

ROOT = Path(__file__).resolve().parent.parent
runner = CliRunner()


@pytest.fixture
def demo_config(tmp_path: Path) -> Path:
    config = yaml.safe_load((ROOT / "configs/demo.yaml").read_text())
    config["output_dir"] = str(tmp_path / "run")
    config["teacher"]["cache_dir"] = str(tmp_path / "cache")
    config["generate"]["num_tickets"] = 120
    config["eval"]["gold_path"] = str(ROOT / "data/gold/gold.jsonl")
    path = tmp_path / "demo.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_run_all_with_fake_provider(demo_config: Path, tmp_path: Path) -> None:
    result = runner.invoke(app, ["run-all", "-c", str(demo_config)])
    assert result.exit_code == 0, result.output
    run = tmp_path / "run"
    for name in [
        "generated.jsonl",
        "labeled.jsonl",
        "filtered.jsonl",
        "funnel.json",
        "dataset/train.jsonl",
        "dataset/val.jsonl",
        "dataset/README.md",
        "eval/report.json",
        "report.html",
    ]:
        assert (run / name).exists(), name
    funnel = read_json(run / "funnel.json")
    assert [step["name"] for step in funnel][:3] == ["requested", "generated", "length"]
    report = read_json(run / "eval/report.json")
    assert set(report["systems"]) == {"teacher", "student_finetuned"}
    assert report["gold_items"] == 80
    assert "router" in report and 0 <= report["router"]["gold"]["offload_rate"] <= 1
    first = next(iter_jsonl(run / "dataset/train.jsonl"))
    assert [message["role"] for message in first["messages"]] == ["system", "user", "assistant"]
    json.loads(first["messages"][2]["content"])
    assert "Gold-set comparison" in (run / "report.html").read_text()
    assert not (run / "calls.jsonl").exists()  # the fake provider makes no real API calls


def test_stages_individually_and_rerun_is_deterministic(demo_config: Path, tmp_path: Path) -> None:
    for command in ["generate", "label", "filter", "build"]:
        result = runner.invoke(app, [command, "-c", str(demo_config)])
        assert result.exit_code == 0, result.output
    first = (tmp_path / "run/dataset/train.jsonl").read_text()
    for command in ["generate", "label", "filter", "build"]:
        runner.invoke(app, [command, "-c", str(demo_config)])
    assert (tmp_path / "run/dataset/train.jsonl").read_text() == first


def test_stage_without_inputs_fails_clearly(demo_config: Path) -> None:
    result = runner.invoke(app, ["filter", "-c", str(demo_config)])
    assert result.exit_code != 0
    assert "run the previous stage first" in str(result.exception)


def test_train_with_fake_backend_is_a_noop(demo_config: Path) -> None:
    result = runner.invoke(app, ["train", "-c", str(demo_config)])
    assert result.exit_code == 0 and "nothing to train" in result.output


def test_show_config(demo_config: Path) -> None:
    result = runner.invoke(app, ["show-config", "-c", str(demo_config)])
    assert result.exit_code == 0 and '"require_free_models": false' in result.output


def test_export_copies_shareable_artifacts(demo_config: Path, tmp_path: Path) -> None:
    assert runner.invoke(app, ["run-all", "-c", str(demo_config)]).exit_code == 0
    destination = tmp_path / "results"
    result = runner.invoke(app, ["export", str(destination), "-c", str(demo_config)])
    assert result.exit_code == 0, result.output
    assert (destination / "eval/report.json").exists()
    assert (destination / "dataset/train.jsonl").exists()
    assert (destination / "funnel.json").exists()
    assert not (destination / "labeled.jsonl").exists()


def test_generate_top_up_pass_appends_and_counts_in_funnel(demo_config: Path, tmp_path: Path) -> None:
    assert runner.invoke(app, ["generate", "-c", str(demo_config)]).exit_code == 0
    result = runner.invoke(
        app, ["generate", "-c", str(demo_config), "--append", "--num", "10", "--topic", "other", "--prefix", "t"]
    )
    assert result.exit_code == 0, result.output
    rows = list(iter_jsonl(tmp_path / "run/generated.jsonl"))
    top_up = [row for row in rows if row["id"].startswith("t")]
    assert len(rows) == 130 and len(top_up) == 10
    assert {row["seed"]["topic"] for row in top_up} == {"other"}
    manifest = read_json(tmp_path / "run/generation.json")
    assert [item["requested"] for item in manifest["passes"]] == [120, 10]
    duplicate = runner.invoke(app, ["generate", "-c", str(demo_config), "--append", "--num", "5", "--prefix", "t"])
    assert duplicate.exit_code != 0
    for command in ["label", "filter"]:
        assert runner.invoke(app, [command, "-c", str(demo_config)]).exit_code == 0
    assert read_json(tmp_path / "run/funnel.json")[0] == {
        "name": "requested",
        "description": "Seed specs sent to the generator",
        "kept": 130,
        "dropped": 0,
        "modified": 0,
        "reasons": {},
    }
