"""Command-line interface: one command per stage plus run-all, serve and helpers."""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from distillery.config import Settings, load_config
from distillery.io import iter_jsonl, read_json
from distillery.logging_config import configure_logging
from distillery.pipeline import Pipeline
from distillery.schema import Intent

app = typer.Typer(
    help="Distillery: distill a large LLM into a small model for support-ticket triage.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)
console = Console()

ConfigOption = Annotated[Path, typer.Option("--config", "-c", help="Pipeline YAML config.")]
DEFAULT_CONFIG = Path("configs/demo.yaml")
ALL_SYSTEMS = "teacher,student_zero_shot,student_finetuned"


def _pipeline(config_path: Path) -> Pipeline:
    settings = Settings()
    configure_logging(settings.log_level, settings.log_format)
    config = load_config(config_path)
    return Pipeline(config, settings)


def _print_funnel(pipeline: Pipeline) -> None:
    if not pipeline.paths.funnel.exists():
        return
    table = Table(title="Filter funnel")
    for column in ("step", "kept", "dropped", "details"):
        table.add_column(column, justify="right" if column in {"kept", "dropped"} else "left")
    for step in read_json(pipeline.paths.funnel):
        details = ", ".join(f"{key}={value}" for key, value in step["reasons"].items())
        if step.get("modified"):
            details = f"rewritten={step['modified']} " + details
        table.add_row(step["name"], str(step["kept"]), str(step["dropped"] or ""), details)
    console.print(table)


def _print_eval(report: dict[str, object]) -> None:
    systems = report.get("systems", {})
    assert isinstance(systems, dict)
    table = Table(title=f"Gold set: {report.get('gold_items')} hand-labeled tickets")
    for column in (
        "system",
        "JSON valid",
        "exact match",
        "intent",
        "urgency",
        "sentiment",
        "product",
        "order id",
        "intent F1",
        "s/ticket",
    ):
        table.add_column(column, justify="left" if column == "system" else "right")
    for name, scores in systems.items():
        accuracy = scores["field_accuracy"]
        table.add_row(
            name,
            f"{scores['json_valid_rate']:.0%}",
            f"{scores['exact_match']:.1%}",
            f"{accuracy['intent']:.1%}",
            f"{accuracy['urgency']:.1%}",
            f"{accuracy['sentiment']:.1%}",
            f"{accuracy['product_area']:.1%}",
            f"{accuracy['order_id']:.1%}",
            f"{scores['macro_f1']['intent']:.2f}",
            f"{scores['latency_s_per_item']['mean']:.2f}",
        )
    console.print(table)
    router = report.get("router")
    if isinstance(router, dict) and "gold" in router:
        gold = router["gold"]
        console.print(
            f"Router (threshold {router['threshold']:.2f}): student handles {gold['offload_rate']:.0%} of tickets, "
            f"routed exact match {gold['exact_match']:.1%}"
        )


@app.command()
def generate(
    config: ConfigOption = DEFAULT_CONFIG,
    append: Annotated[bool, typer.Option(help="Add a top-up pass to the existing generated.jsonl.")] = False,
    num: Annotated[int | None, typer.Option(help="Number of seeds (default: generate.num_tickets).")] = None,
    topic: Annotated[list[str] | None, typer.Option(help="Restrict seeds to these intents (repeatable).")] = None,
    prefix: Annotated[str, typer.Option(help="Seed id prefix; use a new one for each top-up pass.")] = "s",
) -> None:
    """Stage 1: generate synthetic tickets with the teacher (and/or import real ones)."""
    pipeline = _pipeline(config)
    topics = [Intent(value) for value in topic] if topic else None
    tickets = asyncio.run(pipeline.generate(append=append, num=num, topics=topics, prefix=prefix))
    console.print(f"[green]{len(tickets)} tickets[/] -> {pipeline.paths.generated}")


@app.command()
def label(config: ConfigOption = DEFAULT_CONFIG) -> None:
    """Stage 2: label every ticket with each teacher (validated, repaired, agreement measured)."""
    pipeline = _pipeline(config)
    labeled = asyncio.run(pipeline.label())
    stats = read_json(pipeline.paths.label_stats)
    console.print(f"[green]{len(labeled)} tickets labeled[/] -> {pipeline.paths.labeled}")
    if "all_fields_agreement" in stats:
        console.print(f"Teachers agree on all structured fields for {stats['all_fields_agreement']:.1%} of tickets")


@app.command(name="filter")
def filter_(config: ConfigOption = DEFAULT_CONFIG) -> None:
    """Stage 3: deterministic filters (validity, agreement, dedup, length, PII, balance)."""
    pipeline = _pipeline(config)
    examples = pipeline.filter()
    _print_funnel(pipeline)
    console.print(f"[green]{len(examples)} clean examples[/] -> {pipeline.paths.filtered}")


@app.command()
def build(config: ConfigOption = DEFAULT_CONFIG) -> None:
    """Stage 4: stratified split, chat-format JSONL and the dataset card."""
    pipeline = _pipeline(config)
    stats = pipeline.build()
    console.print(
        f"[green]train {stats['sizes']['train']} / val {stats['sizes']['val']}[/] -> {pipeline.paths.dataset_dir}"
    )


@app.command()
def train(
    config: ConfigOption = DEFAULT_CONFIG,
    max_steps: Annotated[int | None, typer.Option(help="Stop after N optimizer steps (smoke runs).")] = None,
) -> None:
    """Stage 5: LoRA fine-tune the student (needs the `train` extra)."""
    pipeline = _pipeline(config)
    if pipeline.config.student.backend == "fake":
        console.print(
            "[yellow]student.backend is 'fake': nothing to train "
            "(use configs/openrouter-free.yaml or set backend: hf)[/]"
        )
        return
    metrics = pipeline.train(max_steps=max_steps)
    console.print(
        f"[green]adapter saved[/] -> {pipeline.paths.adapter_dir} "
        f"({metrics['steps']} steps, loss {metrics['train_loss']}, {metrics['wall_seconds']}s on {metrics['device']})"
    )


@app.command(name="eval")
def eval_(
    config: ConfigOption = DEFAULT_CONFIG,
    systems: Annotated[
        str, typer.Option(help="Comma-separated: teacher, student_zero_shot, student_finetuned.")
    ] = ALL_SYSTEMS,
) -> None:
    """Stage 6: score teacher and students on the hand-labeled gold set, simulate the router, estimate cost."""
    pipeline = _pipeline(config)
    report = asyncio.run(pipeline.evaluate([system.strip() for system in systems.split(",") if system.strip()]))
    _print_eval(report)
    console.print(f"report -> {pipeline.paths.eval_report}")


@app.command()
def report(
    config: ConfigOption = DEFAULT_CONFIG,
    run_dir: Annotated[Path | None, typer.Option(help="Render this run directory instead (e.g. results/...).")] = None,
) -> None:
    """Render the static HTML report (same page as the dashboard, without the playground)."""
    pipeline = _pipeline(config)
    console.print(f"[green]report[/] -> {pipeline.report(run_dir)}")


@app.command()
def export(
    destination: Annotated[Path, typer.Argument(help="Target directory, e.g. results/my-run.")],
    config: ConfigOption = DEFAULT_CONFIG,
    with_adapter: Annotated[bool, typer.Option(help="Also copy the LoRA adapter weights.")] = False,
) -> None:
    """Copy a run's shareable artifacts (reports, dataset, call ledger) out of runs/ for committing."""
    pipeline = _pipeline(config)
    copied = pipeline.export(destination, include_adapter=with_adapter)
    console.print(f"[green]{len(copied)} files[/] -> {destination}")


@app.command(name="run-all")
def run_all(
    config: ConfigOption = DEFAULT_CONFIG,
    skip_train: Annotated[bool, typer.Option(help="Skip training (eval uses an existing adapter).")] = False,
) -> None:
    """Every stage in order: generate, label, filter, build, train, eval, report."""
    pipeline = _pipeline(config)
    asyncio.run(pipeline.generate())
    asyncio.run(pipeline.label())
    pipeline.filter()
    _print_funnel(pipeline)
    pipeline.build()
    systems = ["teacher", "student_finetuned"]
    if pipeline.config.student.backend == "hf":
        if not skip_train:
            pipeline.train()
        systems.insert(1, "student_zero_shot")
    report_data = asyncio.run(pipeline.evaluate(systems))
    _print_eval(report_data)
    console.print(f"[green]done[/]: {pipeline.report()}")


@app.command()
def serve(
    config: ConfigOption = DEFAULT_CONFIG,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    """Run the /triage API and the dashboard."""
    import os

    import uvicorn

    os.environ["DISTILLERY_CONFIG"] = str(config)
    uvicorn.run("distillery.serve:create_app", factory=True, host=host, port=port)


@app.command()
def calls(config: ConfigOption = DEFAULT_CONFIG) -> None:
    """Summarize the real API calls recorded in the run's call ledger."""
    pipeline = _pipeline(config)
    if not pipeline.paths.calls.exists():
        console.print("no real API calls recorded for this run")
        return
    rows = list(iter_jsonl(pipeline.paths.calls))
    table = Table(title=f"{len(rows)} real API requests ({pipeline.paths.calls})")
    for column in ("stage", "status", "served model", "requests"):
        table.add_column(column)
    counts = Counter((row["stage"], row["status"], row.get("served_model") or "-") for row in rows)
    for (stage, status, model), count in sorted(counts.items()):
        table.add_row(stage, status, model, str(count))
    console.print(table)


@app.command(name="free-models")
def free_models(
    structured_only: Annotated[bool, typer.Option(help="Only models that support response_format.")] = True,
) -> None:
    """List free (":free") OpenRouter models, which the demo config is restricted to."""
    import httpx

    settings = Settings()
    response = httpx.get(f"{settings.openrouter_base_url}/models", timeout=30)
    response.raise_for_status()
    table = Table(title="Free OpenRouter models")
    for column in ("id", "context", "response_format", "structured_outputs"):
        table.add_column(column)
    for model in response.json()["data"]:
        if not model["id"].endswith(":free"):
            continue
        params = set(model.get("supported_parameters") or [])
        if structured_only and "response_format" not in params:
            continue
        table.add_row(
            model["id"],
            str(model.get("context_length", "")),
            "yes" if "response_format" in params else "",
            "yes" if "structured_outputs" in params else "",
        )
    console.print(table)


@app.command(name="show-config")
def show_config(config: ConfigOption = DEFAULT_CONFIG) -> None:
    """Print the fully resolved config (defaults included)."""
    console.print_json(json.dumps(load_config(config).model_dump(mode="json")))


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
