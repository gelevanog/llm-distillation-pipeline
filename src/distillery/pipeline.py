"""Stage runners: each reads the previous stage's files from the run directory and writes its own."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from distillery.build import dataset_card, dataset_stats, stratified_split, to_chat_record
from distillery.config import PipelineConfig, Settings
from distillery.filter import run_filter
from distillery.generate import generate_synthetic, import_tickets, seed_summary
from distillery.io import read_json, read_jsonl, write_json, write_jsonl
from distillery.label import agreement_stats, label_tickets
from distillery.logging_config import get_logger
from distillery.providers.factory import ClientFactory
from distillery.records import Example, FunnelStep, LabeledTicket, RawTicket, RunPaths
from distillery.schema import Intent

log = get_logger(__name__)


class Pipeline:
    def __init__(self, config: PipelineConfig, settings: Settings | None = None) -> None:
        self.config = config
        self.settings = settings or Settings()
        self.paths = RunPaths(root=config.output_dir, dataset_override=config.dataset_dir)
        self._factory: ClientFactory | None = None

    @property
    def factory(self) -> ClientFactory:
        if self._factory is None:
            self._factory = ClientFactory(self.config, self.settings, self.paths.calls)
        return self._factory

    def _require(self, *paths: Any) -> None:
        missing = [str(path) for path in paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"missing input(s): {', '.join(missing)}; run the previous stage first")

    async def generate(
        self, *, append: bool = False, num: int | None = None, topics: list[Intent] | None = None, prefix: str = "s"
    ) -> list[RawTicket]:
        """Write generated.jsonl. With `append`, add a top-up pass (e.g. one under-represented topic) to it."""
        existing = read_jsonl(self.paths.generated, RawTicket) if append and self.paths.generated.exists() else []
        manifest = read_json(self.paths.generation) if append and self.paths.generation.exists() else {"passes": []}
        tickets: list[RawTicket] = []
        if self.config.generate.synthetic and (num if num is not None else self.config.generate.num_tickets) > 0:
            client = self.factory.client(self.config.teacher.generator, stage="generate")
            generated = await generate_synthetic(self.config, client, num=num, topics=topics, prefix=prefix)
            requested = num if num is not None else self.config.generate.num_tickets
            manifest["passes"].append(
                {
                    "requested": requested,
                    "generated": len(generated),
                    "topics": [topic.value for topic in topics] if topics else "all",
                    "id_prefix": prefix,
                }
            )
            tickets.extend(generated)
        if self.config.generate.import_path is not None and not append:
            imported = import_tickets(self.config.generate.import_path)
            log.info("generate.imported", path=str(self.config.generate.import_path), tickets=len(imported))
            manifest["imported"] = len(imported)
            tickets.extend(imported)
        known = {ticket.id for ticket in existing}
        duplicates = [ticket.id for ticket in tickets if ticket.id in known]
        if duplicates:
            raise ValueError(f"ticket ids already in {self.paths.generated}: use another --prefix")
        all_tickets = [*existing, *tickets]
        write_jsonl(self.paths.generated, all_tickets)
        write_json(self.paths.generation, manifest)
        return all_tickets

    async def label(self) -> list[LabeledTicket]:
        self._require(self.paths.generated)
        tickets = read_jsonl(self.paths.generated, RawTicket)
        labeled = await label_tickets(self.config, tickets, self.factory)
        write_jsonl(self.paths.labeled, labeled)
        stats = agreement_stats(labeled)
        stats["seed_matrix"] = seed_summary(tickets)
        write_json(self.paths.label_stats, stats)
        return labeled

    def filter(self) -> list[Example]:
        self._require(self.paths.labeled)
        labeled = read_jsonl(self.paths.labeled, LabeledTicket)
        requested: int | None = None
        if self.paths.generation.exists():
            manifest = read_json(self.paths.generation)
            if "imported" not in manifest:
                requested = sum(int(item["requested"]) for item in manifest["passes"]) or None
        result = run_filter(self.config.filter, labeled, requested=requested, seed=self.config.seed)
        write_jsonl(self.paths.filtered, result.examples)
        write_jsonl(self.paths.dropped, result.dropped)
        write_json(self.paths.funnel, [step.model_dump() for step in result.funnel])
        log.info("filter.done", **{step.name: step.kept for step in result.funnel})
        return result.examples

    def build(self) -> dict[str, Any]:
        self._require(self.paths.filtered, self.paths.funnel)
        examples = read_jsonl(self.paths.filtered, Example)
        funnel = [FunnelStep.model_validate(step) for step in read_json(self.paths.funnel)]
        train, val = stratified_split(examples, self.config.build.val_fraction, self.config.seed)
        write_jsonl(self.paths.train, [to_chat_record(example) for example in train])
        write_jsonl(self.paths.val, [to_chat_record(example) for example in val])
        label_stats = read_json(self.paths.label_stats) if self.paths.label_stats.exists() else None
        teachers = [model.provider + "/" + model.model for model in self.config.teacher.labelers]
        stats = dataset_stats(train, val, funnel, label_stats, teachers)
        write_json(self.paths.dataset_stats, stats)
        self.paths.dataset_card.write_text(dataset_card(stats, self.config.name), encoding="utf-8")
        log.info("build.done", train=len(train), val=len(val))
        return stats

    def train(self, *, max_steps: int | None = None) -> dict[str, Any]:
        from distillery.train import train_student

        self._require(self.paths.train)
        return train_student(
            self.paths.train,
            self.paths.val,
            self.paths.adapter_dir,
            student=self.config.student,
            train=self.config.train,
            metrics_path=self.paths.train_metrics,
            merged_dir=self.paths.merged_dir if self.config.train.merge_adapter else None,
            max_steps=max_steps,
        )

    async def evaluate(self, systems: list[str]) -> dict[str, Any]:
        from distillery.evaluate import run_eval

        return await run_eval(self.config, self.factory, self.paths, systems=systems)

    def report(self, run_dir: Path | None = None) -> str:
        from distillery.dashboard.render import render_static_report

        paths = RunPaths(root=run_dir) if run_dir is not None else self.paths
        html = render_static_report(paths, self.config)
        paths.report_html.write_text(html, encoding="utf-8")
        return str(paths.report_html)

    def export(self, destination: Path, *, include_adapter: bool = False) -> list[Path]:
        """Copy the small, shareable artifacts of a run (reports, dataset, call ledger) to `destination`.

        Generated/labeled intermediates and model weights stay behind; the LoRA adapter only with
        `include_adapter`. The copy is a valid run directory for the dashboard (`DISTILLERY_RUN_DIR`).
        """
        paths = self.paths
        files = [
            paths.generation,
            paths.funnel,
            paths.label_stats,
            paths.train,
            paths.val,
            paths.dataset_stats,
            paths.dataset_card,
            paths.train_metrics,
            paths.eval_report,
            paths.calls,
            paths.report_html,
            *sorted(paths.eval_dir.glob("predictions_*.jsonl")),
        ]
        if include_adapter:
            files += sorted(path for path in paths.adapter_dir.glob("*") if path.is_file())
        copied = []
        for source in files:
            if not source.exists():
                continue
            relative = (
                source.relative_to(paths.dataset_dir).as_posix() if source.is_relative_to(paths.dataset_dir) else None
            )
            target = destination / ("dataset/" + relative if relative else source.relative_to(paths.root).as_posix())
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied.append(target)
        return copied
