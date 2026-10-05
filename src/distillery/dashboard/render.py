"""Jinja2 rendering shared by the live dashboard and the static `report.html`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from distillery.config import PipelineConfig
from distillery.dashboard.data import load_view
from distillery.records import RunPaths

TEMPLATES_DIR = Path(__file__).parent / "templates"

environment = Environment(
    loader=FileSystemLoader(TEMPLATES_DIR),
    autoescape=select_autoescape(["html"]),
    trim_blocks=True,
    lstrip_blocks=True,
    extensions=["jinja2.ext.do"],
)


def render(template: str, **context: Any) -> str:
    return environment.get_template(template).render(**context)


def render_static_report(paths: RunPaths, config: PipelineConfig) -> str:
    """Single self-contained HTML file (inline CSS and SVG, no JavaScript)."""
    return render("overview.html", view=load_view(paths.root), config_name=config.name, live=False, page="overview")
