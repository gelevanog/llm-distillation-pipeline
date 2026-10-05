.DEFAULT_GOAL := help
.PHONY: help install demo real train eval report serve dev test lint format docker-build docker-up free-models clean

CONFIG ?= configs/demo.yaml

help:  ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-13s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies incl. dev tools and the `train` extra (CPU torch)
	uv sync --all-extras

demo:  ## Whole pipeline offline with the fake teacher and student (seconds, no keys)
	uv run distillery run-all -c configs/demo.yaml

real:  ## Whole pipeline with free OpenRouter models + CPU training (needs OPENROUTER_API_KEY)
	uv run distillery run-all -c configs/openrouter-free.yaml

train:  ## Fine-tune the student for CONFIG
	uv run distillery train -c $(CONFIG)

eval:  ## Evaluate teacher and students on the gold set for CONFIG
	uv run distillery eval -c $(CONFIG)

report:  ## Render the static HTML report for CONFIG
	uv run distillery report -c $(CONFIG)

serve:  ## API + dashboard on http://localhost:8000 for CONFIG
	uv run distillery serve -c $(CONFIG)

dev:  ## Same as serve, with auto-reload
	DISTILLERY_CONFIG=$(CONFIG) uv run uvicorn distillery.serve:create_app --factory --reload --port 8000

test:  ## Run the test-suite (no API keys, no model downloads)
	uv run pytest

lint:  ## Ruff lint + format check + mypy (strict)
	uv run ruff check src tests
	uv run ruff format --check src tests
	uv run mypy

format:  ## Auto-format and fix lint issues
	uv run ruff format src tests
	uv run ruff check --fix src tests

free-models:  ## List free OpenRouter models with structured-output support
	uv run distillery free-models

docker-build:  ## Build the Docker image
	docker compose build

docker-up:  ## Dashboard + API in Docker (fake student by default)
	docker compose up --build

clean:  ## Remove run outputs and caches (keeps results/)
	rm -rf runs .cache .pytest_cache .mypy_cache .ruff_cache
