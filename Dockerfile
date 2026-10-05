# syntax=docker/dockerfile:1

# ---- build: resolve dependencies with uv into a self-contained virtualenv ----
FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /bin/uv
# EXTRAS="train" adds CPU torch + transformers + peft so the API can serve the real LoRA student.
ARG EXTRAS=""
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0
WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project $(for extra in $EXTRAS; do printf -- "--extra %s " "$extra"; done)

COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable $(for extra in $EXTRAS; do printf -- "--extra %s " "$extra"; done)

# ---- runtime: slim image, non-root user ----
FROM python:3.12-slim
RUN useradd --create-home --uid 1000 app
WORKDIR /app

COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app configs ./configs
COPY --chown=app:app data ./data
COPY --chown=app:app results ./results

# Defaults: the committed real-run results on the dashboard, the offline fake student/teacher in
# the playground and /triage (no keys, no downloads). See README > Docker for the real student.
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/app/.cache/huggingface \
    DISTILLERY_CONFIG=configs/demo.yaml \
    DISTILLERY_RUN_DIR=results/openrouter-free-2026-10-05 \
    LOG_FORMAT=json

USER app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"]
CMD ["uvicorn", "distillery.serve:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
