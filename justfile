set windows-shell := ["powershell.exe", "-NoLogo", "-Command"]

default:
    @just --list

install:
    uv venv
    uv pip install -e ".[torch,tok,data,dev]"

test:
    uv run pytest -q

test-fast:
    uv run pytest -q -m "not slow and not gpu and not net and not docker"

lint:
    uv run ruff check localmind tests
    uv run ruff format --check localmind tests

fmt:
    uv run ruff format localmind tests

types:
    uv run pyright

up profile="core":
    docker compose --profile {{profile}} up -d

down:
    docker compose --profile full down

train-pretrain config="configs/train/pretrain.yaml":
    uv run python -m localmind.train.loop --config {{config}}

train-smoke:
    uv run python -m localmind.train.loop --config configs/train/smoke.yaml

bench-tokenizer:
    uv run python -m localmind.tokenizer.bench

bench-inference:
    uv run python -m localmind.inference.bench

eval-retrieval:
    uv run python -m localmind.eval.retrieval --config configs/retrieval/default.yaml

eval-compare baseline="main" max_regression="0.02":
    uv run python -m localmind.eval.report --baseline {{baseline}} --max-regression {{max_regression}}

serve:
    uv run uvicorn localmind.inference.server:app --host 0.0.0.0 --port 8000
