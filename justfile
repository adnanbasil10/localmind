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

# Regenerates artifacts/benchmarks/retrieval.json. The benchmark lives in the test
# suite, but a plain `pytest` run must not mutate a committed artifact, so publishing
# is opt-in and this recipe is the documented way to ask for it.
bench-retrieval:
    $env:LOCALMIND_PUBLISH_ARTIFACTS = "1"; uv run pytest -q -s tests/test_retrieval.py::test_benchmark_table_and_fusion_comparison

eval-retrieval:
    uv run python -m localmind.eval.retrieval --config configs/retrieval/default.yaml

eval-compare baseline="main" max_regression="0.02":
    uv run python -m localmind.eval.report --baseline {{baseline}} --max-regression {{max_regression}}

serve:
    uv run uvicorn localmind.inference.server:app --host 0.0.0.0 --port 8000
