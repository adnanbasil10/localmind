# Hugging Face Space — Docker SDK, free CPU tier (2 vCPU / 16 GB).
#
# The whole point: LocalMind-31M at int8 is ~31 MB, so the public demo needs no GPU and no
# cloud account. Ships the API + the int8 model + a prebuilt FAISS/SQLite index (no pgvector,
# because a Space is a single container).
#
# HF Spaces requires the app to listen on port 7860 and to run as a non-root user with uid 1000.

FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/home/user/.cache/huggingface \
    PORT=7860

# uid 1000 is what Spaces expects; writing outside /home/user fails at runtime.
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"
WORKDIR /home/user/app

COPY --chown=user pyproject.toml README.md ./
COPY --chown=user localmind ./localmind
COPY --chown=user configs ./configs

# CPU-only torch. Skip the `obs` extra — no collector runs inside a Space.
RUN pip install --no-cache-dir --user \
      --extra-index-url https://download.pytorch.org/whl/cpu \
      -e ".[rag]"

# Prebuilt artifacts: int8 weights + the demo index. Both are small enough to bake into the image.
COPY --chown=user artifacts/space/ ./artifacts/

EXPOSE 7860

CMD ["uvicorn", "localmind.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
