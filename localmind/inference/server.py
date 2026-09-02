"""OpenAI-compatible HTTP surface for the LocalMind engine (section 10, "API surface").

Endpoints: ``/v1/chat/completions`` (with SSE streaming), ``/v1/completions``,
``/v1/embeddings``, ``/v1/models``, ``/health``, ``/metrics``.

Why OpenAI-compatible and not a bespoke ``/generate``: every eval harness, every client
SDK, every proxy, and the ``openai`` Python package already speak this wire format. A
base-URL change is the entire integration cost. A bespoke endpoint is worth strictly less
than the hour it takes to write.

Layering, and why it is like this
---------------------------------
Everything that decides the *bytes on the wire* -- :func:`chat_completion_response`,
:func:`chat_completion_chunk`, :func:`sse_frame`, :data:`SSE_DONE` -- is plain Python with
no web framework anywhere near it. FastAPI is imported **lazily inside** :func:`create_app`
so ``import localmind.inference`` works with only the base dependencies installed
(``fastapi``/``uvicorn`` live in the ``rag`` extra and may be absent).

That split is also what makes the conformance test possible offline: the test asserts the
exact envelope, ``choices`` shape, ``usage`` accounting, SSE ``data:`` framing and the
``[DONE]`` sentinel against these functions directly, and the live ``openai``-client test
is marked ``net`` so ``just test-fast`` stays green without the package installed.

``uvicorn localmind.inference.server:app`` still works: module-level ``app`` is resolved
through :pep:`562` ``__getattr__``, so the FastAPI import happens at attribute access, not
at import time.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "SSE_DONE",
    "ChatCompletionRequest",
    "CompletionRequest",
    "EmbeddingsRequest",
    "EngineState",
    "ServerConfig",
    "ServerMetrics",
    "chat_completion_chunk",
    "chat_completion_response",
    "completion_response",
    "create_app",
    "embeddings_response",
    "models_response",
    "prometheus_text",
    "sse_frame",
    "stream_chat_completion",
]

#: The exact terminator the OpenAI SSE protocol requires. Clients loop until they see it.
SSE_DONE = "data: [DONE]\n\n"


# ---------------------------------------------------------------------------------
# Request models (pydantic only -- no FastAPI needed to validate or to test)
# ---------------------------------------------------------------------------------
class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: str
    content: str | None = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow", protected_namespaces=())

    model: str = "localmind"
    messages: list[ChatMessage]
    max_tokens: int | None = Field(default=64, gt=0)
    temperature: float = Field(default=0.0, ge=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    n: int = Field(default=1, ge=1)
    stream: bool = False
    stop: str | list[str] | None = None
    seed: int | None = None
    response_format: dict[str, Any] | None = None


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow", protected_namespaces=())

    model: str = "localmind"
    prompt: str | list[str] = ""
    max_tokens: int | None = Field(default=64, gt=0)
    temperature: float = Field(default=0.0, ge=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    n: int = Field(default=1, ge=1)
    stream: bool = False
    seed: int | None = None


class EmbeddingsRequest(BaseModel):
    model_config = ConfigDict(extra="allow", protected_namespaces=())

    model: str = "localmind"
    input: str | list[str]
    encoding_format: str = "float"


class ServerConfig(BaseModel):
    """Serving knobs. Env-overridable so the container needs no argparse."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_config_path: str = "configs/model/31m.yaml"
    checkpoint_path: str | None = None
    tokenizer_path: str | None = None
    engine: str = "cached"
    served_model_name: str = "localmind-31m"
    max_new_tokens_cap: int = 512
    prefill_chunk_size: int | None = 64

    @classmethod
    def from_env(cls) -> ServerConfig:
        return cls(
            model_config_path=os.getenv("LOCALMIND_MODEL_CONFIG", "configs/model/31m.yaml"),
            checkpoint_path=os.getenv("LOCALMIND_CHECKPOINT") or None,
            tokenizer_path=os.getenv("LOCALMIND_TOKENIZER") or None,
            engine=os.getenv("LOCALMIND_ENGINE", "cached"),
            served_model_name=os.getenv("LOCALMIND_MODEL_NAME", "localmind-31m"),
        )


# ---------------------------------------------------------------------------------
# Wire-format builders  (framework-free, and therefore testable offline)
# ---------------------------------------------------------------------------------
def _now() -> int:
    return int(time.time())


def _usage(prompt_tokens: int, completion_tokens: int) -> dict[str, int]:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def chat_completion_response(
    texts: Sequence[str],
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    finish_reasons: Sequence[str] | None = None,
    request_id: str | None = None,
    created: int | None = None,
) -> dict[str, Any]:
    """A ``chat.completion`` object, byte-compatible with the OpenAI schema."""
    reasons = list(finish_reasons or ["stop"] * len(texts))
    return {
        "id": request_id or f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": created if created is not None else _now(),
        "model": model,
        "choices": [
            {
                "index": i,
                "message": {"role": "assistant", "content": text},
                "logprobs": None,
                "finish_reason": reasons[i] if i < len(reasons) else "stop",
            }
            for i, text in enumerate(texts)
        ],
        "usage": _usage(prompt_tokens, completion_tokens),
    }


def chat_completion_chunk(
    delta: dict[str, Any],
    model: str,
    request_id: str,
    index: int = 0,
    finish_reason: str | None = None,
    created: int | None = None,
) -> dict[str, Any]:
    """One ``chat.completion.chunk``. ``delta`` is the incremental payload, never the full text."""
    return {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created if created is not None else _now(),
        "model": model,
        "choices": [
            {"index": index, "delta": delta, "logprobs": None, "finish_reason": finish_reason}
        ],
    }


def completion_response(
    texts: Sequence[str],
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    finish_reasons: Sequence[str] | None = None,
    request_id: str | None = None,
    created: int | None = None,
) -> dict[str, Any]:
    reasons = list(finish_reasons or ["stop"] * len(texts))
    return {
        "id": request_id or f"cmpl-{uuid.uuid4().hex[:24]}",
        "object": "text_completion",
        "created": created if created is not None else _now(),
        "model": model,
        "choices": [
            {
                "index": i,
                "text": text,
                "logprobs": None,
                "finish_reason": reasons[i] if i < len(reasons) else "stop",
            }
            for i, text in enumerate(texts)
        ],
        "usage": _usage(prompt_tokens, completion_tokens),
    }


def embeddings_response(
    vectors: Sequence[Sequence[float]],
    model: str,
    prompt_tokens: int,
    created: int | None = None,
) -> dict[str, Any]:
    del created
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "index": i, "embedding": list(v)} for i, v in enumerate(vectors)
        ],
        "model": model,
        "usage": {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens},
    }


def models_response(names: Sequence[str], created: int | None = None) -> dict[str, Any]:
    ts = created if created is not None else _now()
    return {
        "object": "list",
        "data": [
            {"id": n, "object": "model", "created": ts, "owned_by": "localmind"} for n in names
        ],
    }


def sse_frame(payload: dict[str, Any]) -> str:
    """``data: <compact json>\\n\\n`` -- the framing the OpenAI SDK's parser expects.

    Two newlines terminate an SSE event. One is the single most common bug in a
    hand-rolled streaming endpoint, and it manifests as a client that hangs forever.
    """
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"


def stream_chat_completion(
    pieces: Sequence[str],
    model: str,
    request_id: str | None = None,
    finish_reason: str = "stop",
    created: int | None = None,
) -> Iterator[str]:
    """The full SSE event sequence for one streamed chat completion.

    Role chunk first (OpenAI always sends ``delta.role`` once), then one chunk per token,
    then a terminating chunk carrying ``finish_reason`` and an empty delta, then
    ``[DONE]``.
    """
    rid = request_id or f"chatcmpl-{uuid.uuid4().hex[:24]}"
    ts = created if created is not None else _now()
    yield sse_frame(
        chat_completion_chunk({"role": "assistant", "content": ""}, model, rid, created=ts)
    )
    for piece in pieces:
        yield sse_frame(chat_completion_chunk({"content": piece}, model, rid, created=ts))
    yield sse_frame(chat_completion_chunk({}, model, rid, finish_reason=finish_reason, created=ts))
    yield SSE_DONE


# ---------------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------------
@dataclass
class ServerMetrics:
    """The section 10 vocabulary, exposed on ``/metrics`` in Prometheus text format."""

    requests_total: int = 0
    requests_failed: int = 0
    prompt_tokens_total: int = 0
    generation_tokens_total: int = 0
    ttft_s: list[float] = field(default_factory=list)
    tpot_s: list[float] = field(default_factory=list)
    e2e_s: list[float] = field(default_factory=list)
    ttft_slo_s: float = 0.5
    tpot_slo_s: float = 0.05
    goodput_requests: int = 0
    started_at: float = field(default_factory=time.perf_counter)

    def observe(
        self, prompt_tokens: int, gen_tokens: int, ttft: float, tpot: float, e2e: float
    ) -> None:
        self.requests_total += 1
        self.prompt_tokens_total += prompt_tokens
        self.generation_tokens_total += gen_tokens
        self.ttft_s.append(ttft)
        if tpot == tpot:
            self.tpot_s.append(tpot)
        self.e2e_s.append(e2e)
        if ttft < self.ttft_slo_s and (tpot != tpot or tpot < self.tpot_slo_s):
            self.goodput_requests += 1

    def snapshot(self) -> dict[str, Any]:
        from localmind.inference.scheduler import percentiles

        uptime = max(1e-9, time.perf_counter() - self.started_at)
        return {
            "requests_total": self.requests_total,
            "requests_failed": self.requests_failed,
            "prompt_tokens_total": self.prompt_tokens_total,
            "generation_tokens_total": self.generation_tokens_total,
            "output_throughput_tok_s": self.generation_tokens_total / uptime,
            "goodput_req_s": self.goodput_requests / uptime,
            "ttft": percentiles(self.ttft_s),
            "tpot": percentiles(self.tpot_s),
            "e2e": percentiles(self.e2e_s),
            "uptime_s": uptime,
        }


def prometheus_text(metrics: ServerMetrics) -> str:
    """Prometheus exposition format. Names follow the vLLM convention so dashboards port over."""
    s = metrics.snapshot()
    lines = [
        "# HELP localmind:requests_total Total completion requests served.",
        "# TYPE localmind:requests_total counter",
        f"localmind:requests_total {s['requests_total']}",
        "# HELP localmind:generation_tokens_total Total tokens generated.",
        "# TYPE localmind:generation_tokens_total counter",
        f"localmind:generation_tokens_total {s['generation_tokens_total']}",
        "# HELP localmind:prompt_tokens_total Total prompt tokens processed.",
        "# TYPE localmind:prompt_tokens_total counter",
        f"localmind:prompt_tokens_total {s['prompt_tokens_total']}",
        "# HELP localmind:output_throughput_tok_s Output tokens per second since start.",
        "# TYPE localmind:output_throughput_tok_s gauge",
        f"localmind:output_throughput_tok_s {s['output_throughput_tok_s']:.6f}",
        "# HELP localmind:goodput_req_s Requests/s meeting the TTFT+TPOT SLO.",
        "# TYPE localmind:goodput_req_s gauge",
        f"localmind:goodput_req_s {s['goodput_req_s']:.6f}",
        "# HELP localmind:time_to_first_token_seconds TTFT quantiles.",
        "# TYPE localmind:time_to_first_token_seconds summary",
    ]
    for q in ("p50", "p95", "p99"):
        lines.append(
            f'localmind:time_to_first_token_seconds{{quantile="{q[1:]}"}} {s["ttft"][q]:.6f}'
        )
    lines += [
        "# HELP localmind:time_per_output_token_seconds TPOT/ITL quantiles.",
        "# TYPE localmind:time_per_output_token_seconds summary",
    ]
    for q in ("p50", "p95", "p99"):
        lines.append(
            f'localmind:time_per_output_token_seconds{{quantile="{q[1:]}"}} {s["tpot"][q]:.6f}'
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------
# Engine state (torch imported lazily so the wire helpers stay importable)
# ---------------------------------------------------------------------------------
class _ByteTokenizer:
    """UTF-8 byte fallback used when no trained tokenizer artifact is configured.

    Not a good tokenizer -- it is a *correct* one, which is what the HTTP layer needs to
    be exercisable end to end before Phase 1's artifact exists. ``LOCALMIND_TOKENIZER``
    swaps in the real one.
    """

    def __init__(self, vocab_size: int) -> None:
        self._vocab_size = vocab_size
        self.eos_token_id = 0

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        del add_bos, add_eos
        return [b % self._vocab_size for b in text.encode("utf-8")] or [0]

    def decode(self, ids: Sequence[int]) -> str:
        return bytes(i % 256 for i in ids).decode("utf-8", errors="replace")

    def apply_chat_template(
        self, messages: Sequence[dict[str, str]], add_generation_prompt: bool = False
    ) -> str:
        parts = [f"<|{m.get('role', 'user')}|>{m.get('content') or ''}" for m in messages]
        if add_generation_prompt:
            parts.append("<|assistant|>")
        return "\n".join(parts)


@dataclass
class EngineState:
    """Everything a request handler needs. Built once at startup."""

    config: ServerConfig
    model: Any
    tokenizer: Any
    engine: Any
    metrics: ServerMetrics = field(default_factory=ServerMetrics)

    @classmethod
    def build(cls, config: ServerConfig | None = None) -> EngineState:
        import torch

        from localmind.inference.engine import build_engine
        from localmind.model import LocalMindTransformer

        cfg = config or ServerConfig.from_env()
        model = LocalMindTransformer.from_yaml(cfg.model_config_path)
        if cfg.checkpoint_path:
            state = torch.load(cfg.checkpoint_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state.get("model", state))
        model.eval()

        tokenizer: Any
        if cfg.tokenizer_path:
            from localmind.tokenizer.tokenizer import Tokenizer

            tokenizer = Tokenizer.load(cfg.tokenizer_path)
        else:
            tokenizer = _ByteTokenizer(model.cfg.vocab_size)

        engine = build_engine(
            model,
            kind=cfg.engine,
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
            prefill_chunk_size=cfg.prefill_chunk_size,
        )
        return cls(config=cfg, model=model, tokenizer=tokenizer, engine=engine)

    # -- request handling ------------------------------------------------------------
    def _prompt_ids(self, text: str) -> list[int]:
        ids = self.tokenizer.encode(text)
        cap = self.model.cfg.max_seq_len - 2
        return list(ids[-cap:]) or [0]

    def chat_prompt(self, messages: Sequence[Any]) -> str:
        payload = [
            {"role": m.role, "content": m.content or ""} if hasattr(m, "role") else dict(m)
            for m in messages
        ]
        return self.tokenizer.apply_chat_template(payload, add_generation_prompt=True)

    def run(self, prompt: str, max_tokens: int, temperature: float, top_p: float, seed: int) -> Any:
        from localmind.inference.sampling import SamplingParams

        ids = self._prompt_ids(prompt)
        params = SamplingParams(
            max_new_tokens=min(max_tokens, self.config.max_new_tokens_cap),
            temperature=temperature,
            top_p=top_p,
            seed=seed,
        )
        t0 = time.perf_counter()
        result = self.engine.generate_detailed(ids, params.max_new_tokens, params=params)
        self.metrics.observe(
            prompt_tokens=len(ids),
            gen_tokens=result.n_generated,
            ttft=result.ttft_s,
            tpot=result.tpot_s,
            e2e=time.perf_counter() - t0,
        )
        return result

    def stream_tokens(
        self, prompt: str, max_tokens: int, temperature: float, top_p: float, seed: int
    ) -> Iterator[str]:
        from localmind.inference.sampling import SamplingParams

        ids = self._prompt_ids(prompt)
        params = SamplingParams(
            max_new_tokens=min(max_tokens, self.config.max_new_tokens_cap),
            temperature=temperature,
            top_p=top_p,
            seed=seed,
        )
        for chunk in self.engine.stream(ids, params.max_new_tokens, params=params):
            yield self.tokenizer.decode([chunk.token_id])

    def embed(self, texts: Sequence[str]) -> tuple[list[list[float]], int]:
        """Mean-pooled, L2-normalised final hidden states.

        The 31M model is a *control-plane* model, not a retrieval encoder -- Phase 8 owns
        real embeddings. This endpoint exists so the OpenAI surface is complete and so a
        client can point at one base URL; it is documented as such rather than dressed up.
        """
        import torch

        vectors: list[list[float]] = []
        total = 0
        with torch.no_grad():
            for text in texts:
                ids = self._prompt_ids(text)
                total += len(ids)
                t = torch.tensor([ids], dtype=torch.long)
                h = self.model.tok_emb(t)
                for block in self.model.blocks:
                    h, _ = block(h, self.model.rope)
                h = self.model.norm_f(h).mean(dim=1)[0]
                h = h / h.norm().clamp(min=1e-9)
                vectors.append([float(x) for x in h])
        return vectors, total


# ---------------------------------------------------------------------------------
# FastAPI app (lazy import)
# ---------------------------------------------------------------------------------
def create_app(state: EngineState | None = None, config: ServerConfig | None = None) -> Any:
    """Build the FastAPI application. Raises a clear error if the ``rag`` extra is missing."""
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import PlainTextResponse, StreamingResponse
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "fastapi/uvicorn are in the 'rag' optional dependency group. "
            "Install with: uv pip install -e '.[rag]'"
        ) from exc

    app = FastAPI(title="LocalMind inference server", version="0.1.0")
    app.state.engine_state = state  # built lazily on first use if None
    app.state.server_config = config

    def get_state() -> EngineState:
        if app.state.engine_state is None:
            app.state.engine_state = EngineState.build(app.state.server_config)
        return app.state.engine_state

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "model": (config or ServerConfig()).served_model_name}

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics() -> str:
        return prometheus_text(get_state().metrics)

    @app.get("/v1/models")
    def list_models() -> dict[str, Any]:
        return models_response([get_state().config.served_model_name])

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatCompletionRequest) -> Any:
        st = get_state()
        prompt = st.chat_prompt(req.messages)
        max_tokens = req.max_tokens or 64
        seed = req.seed if req.seed is not None else 0
        model_name = st.config.served_model_name
        if req.stream:
            rid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
            created = _now()

            def gen() -> Iterator[str]:
                yield sse_frame(
                    chat_completion_chunk(
                        {"role": "assistant", "content": ""}, model_name, rid, created=created
                    )
                )
                for piece in st.stream_tokens(prompt, max_tokens, req.temperature, req.top_p, seed):
                    yield sse_frame(
                        chat_completion_chunk({"content": piece}, model_name, rid, created=created)
                    )
                yield sse_frame(
                    chat_completion_chunk(
                        {}, model_name, rid, finish_reason="stop", created=created
                    )
                )
                yield SSE_DONE

            return StreamingResponse(gen(), media_type="text/event-stream")

        try:
            result = st.run(prompt, max_tokens, req.temperature, req.top_p, seed)
        except Exception as exc:  # pragma: no cover - defensive
            st.metrics.requests_failed += 1
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        text = st.tokenizer.decode(result.token_ids)
        return chat_completion_response(
            [text],
            model_name,
            prompt_tokens=result.prompt_len,
            completion_tokens=result.n_generated,
            finish_reasons=[result.finish_reason],
        )

    @app.post("/v1/completions")
    def completions(req: CompletionRequest) -> Any:
        st = get_state()
        prompts = [req.prompt] if isinstance(req.prompt, str) else list(req.prompt)
        max_tokens = req.max_tokens or 64
        seed = req.seed if req.seed is not None else 0
        texts: list[str] = []
        reasons: list[str] = []
        p_tok = 0
        c_tok = 0
        for p in prompts:
            result = st.run(p, max_tokens, req.temperature, req.top_p, seed)
            texts.append(st.tokenizer.decode(result.token_ids))
            reasons.append(result.finish_reason)
            p_tok += result.prompt_len
            c_tok += result.n_generated
        return completion_response(
            texts, st.config.served_model_name, p_tok, c_tok, finish_reasons=reasons
        )

    @app.post("/v1/embeddings")
    def embeddings(req: EmbeddingsRequest) -> Any:
        st = get_state()
        texts = [req.input] if isinstance(req.input, str) else list(req.input)
        vectors, total = st.embed(texts)
        return embeddings_response(vectors, st.config.served_model_name, total)

    return app


def __getattr__(name: str) -> Any:
    """PEP 562 hook so ``uvicorn localmind.inference.server:app`` works without an
    import-time FastAPI dependency."""
    if name == "app":
        return create_app()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
