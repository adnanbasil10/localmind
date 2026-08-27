# Build conventions — binding on every contributor (human or agent)

Spec: `implementation.md`. Section refs (§N) point there. It is the authority; this file
records only the cross-cutting contracts the spec leaves implicit.

## Hard constraints (§20)
1. **Nothing without a benchmark.** Every component ships a measurable baseline + upgrade + delta.
2. **Report negatives.** A losing result is a deliverable, not a failure.
3. **No bf16, ever.** Target hardware is a free Kaggle T4 (SM 7.5). Use fp16 autocast +
   `torch.amp.GradScaler`, fp32 master weights, fp32 loss. bf16 in a commit is a bug.
4. **No FlashAttention-2.** Requires SM 8.0+. Use `F.scaled_dot_product_attention` and
   assert which backend fired via `torch.nn.attention.sdpa_kernel`.
5. **Never report a bare number.** ≥3 seeds, bootstrap 95% CI. State the hardware.

## Code contracts
- Python ≥3.11. Type-annotate all public functions; `pyright` standard mode must pass on `localmind/`.
- `ruff check` and `ruff format --check` must pass. Line length 100.
- Config via pydantic v2 models loaded from `configs/**.yaml`. No argparse spaghetti; no magic numbers
  inline — every hyperparameter comes from a config file and is hash-logged.
- Determinism: anything stochastic takes an explicit `seed`. Seeded runs must reproduce bit-exactly.
- No network calls at import time. Anything needing net/GPU/docker gets a pytest marker
  (`net`, `gpu`, `docker`, `slow`) so `just test-fast` stays green offline.

## Ownership (do not edit outside your directory)
Controller-owned, off-limits: `pyproject.toml`, `configs/`, `justfile`, `CONVENTIONS.md`,
`implementation.md`, `.github/`, `docker-compose.yml`.
Each task owns exactly one `localmind/<subpackage>/` plus `tests/test_<subpackage>*.py`.

## Shared interfaces (frozen — depend on these, don't redefine them)

```python
# localmind/model/config.py    — owned by the model task
class ModelConfig(BaseModel):
    name: str; vocab_size: int; d_model: int; n_layers: int
    n_heads: int; n_kv_heads: int; head_dim: int; ffn_hidden: int
    max_seq_len: int; rope_theta: float; qk_norm: bool; bias: bool
    tie_embeddings: bool; z_loss: float; attn_dropout: float; init_std: float
    @classmethod
    def from_yaml(cls, path: str | Path) -> "ModelConfig": ...

# localmind/tokenizer/tokenizer.py  — owned by the tokenizer task
class Tokenizer(Protocol):
    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...
    def apply_chat_template(self, messages: list[dict[str, str]], add_generation_prompt: bool = False) -> str: ...
    @property
    def vocab_size(self) -> int: ...

# localmind/inference/engine.py     — owned by the inference task
class GenerationEngine(Protocol):
    def generate(self, prompt_ids: list[int], max_new_tokens: int, **kw) -> list[int]: ...

# The three production jobs the 31M model does (§9 5a). Agent depends on THIS, not on torch.
# localmind/agent/router.py / grader.py / rewriter.py
class ControlPlane(Protocol):
    def route(self, query: str) -> Literal["in_domain", "out_of_domain", "needs_web"]: ...
    def grade(self, query: str, chunk: str) -> tuple[bool, float]: ...   # (relevant, score)
    def rewrite(self, query: str, history: list[str]) -> str: ...
```

## Reporting
Benchmarks write JSON to `artifacts/benchmarks/<name>.json` and a markdown table appended to
`docs/benchmarks.md`. Schema: `{"name","hardware","seeds","rows":[{...}],"ci":"bootstrap95"}`.
