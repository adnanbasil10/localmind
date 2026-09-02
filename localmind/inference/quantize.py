"""Weight-only int8/int4 quantization and a GGUF exporter (section 10 step 9).

Quantization
------------
Weight-only, group-wise, symmetric. Activations stay fp32; only the stored weights
shrink. That is the right first move for a 31M model whose decode step is
memory-bandwidth-bound: the bytes you must stream per token *are* the weights.

Two things are measured and both are reported, including the one that loses:

* **size** -- unambiguous win. int8 is ~4x smaller than fp32, int4 ~8x, minus the
  per-group scales.
* **latency** -- our :class:`QuantLinear` dequantizes into fp32 and calls the ordinary
  GEMM, because no int8 CPU GEMM is available to us here. That *adds* work per forward.
  The honest conclusion is that weight-only quantization without an integer kernel buys
  memory, not speed, and the numbers say so. :func:`quantize_dynamic_int8` is included as
  the comparator that does have a real kernel (PyTorch's fbgemm/qnnpack dynamic path).

Quality is reported two ways. Val **BPB** is the metric implementation.md asks for, and it
is computed here -- but on an *untrained* model BPB sits at log2(vocab)/bytes-per-token
regardless of precision, so the BPB delta is dominated by noise and is nearly useless as a
quality signal until real weights exist. The sensitive metric at this stage is the
**KL divergence between the fp32 and quantized next-token distributions**, which responds
to quantization error immediately. Both are in the artifact; the caveat travels with them.

GGUF export
-----------
:func:`write_gguf` writes a real GGUF v3 container: magic/version/counts, typed metadata
key-values, tensor descriptors, alignment padding, then the tensor blob. Tensors are named
and laid out for llama.cpp's ``llama`` architecture, including the q/k permutation that
converts our rotate-half RoPE layout into ggml's interleaved-pair layout (the same
transform ``convert_hf_to_gguf.py`` applies).

**Verification status, stated plainly: this writer has NOT been run against a real
llama.cpp build.** No llama.cpp binary is available in this environment. What *is* verified
is a byte-level round trip -- :func:`read_gguf` parses the file this writer produces and
recovers every metadata value and every tensor to within Q8_0 quantization error, and the
test suite asserts that. Treat "runs in llama.cpp" as untested until someone runs it.

**Known incompatibility, also stated plainly:** LocalMind uses QK-norm, and llama.cpp's
``llama`` architecture has no ``attn_q_norm``/``attn_k_norm`` tensors. A config with
``qk_norm: true`` therefore cannot be represented faithfully; :func:`write_gguf` refuses
unless ``allow_lossy=True``, and records the fact in the file's metadata either way.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from localmind.model import LocalMindTransformer, ModelConfig

__all__ = [
    "GGMLType",
    "GGUFValueType",
    "QuantLinear",
    "bits_per_byte",
    "dequantize_int4",
    "dequantize_int8",
    "logit_kl",
    "model_size_bytes",
    "quantize_dynamic_int8",
    "quantize_int4",
    "quantize_int8",
    "quantize_model",
    "read_gguf",
    "write_gguf",
]


# ---------------------------------------------------------------------------------
# Weight-only quantization primitives
# ---------------------------------------------------------------------------------
def _grouped(w: Tensor, group_size: int) -> tuple[Tensor, int]:
    out_features, in_features = w.shape
    g = group_size if group_size > 0 else in_features
    if in_features % g != 0:
        g = math.gcd(in_features, g) or in_features
    return w.reshape(out_features, in_features // g, g), g


def quantize_int8(w: Tensor, group_size: int = 64) -> tuple[Tensor, Tensor, int]:
    """Symmetric per-group int8. Returns ``(q_int8, scales_fp32, group_size)``.

    Symmetric (zero-point-free) because weights are near zero-mean; an asymmetric scheme
    would spend a whole extra tensor of zero-points to buy almost nothing.
    """
    blocks, g = _grouped(w.float(), group_size)
    amax = blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = amax / 127.0
    q = torch.clamp(torch.round(blocks / scale), -127, 127).to(torch.int8)
    return q.reshape(w.shape), scale.squeeze(-1), g


def dequantize_int8(q: Tensor, scale: Tensor, group_size: int) -> Tensor:
    out_features, in_features = q.shape
    blocks = q.reshape(out_features, in_features // group_size, group_size).float()
    return (blocks * scale.unsqueeze(-1)).reshape(out_features, in_features)


def quantize_int4(w: Tensor, group_size: int = 64) -> tuple[Tensor, Tensor, int]:
    """Symmetric per-group int4, two nibbles packed per byte.

    Values live in ``[-8, 7]``; the nibble is stored biased by +8 so it fits an unsigned
    4-bit field, and ``dequantize_int4`` subtracts the bias back out.
    """
    blocks, g = _grouped(w.float(), group_size)
    amax = blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = amax / 7.0
    q = torch.clamp(torch.round(blocks / scale), -8, 7).to(torch.int8).reshape(w.shape)
    biased = (q.to(torch.uint8) + 8) & 0x0F
    low = biased[:, 0::2]
    high = biased[:, 1::2]
    packed = (low | (high << 4)).to(torch.uint8)
    return packed, scale.squeeze(-1), g


def dequantize_int4(packed: Tensor, scale: Tensor, group_size: int, in_features: int) -> Tensor:
    out_features = packed.shape[0]
    low = (packed & 0x0F).to(torch.int16) - 8
    high = ((packed >> 4) & 0x0F).to(torch.int16) - 8
    q = torch.empty((out_features, in_features), dtype=torch.int16, device=packed.device)
    q[:, 0::2] = low
    q[:, 1::2] = high
    blocks = q.reshape(out_features, in_features // group_size, group_size).float()
    return (blocks * scale.unsqueeze(-1)).reshape(out_features, in_features)


class QuantLinear(nn.Module):
    """Drop-in ``nn.Linear`` replacement holding int8 or packed-int4 weights.

    Dequantizes to fp32 inside ``forward``. See the module docstring: this is a *memory*
    optimisation, and the benchmark reports the latency it costs rather than hiding it.
    """

    def __init__(self, linear: nn.Linear, bits: int = 8, group_size: int = 64) -> None:
        super().__init__()
        if bits not in (4, 8):
            raise ValueError(f"bits must be 4 or 8, got {bits}")
        self.bits = bits
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        w = linear.weight.data
        if bits == 8:
            q, scale, g = quantize_int8(w, group_size)
        else:
            q, scale, g = quantize_int4(w, group_size)
        self.group_size = g
        # Bare declarations so pyright resolves `self.qweight` / `self.scales` to Tensor
        # at every use site instead of falling back to nn.Module.__getattr__'s
        # `Tensor | Module` return type (register_buffer alone doesn't give pyright that).
        self.qweight: Tensor
        self.scales: Tensor
        self.register_buffer("qweight", q)
        self.register_buffer("scales", scale)
        self.bias = None if linear.bias is None else nn.Parameter(linear.bias.data.clone())

    def dequantized(self) -> Tensor:
        if self.bits == 8:
            return dequantize_int8(self.qweight, self.scales, self.group_size)
        return dequantize_int4(self.qweight, self.scales, self.group_size, self.in_features)

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.dequantized().to(x.dtype), self.bias)

    def weight_bytes(self) -> int:
        q = self.qweight
        s = self.scales
        return q.numel() * q.element_size() + s.numel() * s.element_size()

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bits={self.bits}, group_size={self.group_size}"
        )


def quantize_model(
    model: nn.Module,
    bits: int = 8,
    group_size: int = 64,
    skip: Sequence[str] = ("lm_head",),
) -> nn.Module:
    """Replace every ``nn.Linear`` with a :class:`QuantLinear`, in place.

    ``lm_head`` is skipped by default: with tied embeddings it *is* the token embedding,
    and quantizing the output projection is where perplexity damage concentrates.
    """
    for name, child in list(model.named_children()):
        if any(s in name for s in skip):
            continue
        if isinstance(child, nn.Linear):
            setattr(model, name, QuantLinear(child, bits=bits, group_size=group_size))
        else:
            quantize_model(child, bits=bits, group_size=group_size, skip=skip)
    return model


def quantize_dynamic_int8(model: nn.Module) -> nn.Module:
    """PyTorch dynamic int8 over ``nn.Linear`` -- the comparator with a real int8 kernel.

    Raises ``RuntimeError`` when no quantized CPU backend is registered, which the
    benchmark catches and records as "not available on this machine" rather than
    reporting a fabricated number.
    """
    from torch.ao.quantization import quantize_dynamic

    return quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)


# ---------------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------------
def model_size_bytes(model: nn.Module) -> int:
    """Bytes of *distinct* parameter and buffer storage (tied weights counted once)."""
    seen: set[int] = set()
    total = 0
    for t in list(model.parameters()) + list(model.buffers()):
        key = id(t)
        if key in seen:
            continue
        seen.add(key)
        total += t.numel() * t.element_size()
    return total


@torch.no_grad()
def cross_entropy_nats(model: nn.Module, ids: Tensor, chunk: int = 256) -> float:
    """Mean next-token cross-entropy in nats over one ``(1, T)`` id sequence."""
    total = 0.0
    count = 0
    seq = ids[0]
    for start in range(0, seq.numel() - 1, chunk):
        piece = seq[start : start + chunk + 1]
        if piece.numel() < 2:
            break
        out = model(piece[:-1].unsqueeze(0))
        logits = out.logits[0].float()
        loss = F.cross_entropy(logits, piece[1:], reduction="sum")
        total += float(loss.item())
        count += piece.numel() - 1
    return total / count if count else float("nan")


def bits_per_byte(model: nn.Module, ids: Tensor, bytes_per_token: float) -> float:
    """BPB = nats/token / ln(2) / bytes/token.

    Tokenizer-independent by construction, which is why the spec prefers it to perplexity.
    """
    nats = cross_entropy_nats(model, ids)
    return nats / math.log(2) / bytes_per_token


@torch.no_grad()
def logit_kl(reference: nn.Module, candidate: nn.Module, ids: Tensor) -> float:
    """Mean KL(reference || candidate) over next-token distributions.

    The quality metric that actually responds to quantization error on an untrained model.
    """
    ref = reference(ids).logits[0].float()
    cand = candidate(ids).logits[0].float()
    p = F.log_softmax(ref, dim=-1)
    q = F.log_softmax(cand, dim=-1)
    return float((p.exp() * (p - q)).sum(dim=-1).mean().item())


# ---------------------------------------------------------------------------------
# GGUF
# ---------------------------------------------------------------------------------
class GGUFValueType(IntEnum):
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


class GGMLType(IntEnum):
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q8_0 = 8


GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
GGUF_DEFAULT_ALIGNMENT = 32
QK8_0 = 32  # elements per Q8_0 block


@dataclass
class GGUFTensor:
    name: str
    shape: tuple[int, ...]  # torch order (out, in)
    ggml_type: GGMLType
    data: bytes


def _pack_string(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


_SCALAR_FMT: dict[GGUFValueType, str] = {
    GGUFValueType.UINT8: "<B",
    GGUFValueType.INT8: "<b",
    GGUFValueType.UINT16: "<H",
    GGUFValueType.INT16: "<h",
    GGUFValueType.UINT32: "<I",
    GGUFValueType.INT32: "<i",
    GGUFValueType.FLOAT32: "<f",
    GGUFValueType.BOOL: "<?",
    GGUFValueType.UINT64: "<Q",
    GGUFValueType.INT64: "<q",
    GGUFValueType.FLOAT64: "<d",
}


def _pack_value(value: Any, vtype: GGUFValueType, elem_type: GGUFValueType | None = None) -> bytes:
    if vtype == GGUFValueType.STRING:
        return _pack_string(str(value))
    if vtype == GGUFValueType.ARRAY:
        assert elem_type is not None
        out = struct.pack("<IQ", int(elem_type), len(value))
        for item in value:
            out += _pack_value(item, elem_type)
        return out
    return struct.pack(_SCALAR_FMT[vtype], value)


def quantize_q8_0(arr: np.ndarray) -> bytes:
    """ggml Q8_0: per 32 contiguous elements, one fp16 scale then 32 int8 values.

    Blocks run along ``ne[0]`` -- the fastest-varying (input) dimension -- which for a
    row-major ``(out, in)`` weight is exactly 32 consecutive elements in memory.
    """
    flat = np.ascontiguousarray(arr, dtype=np.float32).reshape(-1)
    if flat.size % QK8_0 != 0:
        raise ValueError(f"Q8_0 needs a multiple of {QK8_0} elements, got {flat.size}")
    blocks = flat.reshape(-1, QK8_0)
    amax = np.abs(blocks).max(axis=1)
    d = (amax / 127.0).astype(np.float32)
    d_safe = np.where(d == 0, 1.0, d)
    q = np.rint(blocks / d_safe[:, None]).clip(-127, 127).astype(np.int8)
    out = bytearray()
    d16 = d.astype(np.float16)
    for i in range(blocks.shape[0]):
        out += d16[i].tobytes()
        out += q[i].tobytes()
    return bytes(out)


def dequantize_q8_0(raw: bytes, n_elements: int) -> np.ndarray:
    n_blocks = n_elements // QK8_0
    buf = np.frombuffer(raw, dtype=np.uint8).reshape(n_blocks, 2 + QK8_0)
    scales = buf[:, :2].copy().view(np.float16).astype(np.float32).reshape(-1, 1)
    qs = buf[:, 2:].copy().view(np.int8).astype(np.float32)
    return (qs * scales).reshape(-1)[:n_elements]


def permute_for_ggml_rope(w: Tensor, n_head: int) -> Tensor:
    """Rotate-half (ours / HF) -> interleaved-pair (ggml ``llama`` arch) weight layout.

    Our RoPE rotates ``[x_0..x_{d/2}]`` against ``[x_{d/2}..x_d]``; ggml's llama rope
    rotates adjacent pairs ``(x_0, x_1), (x_2, x_3), ...``. Reordering the *rows* of Wq
    and Wk makes the two conventions agree, which is precisely what llama.cpp's HF
    converter does. Omitting it produces a file that loads and generates fluent-looking
    garbage -- the worst kind of bug.
    """
    rows, cols = w.shape
    head_dim = rows // n_head
    return w.reshape(n_head, 2, head_dim // 2, cols).swapaxes(1, 2).reshape(rows, cols).contiguous()


def _llama_metadata(
    cfg: ModelConfig, quant: str, lossy_note: str | None
) -> dict[str, tuple[Any, GGUFValueType, GGUFValueType | None]]:
    md: dict[str, tuple[Any, GGUFValueType, GGUFValueType | None]] = {
        "general.architecture": ("llama", GGUFValueType.STRING, None),
        "general.name": (cfg.name, GGUFValueType.STRING, None),
        "general.file_type": (
            1 if quant == "f16" else (7 if quant == "q8_0" else 0),
            GGUFValueType.UINT32,
            None,
        ),
        "general.quantization_version": (2, GGUFValueType.UINT32, None),
        "general.alignment": (GGUF_DEFAULT_ALIGNMENT, GGUFValueType.UINT32, None),
        "llama.block_count": (cfg.n_layers, GGUFValueType.UINT32, None),
        "llama.context_length": (cfg.max_seq_len, GGUFValueType.UINT32, None),
        "llama.embedding_length": (cfg.d_model, GGUFValueType.UINT32, None),
        "llama.feed_forward_length": (cfg.ffn_hidden, GGUFValueType.UINT32, None),
        "llama.attention.head_count": (cfg.n_heads, GGUFValueType.UINT32, None),
        "llama.attention.head_count_kv": (cfg.n_kv_heads, GGUFValueType.UINT32, None),
        "llama.attention.layer_norm_rms_epsilon": (
            float(cfg.rms_norm_eps),
            GGUFValueType.FLOAT32,
            None,
        ),
        "llama.rope.dimension_count": (cfg.head_dim, GGUFValueType.UINT32, None),
        "llama.rope.freq_base": (float(cfg.rope_theta), GGUFValueType.FLOAT32, None),
        "llama.vocab_size": (cfg.vocab_size, GGUFValueType.UINT32, None),
        "localmind.export_verified_against_llama_cpp": (False, GGUFValueType.BOOL, None),
    }
    if lossy_note:
        md["localmind.lossy_export_note"] = (lossy_note, GGUFValueType.STRING, None)
    return md


def _tensor_entries(
    model: LocalMindTransformer, cfg: ModelConfig, quant: str
) -> list[tuple[str, Tensor]]:
    sd = model.state_dict()
    out: list[tuple[str, Tensor]] = [("token_embd.weight", sd["tok_emb.weight"])]
    for i in range(cfg.n_layers):
        p = f"blocks.{i}."
        out += [
            (f"blk.{i}.attn_norm.weight", sd[p + "attn_norm.w"]),
            (
                f"blk.{i}.attn_q.weight",
                permute_for_ggml_rope(sd[p + "attn.wq.weight"], cfg.n_heads),
            ),
            (
                f"blk.{i}.attn_k.weight",
                permute_for_ggml_rope(sd[p + "attn.wk.weight"], cfg.n_kv_heads),
            ),
            (f"blk.{i}.attn_v.weight", sd[p + "attn.wv.weight"]),
            (f"blk.{i}.attn_output.weight", sd[p + "attn.wo.weight"]),
            (f"blk.{i}.ffn_norm.weight", sd[p + "ffn_norm.w"]),
            (f"blk.{i}.ffn_gate.weight", sd[p + "ffn.gate.weight"]),
            (f"blk.{i}.ffn_up.weight", sd[p + "ffn.up.weight"]),
            (f"blk.{i}.ffn_down.weight", sd[p + "ffn.down.weight"]),
        ]
    out.append(("output_norm.weight", sd["norm_f.w"]))
    if not cfg.tie_embeddings:
        out.append(("output.weight", sd["lm_head.weight"]))
    return out


def _encode_tensor(name: str, t: Tensor, quant: str) -> GGUFTensor:
    arr = t.detach().cpu().float().numpy()
    # 1-D tensors (norm scales) always stay F32: ggml requires it and they are tiny.
    if arr.ndim == 1 or quant == "f32":
        return GGUFTensor(name, tuple(arr.shape), GGMLType.F32, np.ascontiguousarray(arr).tobytes())
    if quant == "f16":
        return GGUFTensor(
            name,
            tuple(arr.shape),
            GGMLType.F16,
            np.ascontiguousarray(arr.astype(np.float16)).tobytes(),
        )
    if quant == "q8_0":
        if arr.shape[-1] % QK8_0 != 0:
            return GGUFTensor(
                name, tuple(arr.shape), GGMLType.F32, np.ascontiguousarray(arr).tobytes()
            )
        return GGUFTensor(name, tuple(arr.shape), GGMLType.Q8_0, quantize_q8_0(arr))
    raise ValueError(f"unknown quant {quant!r}; use f32, f16 or q8_0")


def write_gguf(
    path: str | Path,
    model: LocalMindTransformer,
    quant: str = "q8_0",
    tokens: Sequence[str] | None = None,
    token_scores: Sequence[float] | None = None,
    bos_token_id: int = 1,
    eos_token_id: int = 2,
    allow_lossy: bool = False,
    extra_metadata: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Write a GGUF v3 file for llama.cpp. See the module docstring for what is verified.

    Args:
        quant: ``"f32"``, ``"f16"`` or ``"q8_0"``.
        tokens: vocabulary strings. Without them llama.cpp cannot tokenise, so the file
            is a weights-only artifact; that is recorded in the metadata.
        allow_lossy: required to proceed when the config uses features the ``llama``
            architecture cannot express (currently: QK-norm).
    """
    cfg = model.cfg
    lossy: list[str] = []
    if cfg.qk_norm:
        lossy.append(
            "qk_norm=true: llama.cpp's 'llama' architecture has no attn_q_norm/attn_k_norm "
            "tensors, so those RMSNorm scales are DROPPED and llama.cpp output will differ "
            "from LocalMind's. Use an architecture with QK-norm support (olmo2, gemma2) or "
            "fold the norms into Wq/Wk before trusting this file."
        )
    if lossy and not allow_lossy:
        raise ValueError(
            "refusing to write a silently-wrong GGUF: "
            + " | ".join(lossy)
            + " Pass allow_lossy=True to write it anyway."
        )

    entries = [_encode_tensor(n, t, quant) for n, t in _tensor_entries(model, cfg, quant)]
    metadata = _llama_metadata(cfg, quant, " | ".join(lossy) if lossy else None)
    if tokens is not None:
        metadata["tokenizer.ggml.model"] = ("llama", GGUFValueType.STRING, None)
        metadata["tokenizer.ggml.tokens"] = (
            list(tokens),
            GGUFValueType.ARRAY,
            GGUFValueType.STRING,
        )
        scores = list(token_scores) if token_scores is not None else [0.0] * len(tokens)
        metadata["tokenizer.ggml.scores"] = (scores, GGUFValueType.ARRAY, GGUFValueType.FLOAT32)
        metadata["tokenizer.ggml.token_type"] = (
            [1] * len(tokens),
            GGUFValueType.ARRAY,
            GGUFValueType.INT32,
        )
        metadata["tokenizer.ggml.bos_token_id"] = (bos_token_id, GGUFValueType.UINT32, None)
        metadata["tokenizer.ggml.eos_token_id"] = (eos_token_id, GGUFValueType.UINT32, None)
    else:
        metadata["localmind.tokenizer_embedded"] = (False, GGUFValueType.BOOL, None)
    for k, v in (extra_metadata or {}).items():
        metadata[k] = (v, GGUFValueType.STRING, None)

    header = bytearray(GGUF_MAGIC)
    header += struct.pack("<I", GGUF_VERSION)
    header += struct.pack("<QQ", len(entries), len(metadata))
    for key, (value, vtype, elem) in metadata.items():
        header += _pack_string(key)
        header += struct.pack("<I", int(vtype))
        header += _pack_value(value, vtype, elem)

    # Tensor descriptors carry offsets relative to the start of the (aligned) data blob,
    # so the descriptor block must be built with the offsets already known.
    infos = bytearray()
    offset = 0
    offsets: list[int] = []
    for t in entries:
        offsets.append(offset)
        infos += _pack_string(t.name)
        infos += struct.pack("<I", len(t.shape))
        for d in reversed(t.shape):  # GGUF stores ne[] fastest-varying first
            infos += struct.pack("<Q", int(d))
        infos += struct.pack("<I", int(t.ggml_type))
        infos += struct.pack("<Q", offset)
        offset += len(t.data)
        offset += (-offset) % GGUF_DEFAULT_ALIGNMENT

    pre_data = bytes(header) + bytes(infos)
    pad = (-len(pre_data)) % GGUF_DEFAULT_ALIGNMENT

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as fh:
        fh.write(pre_data)
        fh.write(b"\x00" * pad)
        for t, off in zip(entries, offsets, strict=True):
            fh.write(t.data)
            fh.write(b"\x00" * ((-len(t.data)) % GGUF_DEFAULT_ALIGNMENT))
            del off
    size = out.stat().st_size
    return {
        "path": str(out),
        "bytes": size,
        "mb": size / 1024**2,
        "quant": quant,
        "n_tensors": len(entries),
        "n_metadata": len(metadata),
        "lossy": lossy,
        "verified_against_llama_cpp": False,
    }


class _Reader:
    def __init__(self, buf: bytes) -> None:
        self.buf = buf
        self.pos = 0

    def take(self, n: int) -> bytes:
        b = self.buf[self.pos : self.pos + n]
        self.pos += n
        return b

    def scalar(self, fmt: str) -> Any:
        n = struct.calcsize(fmt)
        return struct.unpack(fmt, self.take(n))[0]

    def string(self) -> str:
        n = self.scalar("<Q")
        return self.take(n).decode("utf-8")

    def value(self, vtype: GGUFValueType) -> Any:
        if vtype == GGUFValueType.STRING:
            return self.string()
        if vtype == GGUFValueType.ARRAY:
            elem = GGUFValueType(self.scalar("<I"))
            count = self.scalar("<Q")
            return [self.value(elem) for _ in range(count)]
        return self.scalar(_SCALAR_FMT[vtype])


def read_gguf(path: str | Path, load_data: bool = True) -> dict[str, Any]:
    """Parse a GGUF file back into metadata + tensors. Used to round-trip-test the writer."""
    raw = Path(path).read_bytes()
    r = _Reader(raw)
    if r.take(4) != GGUF_MAGIC:
        raise ValueError("not a GGUF file: bad magic")
    version = r.scalar("<I")
    n_tensors = r.scalar("<Q")
    n_meta = r.scalar("<Q")
    metadata: dict[str, Any] = {}
    for _ in range(n_meta):
        key = r.string()
        vtype = GGUFValueType(r.scalar("<I"))
        metadata[key] = r.value(vtype)
    tensors: list[dict[str, Any]] = []
    for _ in range(n_tensors):
        name = r.string()
        n_dims = r.scalar("<I")
        ne = [r.scalar("<Q") for _ in range(n_dims)]
        ggml_type = GGMLType(r.scalar("<I"))
        off = r.scalar("<Q")
        tensors.append(
            {"name": name, "ne": ne, "shape": tuple(reversed(ne)), "type": ggml_type, "offset": off}
        )
    align = int(metadata.get("general.alignment", GGUF_DEFAULT_ALIGNMENT))
    data_start = r.pos + ((-r.pos) % align)

    if load_data:
        for t in tensors:
            n_elem = 1
            for d in t["ne"]:
                n_elem *= int(d)
            start = data_start + int(t["offset"])
            if t["type"] == GGMLType.F32:
                blob = raw[start : start + n_elem * 4]
                t["array"] = np.frombuffer(blob, dtype=np.float32).reshape(t["shape"])
            elif t["type"] == GGMLType.F16:
                blob = raw[start : start + n_elem * 2]
                t["array"] = (
                    np.frombuffer(blob, dtype=np.float16).astype(np.float32).reshape(t["shape"])
                )
            elif t["type"] == GGMLType.Q8_0:
                n_blocks = n_elem // QK8_0
                blob = raw[start : start + n_blocks * (2 + QK8_0)]
                t["array"] = dequantize_q8_0(blob, n_elem).reshape(t["shape"])
    return {
        "version": version,
        "n_tensors": n_tensors,
        "metadata": metadata,
        "tensors": tensors,
        "data_start": data_start,
        "file_bytes": len(raw),
    }
