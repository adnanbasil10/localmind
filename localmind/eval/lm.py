"""Score a trained checkpoint on a held-out split — did it learn, or did it memorise?

The §8 run reported `ce_loss 2.4509`, but that is a **training** loss over a corpus the model
saw roughly 30 times. A training loss cannot distinguish a model that learned the language from
one that memorised the shards. Only a held-out split can, and the run had none.

    python -m localmind.eval.lm --checkpoint <path-or-hub-id> --shards /kaggle/working/shards

Reports train loss, val loss, the generalisation gap, and bits-per-byte for both. The verdict
thresholds are deliberately conservative and stated in the output rather than hidden in a
constant, because "did this work" should not be a judgement the reader has to take on trust.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

__all__ = ["GAP_LEARNED", "GAP_MEMORISED", "score_checkpoint", "verdict_for_gap"]

#: A gap this small means val tracks train: the model generalises.
GAP_LEARNED = 0.30
#: A gap this large means the model is reciting shards it has seen ~30 times.
GAP_MEMORISED = 1.00


def verdict_for_gap(gap: float) -> str:
    if gap < GAP_LEARNED:
        return (
            f"LEARNED — val tracks train (gap {gap:.4f} < {GAP_LEARNED}). More epochs on this "
            "corpus are a reasonable use of GPU time."
        )
    if gap > GAP_MEMORISED:
        return (
            f"MEMORISED — val is far worse than train (gap {gap:.4f} > {GAP_MEMORISED}). More "
            "epochs will not help; the corpus needs to be larger, not seen more times."
        )
    return (
        f"PARTIAL — gap {gap:.4f} sits between {GAP_LEARNED} and {GAP_MEMORISED}. Some real "
        "learning plus some recall. A larger corpus would likely pay off more than more steps."
    )


def _shard_seq_len(root: Path) -> int | None:
    """The seq_len the shards were packed at, from the manifest or an .idx sidecar."""
    manifest = root / "manifest.json"
    if manifest.exists():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        shards = data.get("shards") or []
        if shards and "seq_len" in shards[0]:
            return int(shards[0]["seq_len"])
    for idx in sorted(root.glob("*.idx")):
        return int(json.loads(idx.read_text(encoding="utf-8"))["seq_len"])
    return None


def _resolve_checkpoint(spec: str) -> Path:
    """Accept a local path, or `user/repo[:filename]` to pull from the HF Hub."""
    p = Path(spec)
    if p.exists():
        return p
    if "/" not in spec:
        raise FileNotFoundError(f"no such checkpoint: {spec}")
    from huggingface_hub import hf_hub_download

    repo, _, filename = spec.partition(":")
    filename = filename or "pytorch_model_step5722.pt"
    print(f"[eval-lm] downloading {filename} from {repo} ...", flush=True)
    return Path(hf_hub_download(repo_id=repo, filename=filename))


def score_checkpoint(
    checkpoint: str,
    shard_dir: str,
    *,
    model_config: str = "configs/model/31m.yaml",
    batch_size: int = 8,
    iters: int = 50,
    device: str | None = None,
) -> dict[str, Any]:
    import torch

    from localmind.data.loader import build_loaders
    from localmind.model import LocalMindTransformer, ModelConfig

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = _resolve_checkpoint(checkpoint)

    cfg = ModelConfig.from_yaml(model_config)
    model = LocalMindTransformer(cfg).to(dev).eval()

    blob = torch.load(ckpt_path, map_location=dev, weights_only=False)
    state = blob.get("model", blob) if isinstance(blob, dict) else blob
    # DDP checkpoints carry a `module.` prefix the bare model does not have.
    state = {k.removeprefix("module."): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[eval-lm] WARNING: {len(missing)} missing keys, e.g. {missing[:3]}")
    if unexpected:
        print(f"[eval-lm] WARNING: {len(unexpected)} unexpected keys, e.g. {unexpected[:3]}")

    root = Path(shard_dir)
    val_dir = root / "val"
    if not val_dir.exists():
        raise FileNotFoundError(
            f"no validation split at {val_dir}. Rebuild the data with:\n"
            f"    python -m localmind.data.prepare --val-docs 2000 --out {root} ...\n"
            "Without a held-out split, a training loss cannot tell learning from memorisation."
        )

    # Read seq_len from the shards, not the model config: a model with max_seq_len 1024 can
    # legitimately be scored on shards packed at 256, and build_loaders rejects a mismatch.
    packed_seq_len = _shard_seq_len(root) or cfg.max_seq_len
    if packed_seq_len > cfg.max_seq_len:
        raise ValueError(
            f"shards are packed at seq_len={packed_seq_len}, longer than the model's "
            f"max_seq_len={cfg.max_seq_len}; it cannot attend that far."
        )
    train_loader, val_loader = build_loaders(
        seq_len=packed_seq_len, micro_batch_size=batch_size, seed=0, shard_dir=root
    )
    assert val_loader is not None

    @torch.no_grad()
    def mean_loss(loader: Any, n: int) -> float:
        total, count = 0.0, 0
        for _ in range(n):
            x, y, _doc = next(loader)
            out = model(x.to(dev), targets=y.to(dev))
            total += float(out.ce_loss)
            count += 1
        return total / max(1, count)

    train_ce = mean_loss(train_loader, iters)
    val_ce = mean_loss(val_loader, iters)
    gap = val_ce - train_ce

    # bits-per-byte is the tokenizer-invariant figure; ce is in nats per token.
    bpt = getattr(val_loader, "bytes_per_token", None) or 4.0
    to_bpb = lambda ce: ce / (math.log(2) * bpt)  # noqa: E731

    return {
        "checkpoint": str(ckpt_path),
        "device": dev,
        "iters": iters,
        "batch_size": batch_size,
        "train_ce": round(train_ce, 4),
        "val_ce": round(val_ce, 4),
        "gap": round(gap, 4),
        "train_bpb": round(to_bpb(train_ce), 4),
        "val_bpb": round(to_bpb(val_ce), 4),
        "bytes_per_token": bpt,
        "verdict": verdict_for_gap(gap),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m localmind.eval.lm",
        description="Score a checkpoint on a held-out split: learned, or memorised?",
    )
    ap.add_argument("--checkpoint", required=True, help="local path, or 'user/repo[:file]'")
    ap.add_argument("--shards", required=True, help="shard dir containing a val/ subdirectory")
    ap.add_argument("--model-config", default="configs/model/31m.yaml")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="artifacts/benchmarks/lm_holdout.json")
    args = ap.parse_args(argv)

    res = score_checkpoint(
        args.checkpoint,
        args.shards,
        model_config=args.model_config,
        batch_size=args.batch_size,
        iters=args.iters,
        device=args.device,
    )
    print()
    print(f"  train ce   {res['train_ce']:.4f}   bpb {res['train_bpb']:.4f}")
    print(f"  val   ce   {res['val_ce']:.4f}   bpb {res['val_bpb']:.4f}")
    print(f"  gap        {res['gap']:.4f}")
    print()
    print(f"  {res['verdict']}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
