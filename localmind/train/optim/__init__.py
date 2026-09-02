"""Optimizers for the §8 study: AdamW baseline, Muon, and AdamW at 2x LR.

`build_optimizer_arm(model, arm, lr, ...)` is the single entry point the training loop
uses; it returns a **list** of optimizers because the Muon arm is two of them (Muon on
the 2-D hidden matrices, AdamW on embeddings / LM head / norms -- the Kimi K2 split).

The parameter grouping (`adamw.split_param_groups`) is shared by every arm and
implements §8's "wd=0.1, exclude norms/bias/embeddings" by module type rather than by a
name heuristic, because with ``tie_embeddings: true`` a ``p.ndim >= 2`` rule would decay
the embedding matrix -- 8.4M of a 31M model.
"""

from localmind.train.optim.adamw import (
    ParamGroupSpec,
    build_adamw,
    fused_adamw_available,
    iter_trainable_params,
    set_lr,
    split_param_groups,
)
from localmind.train.optim.muon import (
    NS_COEFFS,
    NS_DTYPES,
    Muon,
    MuonParamSplit,
    UpdateScale,
    build_muon_hybrid,
    build_optimizer_arm,
    muon_update_scale,
    newton_schulz,
    newton_schulz_error,
    split_muon_params,
)

__all__ = [
    "NS_COEFFS",
    "NS_DTYPES",
    "Muon",
    "MuonParamSplit",
    "ParamGroupSpec",
    "UpdateScale",
    "build_adamw",
    "build_muon_hybrid",
    "build_optimizer_arm",
    "fused_adamw_available",
    "iter_trainable_params",
    "muon_update_scale",
    "newton_schulz",
    "newton_schulz_error",
    "set_lr",
    "split_muon_params",
    "split_param_groups",
]
