"""Speculative decoding: n-gram / prompt-lookup first, then a draft model.

implementation.md section 10 step 7.

The idea
--------
Decode is memory-bandwidth-bound: one token costs a full read of the weights. A forward
over *k* tokens costs almost the same as a forward over 1, because the weights are read
once either way. So: cheaply guess k tokens, verify all of them in one target forward,
and keep however many the target agrees with. Latency falls by roughly the mean number of
accepted tokens per iteration; total FLOPs go *up*, which is the trade.

Two proposers
-------------
:class:`NgramProposer` (prompt lookup, Saxena 2023) uses **no second model at all**: it
finds the most recent earlier occurrence of the last n tokens in the context and proposes
whatever followed it. For RAG this is unusually strong, because a grounded answer quotes
its context -- names, numbers, and phrases are literally copied, and those are exactly the
spans a lookup nails.

:class:`DraftModelProposer` runs a genuinely smaller transformer. More general, but it
costs a second model's weights and its own decode loop, so the acceptance rate has to pay
for that too.

Correctness
-----------
Verification is the Leviathan et al. (2023) / Chen et al. (2023) modified rejection
sampling, which makes the output distribution **identical** to the non-speculative one:
accept draft token x with probability ``min(1, p(x)/q(x))``, and on rejection sample from
the normalised residual ``max(0, p - q)``. For a deterministic proposer (n-gram) ``q`` is
a point mass, so the accept probability is ``p(x)`` and the residual is ``p`` with that
token zeroed. Under greedy decoding the rule collapses to "accept while the draft agrees
with the argmax", and the test suite asserts the speculative output is **token-for-token
identical** to :class:`~localmind.inference.engine.CachedEngine` -- which is the only
honest way to claim a speedup.

The threshold
-------------
implementation.md warns that below roughly **0.6 acceptance** there is usually no win: the
extra verification positions and the proposer's own cost outweigh the tokens saved. That
is reported as measured, in both directions.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import torch
from torch import Tensor

from localmind.inference.kv_cache import ContiguousKVCache
from localmind.inference.sampling import SamplingParams, make_generator, prepare_logits
from localmind.model import LocalMindTransformer

__all__ = [
    "DraftModelProposer",
    "NgramProposer",
    "Proposal",
    "Proposer",
    "SpeculativeResult",
    "speculative_generate",
]


@dataclass
class Proposal:
    """``k`` guessed tokens and, when the proposer is stochastic, their distributions."""

    tokens: list[int]
    #: ``(k, vocab)`` draft probabilities, or ``None`` for a deterministic proposer
    #: (which is a point mass and needs no explicit row).
    probs: Tensor | None = None


@runtime_checkable
class Proposer(Protocol):
    name: str

    def propose(self, ids: Sequence[int], k: int) -> Proposal: ...
    def reset(self) -> None: ...


class NgramProposer:
    """Prompt-lookup decoding. No model, no weights, no warmup.

    Scans for the most recent earlier occurrence of the last ``n`` tokens (largest ``n``
    first, so a longer, more specific match wins) and proposes the continuation.
    """

    name = "ngram"

    def __init__(self, max_ngram: int = 4, min_ngram: int = 2, num_speculative: int = 4) -> None:
        self.max_ngram = max_ngram
        self.min_ngram = min_ngram
        self.num_speculative = num_speculative
        self.calls = 0
        self.empty = 0

    def reset(self) -> None:
        self.calls = 0
        self.empty = 0

    def propose(self, ids: Sequence[int], k: int) -> Proposal:
        self.calls += 1
        seq = list(ids)
        for n in range(self.max_ngram, self.min_ngram - 1, -1):
            if len(seq) < n + 1:
                continue
            pattern = seq[-n:]
            for start in range(len(seq) - n - 1, -1, -1):
                if seq[start : start + n] == pattern:
                    cand = seq[start + n : start + n + k]
                    if cand:
                        return Proposal(tokens=cand)
        self.empty += 1
        return Proposal(tokens=[])


class DraftModelProposer:
    """A smaller transformer proposing tokens with its own incremental KV cache.

    After each ``propose`` the draft cache is rolled back to the last *committed*
    position, because the proposals it just made may be rejected. The accepted tokens are
    then re-forwarded on the next call. That costs a few extra draft forwards and keeps
    the draft state provably consistent with the target's committed sequence -- the
    alternative (speculatively advancing the draft cache) is where speculative decoding
    implementations usually go subtly wrong.
    """

    name = "draft_model"

    def __init__(
        self,
        draft: LocalMindTransformer,
        params: SamplingParams | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.model = draft.eval()
        self.cfg = draft.cfg
        self.device = torch.device(device)
        self.params = params or SamplingParams(temperature=0.0)
        self.cache = ContiguousKVCache(self.cfg, dtype=dtype, device=device)
        self.calls = 0
        self.forwards = 0

    def reset(self) -> None:
        self.cache.reset()
        self.calls = 0
        self.forwards = 0

    def _forward(self, tokens: Sequence[int]) -> Tensor:
        ids = torch.tensor([list(tokens)], dtype=torch.long, device=self.device)
        with torch.no_grad():
            out = self.model(ids, past_kvs=self.cache.as_past(), use_cache=True)
        assert out.kv_caches is not None
        self.cache.extend_from(out.kv_caches)
        self.forwards += 1
        return out.logits[0]

    def propose(self, ids: Sequence[int], k: int) -> Proposal:
        self.calls += 1
        seq = list(ids)
        committed = len(seq) - 1  # cache should cover seq[:-1]
        if self.cache.length > committed:
            self.cache.length = committed
        if self.cache.length < committed:
            self._forward(seq[self.cache.length : committed])
        gen = make_generator(self.params.seed + len(seq), self.device)

        tokens: list[int] = []
        rows: list[Tensor] = []
        cur = seq[-1]
        for _ in range(k):
            logits = self._forward([cur])[-1]
            processed = prepare_logits(logits, self.params, seq + tokens)
            probs = torch.softmax(processed, dim=-1)
            if self.params.greedy:
                nxt = int(torch.argmax(processed).item())
            else:
                nxt = int(torch.multinomial(probs, 1, generator=gen).item())
            rows.append(probs)
            tokens.append(nxt)
            cur = nxt
        self.cache.length = committed  # roll back: proposals are not committed
        return Proposal(tokens=tokens, probs=torch.stack(rows) if rows else None)


@dataclass
class SpeculativeResult:
    """Tokens plus the numbers that decide whether speculation was worth it."""

    token_ids: list[int]
    proposed: int = 0
    accepted: int = 0
    bonus: int = 0
    iterations: int = 0
    target_forwards: int = 0
    target_positions: int = 0
    ttft_s: float = 0.0
    total_s: float = 0.0
    finish_reason: str = "length"
    proposer: str = "none"
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def acceptance_rate(self) -> float:
        """Fraction of proposed tokens the target kept. The number section 10 asks for."""
        return self.accepted / self.proposed if self.proposed else 0.0

    @property
    def tokens_per_iteration(self) -> float:
        """Mean tokens committed per target forward -- the theoretical latency divisor."""
        return len(self.token_ids) / self.iterations if self.iterations else 0.0

    @property
    def tokens_per_s(self) -> float:
        return len(self.token_ids) / self.total_s if self.total_s > 0 else float("nan")

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposer": self.proposer,
            "n_generated": len(self.token_ids),
            "proposed": self.proposed,
            "accepted": self.accepted,
            "acceptance_rate": self.acceptance_rate,
            "tokens_per_iteration": self.tokens_per_iteration,
            "iterations": self.iterations,
            "target_forwards": self.target_forwards,
            "target_positions": self.target_positions,
            "ttft_s": self.ttft_s,
            "total_s": self.total_s,
            "tokens_per_s": self.tokens_per_s,
            "finish_reason": self.finish_reason,
            **self.extra,
        }


def _residual_sample(
    p: Tensor, q_row: Tensor | None, rejected: int, generator: torch.Generator
) -> int:
    """Sample from ``normalise(max(0, p - q))`` -- the corrected distribution on rejection."""
    if q_row is None:
        resid = p.clone()
        resid[rejected] = 0.0
    else:
        resid = torch.clamp(p - q_row, min=0.0)
    total = float(resid.sum().item())
    if total <= 0.0:
        return int(torch.argmax(p).item())
    resid = resid / total
    return int(torch.multinomial(resid, 1, generator=generator).item())


def speculative_generate(
    target: LocalMindTransformer,
    proposer: Proposer,
    prompt_ids: Sequence[int],
    params: SamplingParams,
    num_speculative: int = 4,
    eos_token_id: int | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> SpeculativeResult:
    """Draft-and-verify decoding with exact distribution preservation.

    Invariant maintained throughout: the target KV cache covers ``ids[:-1]``, so the next
    forward always begins with the last committed token and every logit row produced lines
    up with a verification position.
    """
    dev = torch.device(device)
    cfg = target.cfg
    target.eval()
    proposer.reset()
    cache = ContiguousKVCache(cfg, dtype=dtype, device=dev)
    gen = make_generator(params.seed, dev)

    ids = list(prompt_ids)
    if len(ids) < 2:
        raise ValueError("speculative decoding needs a prompt of at least 2 tokens")
    out_ids: list[int] = []
    res = SpeculativeResult(token_ids=out_ids, proposer=getattr(proposer, "name", "unknown"))
    t_start = time.perf_counter()

    def fwd(tokens: Sequence[int]) -> Tensor:
        t = torch.tensor([list(tokens)], dtype=torch.long, device=dev)
        with torch.no_grad():
            out = target(t, past_kvs=cache.as_past(), use_cache=True)
        assert out.kv_caches is not None
        cache.extend_from(out.kv_caches)
        res.target_forwards += 1
        res.target_positions += len(tokens)
        return out.logits[0]

    # Prefill everything except the last token: the invariant.
    fwd(ids[:-1])

    stop = False
    while len(out_ids) < params.max_new_tokens and not stop:
        res.iterations += 1
        k = min(num_speculative, params.max_new_tokens - len(out_ids))
        proposal = proposer.propose(ids, k) if k > 0 else Proposal(tokens=[])
        draft = proposal.tokens[:k]
        res.proposed += len(draft)

        logits = fwd([ids[-1], *draft])  # (1 + len(draft), vocab)
        base_len = len(ids) - 1  # cache position of ids[-1]

        n_accept = 0
        emitted: list[int] = []
        for j, d in enumerate(draft):
            row = prepare_logits(logits[j], params, ids + emitted)
            if params.greedy:
                if int(torch.argmax(row).item()) != d:
                    break
                emitted.append(d)
                n_accept += 1
                continue
            p = torch.softmax(row, dim=-1)
            q = 1.0 if proposal.probs is None else float(proposal.probs[j, d].item())
            accept_p = 1.0 if q <= 0 else min(1.0, float(p[d].item()) / q)
            if float(torch.rand((), generator=gen, device=dev).item()) < accept_p:
                emitted.append(d)
                n_accept += 1
                continue
            break
        res.accepted += n_accept

        # One extra token always comes out: either the correction at the rejection
        # point, or the bonus token after a fully accepted block. This is why
        # speculation never makes progress *slower* than 1 token per iteration.
        j = n_accept
        row = prepare_logits(logits[j], params, ids + emitted)
        if j < len(draft):
            if params.greedy:
                extra = int(torch.argmax(row).item())
            else:
                p = torch.softmax(row, dim=-1)
                q_row = None if proposal.probs is None else proposal.probs[j]
                extra = _residual_sample(p, q_row, draft[j], gen)
        else:
            res.bonus += 1
            if params.greedy:
                extra = int(torch.argmax(row).item())
            else:
                extra = int(torch.multinomial(torch.softmax(row, dim=-1), 1, generator=gen).item())
        emitted.append(extra)

        # Roll the cache back to exactly the committed tokens.
        cache.length = base_len + n_accept + 1
        for tok in emitted:
            if len(out_ids) >= params.max_new_tokens:
                break
            out_ids.append(tok)
            ids.append(tok)
            if res.ttft_s == 0.0:
                res.ttft_s = time.perf_counter() - t_start
            if tok in params.stop_token_ids or (
                eos_token_id is not None and tok == eos_token_id and not params.ignore_eos
            ):
                stop = True
                res.finish_reason = "stop"
                break
        if len(ids) >= cfg.max_seq_len - num_speculative - 2:
            res.finish_reason = "length"
            break

    res.total_s = time.perf_counter() - t_start
    if isinstance(proposer, NgramProposer):
        res.extra["proposer_calls"] = proposer.calls
        res.extra["proposer_empty"] = proposer.empty
    if isinstance(proposer, DraftModelProposer):
        res.extra["draft_forwards"] = proposer.forwards
    return res
