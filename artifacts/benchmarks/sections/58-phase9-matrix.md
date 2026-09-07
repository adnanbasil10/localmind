## Phase 9 - the distilled model, measured (§9 5e)

Base: v1 (`pytorch_model_step5722.pt`, held-out val ce 2.6891). Teacher: **Qwen2.5-3B-Instruct**
via `transformers` (vLLM will not install on Kaggle). 3,000 teacher examples, 0% blank. KD capped at
250 steps with a 10% holdout guard. Evaluated on **60 held-out prompts per job**, greedy decoding,
2x T4. **measured**.

### Results

| Job | n | Format-valid | Correct vs teacher | p50 latency |
|---|---|---|---|---|
| **router** | 60 | **100.0%** | **96.7%** | 126.9 ms |
| **grader** | 60 | **100.0%** | **100.0%** | 203.6 ms |
| **rewriter** | 60 | **100.0%** | **100.0%** | 197.9 ms |

Sample predictions:

```
router  pred {"route": "needs_web"}              gold {"route": "needs_web"}
grader  pred {"label": "relevant", "score": 0.85} gold {"label": "relevant", "score": 0.9}
```

The grader answering `0.85` where the teacher said `0.9` is the useful detail: the student is
producing a calibrated score of its own rather than replaying the teacher's token. Correctness is
scored on the label, not the float, for exactly that reason.

### Distillation quality

| | Value |
|---|---|
| KD train loss | 0.0014 |
| KD holdout ce | 0.0306 |
| **KD holdout gap** | **+0.029** (bar: 1.0) — generalises, no memorisation warning |

The fake-teacher run reached a near-identical training loss (0.0012) and was degenerate. The two are
distinguishable *only* by the holdout gap, which is why that check exists.

### GRPO: at ceiling, not broken

| | Fake teacher | Real teacher |
|---|---|---|
| Format-valid | 0.0% | **100.0%** |
| Correct | 0.0% | **100.0%** |
| Degenerate groups | 100% | 100% |

Both runs report 100% degenerate groups and both report `loss 0.0000 -> 0.0000`, and they mean
opposite things. With the fake teacher every group scored 0 — the policy could not produce parseable
output at all. With the real teacher every group scores 1 — the policy solves the task on every
rollout, so the group-relative advantage is zero because there is nothing left to separate.

**GRPO has no gradient here because the task is saturated.** That is not a failure of GRPO; it is a
signal that the grader job needs harder prompts before RL can add anything.

### What is NOT established

- **DPO is still unmeasured.** It ran, but on **synthetic** preference pairs (its own warning says
  so), and its KL hit 1,086,489 against a 0.5 bar. Nothing here shows DPO helped, and the
  `dpo_policy.pt` checkpoint should not be used.
- **Correctness is measured against the teacher, not ground truth.** 100% agreement with
  Qwen2.5-3B means the distillation worked; it does not mean the teacher was right.
- **Held-out prompts, not a held-out task.** The 60 prompts per job are unseen, but drawn from the
  same seed pool and distribution as training.
- **One seed, no CI**, unlike the retrieval and inference tables.
- Latency is **GPU** (T4). §9's claim is CPU-resident inference at p50 < 20 ms; that has not been
  measured for this checkpoint.

### Artifacts

`artifacts/post/kd/kd_sequence.pt`, `artifacts/post/grpo/grpo_policy.pt`, and
`/kaggle/working/post/matrix_5e.json`. Teacher data: 3,000 rows from Qwen2.5-3B-Instruct.
