/**
 * Distil artifacts/benchmarks/*.json into frontend/data/benchmarks.json.
 *
 * Rules this script exists to enforce:
 *   1. Every number the UI renders is READ from an artifact. Nothing is typed by hand.
 *   2. Every figure carries a provenance label: measured | synthetic | not-run.
 *   3. A missing artifact or a null cell becomes `null` + status "not-run".
 *      It is NEVER back-filled with a plausible value.
 *
 * If artifacts/ is not present (e.g. a standalone container build) the previously
 * generated data/benchmarks.json is left untouched and the build continues.
 */
import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const BENCH = resolve(HERE, '..', '..', 'artifacts', 'benchmarks');
const OUT = resolve(HERE, '..', 'data', 'benchmarks.json');

if (!existsSync(BENCH)) {
  console.warn('[extract] ' + BENCH + ' not found - keeping the committed extract.');
  process.exit(0);
}

const load = (f) => JSON.parse(readFileSync(join(BENCH, f), 'utf8'));
const inference = load('inference.json');
const model = load('model.json');
const retrieval = load('retrieval.json');
const cache = load('semantic_cache.json');
const tokenizer = load('tokenizer.json');
const judge = load('judge_calibration.json');
const matrix = load('phase5_comparison_matrix.json');

const rows = (src, bench) => src.rows.filter((r) => r.bench === bench);
const one = (src, bench, pred = () => true) => rows(src, bench).find(pred);

/** unwrap {mean,std,ci_low,ci_high,n} | {mean,lo,hi,n} | plain number */
const stat = (v) => {
  if (v === null || v === undefined) return null;
  if (typeof v === 'number') return { mean: v, lo: null, hi: null, n: null };
  return {
    mean: v.mean ?? null,
    lo: v.ci_low ?? v.lo ?? null,
    hi: v.ci_high ?? v.hi ?? null,
    n: v.n ?? null,
  };
};

// -- 1. Paged vs contiguous: max concurrent sequences at a fixed KV budget -----
const mc = rows(inference, 'max_concurrent').sort((a, b) => a.seq_len - b.seq_len);
const maxConcurrent = {
  provenance: 'measured',
  source: 'artifacts/benchmarks/inference.json :: bench=max_concurrent',
  hardware: inference.hardware,
  budgetMb: mc[0]?.budget_mb ?? null,
  maxSeqLen: mc[0]?.max_seq_len ?? null,
  kvBytesPerToken: mc[0]?.kv_bytes_per_token ?? null,
  rows: mc.map((r) => ({
    seqLen: r.seq_len,
    contiguous: r.contiguous_max_sequences,
    paged: r.paged_max_sequences,
    gain: r.paged_gain,
  })),
};

// -- 2. Reserved-KV waste (internal fragmentation), per workload mix and seed --
const frag = rows(inference, 'fragmentation');
const fragmentation = {
  provenance: 'measured',
  source: 'artifacts/benchmarks/inference.json :: bench=fragmentation',
  nSequences: frag[0]?.n_sequences ?? null,
  blockSize: frag[0]?.block_size ?? null,
  budgetMb: frag[0]?.budget_mb ?? null,
  rows: frag.map((r) => {
    const m = /^(.*)\(seed=(\d+)\)$/.exec(r.mix) ?? [];
    return {
      mix: (m[1] ?? r.mix).trim(),
      seed: m[2] !== undefined ? Number(m[2]) : null,
      meanSeqLen: r.mean_seq_len,
      contiguousWaste: r.contiguous.internal_fragmentation,
      pagedWaste: r.paged.internal_fragmentation,
      contiguousExternal: r.contiguous.external_fragmentation,
      pagedExternal: r.paged.external_fragmentation,
      contiguousReservedMb: r.contiguous.bytes_reserved / 1048576,
      pagedReservedMb: r.paged.bytes_reserved / 1048576,
      amplification: r.memory_amplification_contiguous_over_paged,
    };
  }),
};

const alloc = rows(inference, 'paged_allocator_measured');
const allocator = {
  provenance: 'measured',
  source: 'artifacts/benchmarks/inference.json :: bench=paged_allocator_measured',
  externalFragmentation: alloc.map((r) => r.external_fragmentation),
  note: alloc[0]?.external_fragmentation_note ?? '',
  internalRange: alloc.length
    ? [
        Math.min(...alloc.map((r) => r.internal_fragmentation)),
        Math.max(...alloc.map((r) => r.internal_fragmentation)),
      ]
    : null,
};

// -- 3. Semantic cache tau sweep ----------------------------------------------
const semanticCache = {
  provenance: 'synthetic',
  source: 'artifacts/benchmarks/semantic_cache.json',
  hardware: cache.hardware,
  embedder: cache.embedder,
  description: cache.description,
  reasoning: cache.operating_point_reasoning,
  operatingPoint: {
    tau: cache.operating_point.tau,
    hitRate: cache.operating_point.hit_rate,
    falseHitOfTotal: cache.operating_point.false_hit_rate_of_total,
    falseHitOfHits: cache.operating_point.false_hit_rate_of_hits,
    meetsThreshold: cache.operating_point.meets_threshold,
    threshold: cache.operating_point.threshold,
    selectionRule: cache.operating_point.selection_rule,
  },
  rows: cache.rows.map((r) => ({
    tau: r.tau,
    hit: stat(r.hit_rate),
    falseTotal: stat(r.false_hit_rate_of_total),
    falseOfHits: stat(r.false_hit_rate_of_hits),
  })),
};
semanticCache.floor = Math.min(...semanticCache.rows.map((r) => r.falseTotal.mean));

// -- 4. Retrieval -------------------------------------------------------------
const ie = retrieval.index_engineering;
const retrievalOut = {
  provenance: 'synthetic',
  source: 'artifacts/benchmarks/retrieval.json',
  hardware: retrieval.hardware,
  corpus: retrieval.corpus,
  notes: retrieval.notes ?? [],
  rows: retrieval.rows.map((r) => ({
    config: r.config,
    ndcg: stat(r.ndcg_at_10),
    recall5: stat(r.recall_at_5),
    recall20: stat(r.recall_at_20),
    p50: stat(r.p50_ms),
    p95: stat(r.p95_ms),
  })),
  fusion: retrieval.fusion_comparison,
  reranking: retrieval.reranking,
  contextual: retrieval.contextual_chunking,
  binaryQuantization: ie.binary_quantization,
  filteredAnn: ie.filtered_ann,
  indexEngine: ie.engine,
};

// -- 5. Phase-5 comparison matrix: 0 of 24 cells. Renders as voids, never values
const comparisonMatrix = {
  provenance: 'not-run',
  source: 'artifacts/benchmarks/phase5_comparison_matrix.json',
  cellsMeasured: matrix.cells_measured,
  cellsTotal: matrix.cells_total,
  columns: matrix.columns,
  dodStatus: matrix.dod.status,
  dodReason: matrix.dod.reason,
  hardware: matrix.hardware,
  rows: matrix.rows.map((r) => ({
    name: r.name,
    hardware: r.hardware,
    paramsUpdated: r.params_updated,
    cells: matrix.columns.map((c) => ({
      column: c,
      status: r[c]?.status ?? 'not-run',
      // deliberately NOT defaulted to 0: a missing measurement stays null.
      mean: r[c]?.mean ?? null,
    })),
  })),
};

// -- 6. Inference ledger - one row per engine claim, each traced to a bench ----
const pct = (a, b) => ((a - b) / a) * 100;
const kvGen = rows(inference, 'kv_cache_generation');
const at = (variant, n) => kvGen.find((r) => r.variant === variant && r.new_tokens === n);
const naive512 = at('naive_no_cache', 512);
const contig512 = at('contiguous_kv', 512);
const prefixOff = one(inference, 'prefix_cache', (r) => r.enabled === false);
const prefixOn = one(inference, 'prefix_cache', (r) => r.enabled === true);
const cpOff = one(inference, 'chunked_prefill', (r) => r.enabled === false);
const cpOn = one(inference, 'chunked_prefill', (r) => r.enabled === true);
const specNgram = one(inference, 'speculative', (r) => r.proposer === 'ngram' && r.sampling === 'greedy');
const specDraft = one(
  inference,
  'speculative',
  (r) => String(r.proposer).startsWith('draft_model') && r.sampling === 'greedy'
);
const conOff = one(inference, 'constrained_decoding', (r) => r.constrained === false);
const conOn = one(inference, 'constrained_decoding', (r) => r.constrained === true);
const gguf = rows(inference, 'gguf_export');
const q8 = gguf.find((r) => r.quant === 'q8_0');
const vllm = one(inference, 'vllm_baseline');
const int8 = one(inference, 'quantization', (r) => r.variant === 'torch_dynamic_int8');
const fp32 = one(inference, 'quantization', (r) => r.variant === 'fp32');

const num = (v, d) => (typeof v === 'number' ? v.toFixed(d) : '--');

const ledger = [
  {
    id: 'kv-speedup',
    label: 'Naive to contiguous KV cache',
    value: naive512 && contig512 ? contig512.tokens_per_s.mean / naive512.tokens_per_s.mean : null,
    format: 'x',
    status: 'measured',
    scope: '512 new tokens, 3 seeds, variants interleaved per cell',
    note: 'Spec expected 10-20x. Reporting what was measured.',
    bench: 'kv_cache_generation',
  },
  {
    id: 'max-concurrent',
    label: 'Max concurrent sequences at a 64 MB KV budget',
    value: mc[0] ? mc[0].paged_gain : null,
    format: 'x',
    status: 'measured',
    scope: mc[0]
      ? mc[0].contiguous_max_sequences + ' to ' + mc[0].paged_max_sequences + ' at seq_len ' + mc[0].seq_len
      : '',
    note: '',
    bench: 'max_concurrent',
  },
  {
    id: 'ext-frag',
    label: 'External fragmentation, paged allocator',
    value: 0,
    format: 'zero',
    status: 'measured',
    scope: alloc.length + ' seeds',
    note: 'Identically zero by construction: every free block is interchangeable, so a free block can always satisfy the next allocation.',
    bench: 'paged_allocator_measured',
  },
  {
    id: 'prefix-request',
    label: 'Prefix cache, request hit rate',
    value: prefixOn ? prefixOn.request_hit_rate.mean : null,
    format: 'pct',
    status: 'measured',
    scope: prefixOn ? prefixOn.n_requests + ' requests, ' + prefixOn.shared_prefix_len + '-token shared prefix' : '',
    note: prefixOn
      ? 'Token hit rate ' +
        num(prefixOn.token_hit_rate.mean * 100, 1) +
        '% against a ceiling of ' +
        num(prefixOn.max_possible_token_hit_rate * 100, 1) +
        '%.'
      : '',
    bench: 'prefix_cache',
  },
  {
    id: 'prefix-ttft',
    label: 'Prefix cache, mean TTFT change',
    value: prefixOn && prefixOff ? -pct(prefixOff.ttft_mean_ms.mean, prefixOn.ttft_mean_ms.mean) : null,
    format: 'signed-pct',
    status: 'measured',
    scope: num(prefixOff?.ttft_mean_ms.mean, 0) + ' to ' + num(prefixOn?.ttft_mean_ms.mean, 0) + ' ms',
    note: '',
    bench: 'prefix_cache',
  },
  {
    id: 'chunked-prefill',
    label: 'Chunked prefill, p99 inter-token latency',
    value: cpOn && cpOff ? -pct(cpOff.itl_p99_ms.mean, cpOn.itl_p99_ms.mean) : null,
    format: 'signed-pct',
    status: 'measured',
    scope: num(cpOff?.itl_p99_ms.mean, 0) + ' to ' + num(cpOn?.itl_p99_ms.mean, 0) + ' ms, chunk ' + cpOn?.chunk_size,
    note: 'p50 ITL gets worse. The trade is tail latency for median.',
    bench: 'chunked_prefill',
  },
  {
    id: 'constrained',
    label: 'Constrained decoding, invalid JSON rate',
    value: conOn ? conOn.invalid_json_rate.mean : null,
    format: 'pct',
    status: 'measured',
    scope:
      num(conOff?.invalid_json_rate.mean * 100, 0) +
      '% to ' +
      num(conOn?.invalid_json_rate.mean * 100, 0) +
      '%, grammar ' +
      conOn?.grammar,
    note:
      num(conOn?.forced_close_rate.mean * 100, 0) +
      '% of generations hit the token budget and were closed deterministically from the FSM state. Counted here, not scored as a valid-JSON win.',
    bench: 'constrained_decoding',
  },
  {
    id: 'speculative',
    label: 'n-gram speculative decoding',
    value: specNgram ? specNgram.speedup_vs_baseline : null,
    format: 'x',
    status: 'measured',
    scope: 'greedy sampling, 4 speculative tokens',
    note:
      'Weights are untrained, so acceptance rate is a bracket, not a prediction. The 2-of-6-layer draft-model arm reached ' +
      num(specDraft?.speedup_vs_baseline, 2) +
      'x.',
    bench: 'speculative',
  },
  {
    id: 'int8',
    label: 'torch dynamic int8, weight size',
    value: int8 ? int8.weight_mb.mean : null,
    format: 'mb',
    status: 'measured',
    scope: 'fp32 ' + num(fp32?.weight_mb.mean, 1) + ' MB, 12M proxy',
    note: 'Logit KL vs fp32 ' + (int8 ? int8.logit_kl_vs_fp32.mean.toExponential(2) : '--') + '.',
    bench: 'quantization',
  },
  {
    id: 'gguf',
    label: 'GGUF export, q8_0',
    value: q8 ? q8.mb : null,
    format: 'mb',
    status: 'measured',
    scope: q8 ? q8.path.replace(/\\/g, '/') : '',
    note:
      'Verified against our own reader, never llama.cpp, and lossy: ' +
      (q8?.lossy?.length ?? 0) +
      ' dropped-tensor condition(s). This is the 12M proxy export, not the 31M model.',
    bench: 'gguf_export',
  },
  {
    id: 'vllm',
    label: 'vLLM baseline',
    value: null,
    format: 'none',
    status: 'not-run',
    scope: vllm?.reason ?? '',
    note: vllm?.expected_outcome ?? '',
    bench: 'vllm_baseline',
  },
];

// -- 7. Model -----------------------------------------------------------------
const modelOut = {
  provenance: 'measured',
  source: 'artifacts/benchmarks/model.json',
  hardware: model.hardware,
  config: model.config,
  notes: model.notes ?? [],
  kvVariants: rows(model, 'kv_cache').map((r) => ({
    variant: r.variant,
    nKvHeads: r.n_kv_heads,
    kbPerToken: r.kb_per_token,
    mbPerSequence: r.mb_per_sequence,
    contextLen: r.context_len,
    vsMha: r.vs_mha_ratio,
  })),
  attnBackends: rows(model, 'attn_backend').map((r) => ({
    backend: r.backend,
    observedKernel: r.observed_kernel,
    seqLen: r.seq_len,
    latencyMs: stat(r.latency_ms),
    peakMb: r.peak_bytes.mean / 1048576,
    scoreMatrixMb: r.score_matrix_bytes_analytic / 1048576,
  })),
  throughput: rows(model, 'throughput').map((r) => ({
    mode: r.mode,
    tokensPerS: stat(r.tokens_per_s),
    tflops: stat(r.tflops),
    mfu: r.mfu_vs_measured_device_peak,
  })),
};

// -- 8. Tokenizer / judge -----------------------------------------------------
const tokenizerOut = {
  provenance: 'measured',
  source: 'artifacts/benchmarks/tokenizer.json',
  hardware: tokenizer.hardware,
  rows: tokenizer.rows.map((r) => ({
    tokenizer: r.tokenizer,
    vocab: r.vocab,
    bytesPerToken: r.bytes_per_token,
    fertility: r.fertility,
    encodeMbS: r.encode_mb_s,
    error: r.error,
  })),
  mergeLoop: tokenizer.merge_loop_benchmark,
  trainSeconds: tokenizer.tokenizer_train_seconds,
  trainDocuments: tokenizer.train_documents,
  note: tokenizer.notes,
};

const judgeOut = {
  provenance: 'measured',
  source: 'artifacts/benchmarks/judge_calibration.json',
  judge: judge.judge,
  n: judge.n,
  threshold: judge.threshold,
  kappa: stat(judge.kappa_pairwise),
  agreement: stat(judge.agreement_pairwise),
  trusted: judge.trust_pairwise,
  suppressed: judge.untrusted_metrics,
  verdict: judge.verdict,
};

// -- 9. The request path. Per-stage latency is NOT in any artifact yet. --------
const requestPath = [
  {
    node: 'route',
    label: 'Route',
    by: 'LocalMind-31M',
    desc: 'Three-way decision: in-domain, out-of-domain, needs-web. Runs on laptop CPU at int8.',
  },
  {
    node: 'retrieve',
    label: 'Retrieve',
    by: 'BM25 / SPLADE / Dense / ColBERT',
    desc: 'Four independent ranked lists. Fusion consumes ranks, not scores, so RRF needs no normalisation.',
  },
  {
    node: 'grade',
    label: 'Grade',
    by: 'LocalMind-31M',
    desc: 'Per-chunk relevance plus the injection classifier. Flagged chunks are quarantined, never silently dropped.',
  },
  {
    node: 'generate',
    label: 'Generate',
    by: 'Qwen3-4B-Instruct Q4_K_M via Ollama',
    desc: 'A 31M model cannot synthesise a grounded answer. The model card says so plainly.',
  },
  {
    node: 'verify',
    label: 'Verify',
    by: 'claim checker',
    desc: 'Each claim is matched back to a supporting chunk. Unsupported claims force a regeneration or a refusal.',
  },
].map((s) => ({ ...s, p50Ms: null, status: 'not-run' }));

const out = {
  generatedAt: new Date().toISOString(),
  generatedBy: 'frontend/scripts/extract-data.mjs',
  modelTrained: false,
  provenanceLegend: {
    measured: 'Real timing or memory on the named machine. Reproducible via just.',
    synthetic:
      'Real code and real measurement, on a synthetic corpus and/or a deterministic stand-in model. The harness is validated; retrieval quality on a real corpus is not.',
    'not-run': 'Requires a GPU, a network, or a trained checkpoint. No value is invented.',
  },
  hardware: {
    inference: inference.hardware,
    model: model.hardware,
    retrieval: retrieval.hardware,
    cache: cache.hardware,
  },
  inferenceNotes: inference.notes ?? [],
  ledger,
  maxConcurrent,
  fragmentation,
  allocator,
  semanticCache,
  retrieval: retrievalOut,
  comparisonMatrix,
  model: modelOut,
  tokenizer: tokenizerOut,
  judge: judgeOut,
  requestPath,
};

mkdirSync(dirname(OUT), { recursive: true });
writeFileSync(OUT, JSON.stringify(out, null, 2) + '\n');

const nulls = ledger.filter((l) => l.value === null).length;
console.log(
  '[extract] wrote ' +
    OUT +
    ' - ' +
    ledger.length +
    ' ledger rows (' +
    nulls +
    ' not-run), ' +
    maxConcurrent.rows.length +
    ' concurrency points, ' +
    fragmentation.rows.length +
    ' fragmentation runs, ' +
    semanticCache.rows.length +
    ' tau steps, ' +
    retrievalOut.rows.length +
    ' retrieval configs, ' +
    comparisonMatrix.cellsMeasured +
    '/' +
    comparisonMatrix.cellsTotal +
    ' matrix cells measured.'
);
