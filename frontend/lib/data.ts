import raw from '../data/benchmarks.json';

export type Provenance = 'measured' | 'synthetic' | 'not-run';

export interface Stat {
  mean: number | null;
  lo: number | null;
  hi: number | null;
  n: number | null;
}

export interface LedgerRow {
  id: string;
  label: string;
  value: number | null;
  format: 'x' | 'pct' | 'signed-pct' | 'mb' | 'zero' | 'none';
  status: Provenance;
  scope: string;
  note: string;
  bench: string;
}

export const data = raw as unknown as {
  generatedAt: string;
  generatedBy: string;
  modelTrained: boolean;
  provenanceLegend: Record<Provenance, string>;
  hardware: Record<string, string>;
  inferenceNotes: string[];
  ledger: LedgerRow[];
  maxConcurrent: {
    provenance: Provenance;
    source: string;
    hardware: string;
    budgetMb: number;
    maxSeqLen: number;
    kvBytesPerToken: number;
    rows: { seqLen: number; contiguous: number; paged: number; gain: number }[];
  };
  fragmentation: {
    provenance: Provenance;
    source: string;
    nSequences: number;
    blockSize: number;
    budgetMb: number;
    rows: {
      mix: string;
      seed: number | null;
      meanSeqLen: number;
      contiguousWaste: number;
      pagedWaste: number;
      contiguousExternal: number;
      pagedExternal: number;
      contiguousReservedMb: number;
      pagedReservedMb: number;
      amplification: number;
    }[];
  };
  allocator: {
    provenance: Provenance;
    source: string;
    externalFragmentation: number[];
    note: string;
    internalRange: [number, number] | null;
  };
  semanticCache: {
    provenance: Provenance;
    source: string;
    hardware: string;
    embedder: string;
    description: string;
    reasoning: string;
    floor: number;
    operatingPoint: {
      tau: number;
      hitRate: number;
      falseHitOfTotal: number;
      falseHitOfHits: number;
      meetsThreshold: boolean;
      threshold: number;
      selectionRule: string;
    };
    rows: { tau: number; hit: Stat; falseTotal: Stat; falseOfHits: Stat }[];
  };
  retrieval: {
    provenance: Provenance;
    source: string;
    hardware: string;
    corpus: { type: string; n_documents: number; n_queries: number; notes: string };
    notes: string[];
    indexEngine: string;
    rows: {
      config: string;
      ndcg: Stat;
      recall5: Stat;
      recall20: Stat;
      p50: Stat;
      p95: Stat;
    }[];
    fusion: Record<string, unknown> & {
      dev_rrf: number;
      dev_tuned: number;
      test_rrf: number;
      test_tuned: number;
      test_tuned_beat_rrf: boolean;
      metric: string;
      note: string;
      dev_queries: number;
      test_queries: number;
    };
    reranking: { cross_encoder: string; ndcg_at_10_gain: number; added_p95_ms: number };
    contextual: {
      recall_at_20_without_context: number;
      recall_at_20_with_context: number;
      delta: number;
    };
    binaryQuantization: {
      recall_at_10_full_precision: number;
      recall_at_10_binary_rescored: number;
      retained_fraction: number;
      hamming_top_k: number;
      p50_ms: number;
      p95_ms: number;
    };
    filteredAnn: {
      selectivity: number;
      post_filter_recall: number;
      pre_filter_recall: number;
      post_filter_latency_ms: number;
      pre_filter_latency_ms: number;
    }[];
  };
  comparisonMatrix: {
    provenance: Provenance;
    source: string;
    cellsMeasured: number;
    cellsTotal: number;
    columns: string[];
    dodStatus: string;
    dodReason: string;
    hardware: string;
    rows: {
      name: string;
      hardware: string;
      paramsUpdated: string;
      cells: { column: string; status: string; mean: number | null }[];
    }[];
  };
  model: {
    provenance: Provenance;
    source: string;
    hardware: string;
    config: {
      name: string;
      params_excl_norms: number;
      params_incl_norms: number;
      [k: string]: unknown;
    };
    notes: string[];
    kvVariants: {
      variant: string;
      nKvHeads: number;
      kbPerToken: number;
      mbPerSequence: number;
      contextLen: number;
      vsMha: number;
    }[];
    attnBackends: {
      backend: string;
      observedKernel: string;
      seqLen: number;
      latencyMs: Stat;
      peakMb: number;
      scoreMatrixMb: number;
    }[];
    throughput: { mode: string; tokensPerS: Stat; tflops: Stat; mfu: number }[];
  };
  tokenizer: {
    provenance: Provenance;
    source: string;
    hardware: string;
    rows: {
      tokenizer: string;
      vocab: number | null;
      bytesPerToken: number | null;
      fertility: number | null;
      encodeMbS: number | null;
      error: string | null;
    }[];
    mergeLoop: { vocab_size: number; naive_seconds: number; incremental_seconds: number; speedup_x: number };
    trainSeconds: number;
    trainDocuments: number;
    note: string;
  };
  judge: {
    provenance: Provenance;
    source: string;
    judge: string;
    n: number;
    threshold: number;
    kappa: Stat;
    agreement: Stat;
    trusted: boolean;
    suppressed: string[];
    verdict: string;
  };
  requestPath: {
    node: string;
    label: string;
    by: string;
    desc: string;
    p50Ms: number | null;
    status: Provenance;
  }[];
};
