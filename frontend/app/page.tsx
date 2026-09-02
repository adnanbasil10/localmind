import Link from 'next/link';
import { Panel, Stamp } from '@/components/Panel';
import { Matrix } from '@/components/Matrix';
import { Concurrency } from '@/components/charts/Concurrency';
import { Waste } from '@/components/charts/Waste';
import { TauSweep } from '@/components/charts/TauSweep';
import { Ndcg } from '@/components/charts/Ndcg';
import { data } from '@/lib/data';
import { int, ledgerValue, n, pct, withCi } from '@/lib/fmt';

const NEGATIVES = [
  'Naive → KV speedup is 7.9×, under the 10–20× the plan expected.',
  'Injection defence generalises to 37.5% on held-out paraphrases despite 100% in-sample.',
  'Tuned fusion beat RRF on the synthetic corpus, contradicting ADR 0006’s prior. Recorded in the ADR as provisional, with what would settle it.',
  'Continuous batching wins TTFT but not throughput. Length-bucketed decode is a workaround for the model’s dense past_kv; real ragged batching needs a Phase 2 change.',
  'The GGUF export is verified against our own reader, never llama.cpp, and is lossy: the llama architecture has no QK-norm tensors.',
  'BPE exhausted mergeable pairs at 8,062 of a requested 16,384 vocab on the small corpus.',
  'An earlier inference benchmark was discarded for aliasing machine drift onto variant identity. The discredited run is kept so the correction is auditable.',
];

export default function Results() {
  const m = data.model;
  const r = data.retrieval;
  const c = data.semanticCache;
  const frag = data.fragmentation;
  const contigWaste = frag.rows.map((x) => x.contiguousWaste);
  const pagedWaste = frag.rows.map((x) => x.pagedWaste);
  const top = data.maxConcurrent.rows[0];
  const bestNdcg = r.rows.reduce((a, b) => ((b.ndcg.mean ?? 0) > (a.ndcg.mean ?? 0) ? b : a));

  return (
    <>
      <div className="shell">
        <header className="titleblock">
          <p className="titleblock__eyebrow">Engineering dossier · systems benchmarks · free-tier compute</p>
          <div className="titleblock__main">
            <h1>
              Local
              <br />
              Mind
            </h1>
            <div>
              <p className="titleblock__lede">
                A <strong>31M-parameter</strong> decoder-only LM, built from scratch, sitting inside an
                agentic RAG system. Everything below measures the <strong>systems</strong> — tokenizer,
                attention backends, inference engine, retrieval fusion, caches. None of it measures model
                quality.
              </p>
              <ul className="titleblock__spec">
                <li>
                  control plane <b>LocalMind-31M · laptop CPU · int8</b>
                </li>
                <li>
                  generator <b>Qwen3-4B-Instruct Q4_K_M · Ollama</b>
                </li>
                <li>
                  retrieval <b>BM25 · SPLADE · Dense · ColBERT → RRF k=60</b>
                </li>
                <li>
                  measured on <b>{data.hardware.inference.split('|')[0].trim()}, no GPU</b>
                </li>
              </ul>
            </div>
          </div>
          <div className="readouts">
            <div className="readout">
              <div className="readout__k">Parameters</div>
              <div className="readout__v">{int(m.config.params_excl_norms)}</div>
              <div className="readout__n">excl. norms · {int(m.config.params_incl_norms)} incl.</div>
            </div>
            <div className="readout">
              <div className="readout__k">Cash cost</div>
              <div className="readout__v">$0.00</div>
              <div className="readout__n">0 GPU-hours · docs/compute_log.md</div>
            </div>
            <div className="readout">
              <div className="readout__k">Paged KV headroom</div>
              <div className="readout__v">{top.gain.toFixed(1)}×</div>
              <div className="readout__n">
                {top.contiguous} → {top.paged} sequences @ {data.maxConcurrent.budgetMb} MB
              </div>
            </div>
            <div className="readout readout--alert">
              <div className="readout__k">Headline matrix</div>
              <div className="readout__v">
                {data.comparisonMatrix.cellsMeasured} / {data.comparisonMatrix.cellsTotal}
              </div>
              <div className="readout__n">cells measured · {data.comparisonMatrix.dodStatus}</div>
            </div>
          </div>
        </header>

        <div className="notice">
          <div className="notice__hatch" aria-hidden="true" />
          <div className="notice__body">
            <h2>The model is not trained</h2>
            <div>
              <p>
              No checkpoint exists. Every number on this page is a measurement of code running on{' '}
              <strong>CPU with randomly initialised weights</strong>, or on a synthetic corpus with a
              deterministic stand-in model. Systems metrics — latency, memory, throughput, hit rates — are
              unaffected by weight quality and are real. Quality-dependent metrics are bracketed and flagged,
                never presented as predictions.
              </p>
              <p style={{ marginBottom: 0 }}>
                Three labels appear on every panel. <Stamp p="measured" /> real timing or memory on the
                named machine. <Stamp p="synthetic" /> real code and real measurement, synthetic corpus or
                stand-in model. <Stamp p="not-run" /> needs a GPU, a network, or a checkpoint — and no
                value is invented to fill the gap.
              </p>
            </div>
          </div>
        </div>

        {/* ---------------------------------------------------------- KV -- */}
        <section className="section" aria-labelledby="kv">
          <div className="section__head">
            <h2 id="kv">KV memory</h2>
            <p>
              A contiguous KV cache reserves <code>max_seq_len</code> per sequence at admission. A paged
              cache reserves one 16-token block at a time. Both were run on the same machine, same seeds,
              same workload mixes. This is the strongest result in the repository.
            </p>
          </div>

          <Panel
            title="Max concurrent sequences at a fixed KV budget"
            provenance={data.maxConcurrent.provenance}
            source="inference.json :: max_concurrent"
            caption={
              <>
                Budget {data.maxConcurrent.budgetMb} MB, <code>max_seq_len</code>{' '}
                {data.maxConcurrent.maxSeqLen}, {data.maxConcurrent.kvBytesPerToken} bytes/token. Contiguous
                admission is flat at {top.contiguous} regardless of how short the sequences actually are —
                that flatness <em>is</em> the bug.
              </>
            }
            foot={<>Hardware: {data.maxConcurrent.hardware}</>}
          >
            <Concurrency />
          </Panel>

          <Panel
            title="Reserved-KV waste, per run"
            provenance={frag.provenance}
            source="inference.json :: fragmentation"
            caption={
              <>
                Internal fragmentation — the fraction of reserved KV bytes holding no real token — across{' '}
                {frag.rows.length} runs ({frag.nSequences} sequences, block size {frag.blockSize}). Contiguous
                wastes {pct(Math.min(...contigWaste))}–{pct(Math.max(...contigWaste))}. Paged wastes{' '}
                {pct(Math.min(...pagedWaste))}–{pct(Math.max(...pagedWaste))}.
              </>
            }
            foot={
              <>
                External fragmentation on the paged allocator is{' '}
                <strong style={{ color: 'var(--ink)' }}>identically zero</strong> across{' '}
                {data.allocator.externalFragmentation.length} seeds — by construction, not by luck: every
                free block is interchangeable, so a free block can always satisfy the next allocation.
              </>
            }
          >
            <Waste />
          </Panel>
        </section>

        {/* --------------------------------------------------- retrieval -- */}
        <section className="section" aria-labelledby="ret">
          <div className="section__head">
            <h2 id="ret">Retrieval</h2>
            <p>
              {r.corpus.n_documents} documents, {r.corpus.n_queries} queries, a{' '}
              <strong>{r.corpus.type}</strong> corpus and deterministic stand-in models. What this validates
              is the harness — fusion, reranking, index engineering. It does not validate retrieval quality
              on a real corpus, and it is labelled accordingly.
            </p>
          </div>

          <Panel
            title="nDCG@10 by configuration"
            provenance={r.provenance}
            source="retrieval.json :: rows"
            caption={
              <>
                Bootstrap 95% CI shown as whiskers on every bar. Best measured configuration:{' '}
                <strong style={{ color: 'var(--ink)' }}>{bestNdcg.config}</strong> at {withCi(bestNdcg.ndcg)}.
                BM25 is a from-scratch implementation, hand-verified.
              </>
            }
            foot={<>Index engine: {r.indexEngine}</>}
          >
            <Ndcg />
          </Panel>

          <div className="grid-2">
            <Panel
              title="Index engineering"
              provenance={r.provenance}
              source="retrieval.json :: index_engineering"
            >
              <div className="tablewrap">
                <table className="data">
                  <caption className="hint" style={{ textAlign: 'left', paddingBottom: '0.5rem' }}>
                    Binary quantisation and filtered ANN
                  </caption>
                  <tbody>
                    <tr>
                      <td className="lead">Recall@10, full precision</td>
                      <td className="num">{n(r.binaryQuantization.recall_at_10_full_precision, 3)}</td>
                    </tr>
                    <tr>
                      <td className="lead">Recall@10, binary + rescore</td>
                      <td className="num">{n(r.binaryQuantization.recall_at_10_binary_rescored, 3)}</td>
                    </tr>
                    <tr>
                      <td className="lead">Retained at 32× compression</td>
                      <td className="num">{pct(r.binaryQuantization.retained_fraction)}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
              <div className="tablewrap" style={{ marginTop: '1rem' }}>
                <table className="data">
                  <thead>
                    <tr>
                      <th scope="col">filter selectivity</th>
                      <th scope="col" className="num">
                        post-filter recall
                      </th>
                      <th scope="col" className="num">
                        pre-filter recall
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {r.filteredAnn.map((f) => (
                      <tr key={f.selectivity}>
                        <td className="lead">{pct(f.selectivity)}</td>
                        <td
                          className="num"
                          style={{ color: f.post_filter_recall < 0.9 ? 'var(--red-ink)' : undefined }}
                        >
                          {n(f.post_filter_recall, 3)}
                        </td>
                        <td className="num">{n(f.pre_filter_recall, 3)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="hint" style={{ marginTop: '0.9rem' }}>
                Post-filter recall collapses to {n(r.filteredAnn[r.filteredAnn.length - 1].post_filter_recall, 2)} under a{' '}
                {pct(r.filteredAnn[r.filteredAnn.length - 1].selectivity)}-selective filter while pre-filter
                holds at 1.0. Reproduced as a runnable test rather than asserted.
              </p>
            </Panel>

            <Panel title="Fusion: RRF vs tuned weights" provenance={r.provenance} source="retrieval.json :: fusion_comparison">
              <div className="tablewrap">
                <table className="data">
                  <thead>
                    <tr>
                      <th scope="col">split</th>
                      <th scope="col" className="num">
                        RRF
                      </th>
                      <th scope="col" className="num">
                        tuned
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td className="lead">dev ({r.fusion.dev_queries} q)</td>
                      <td className="num">{n(r.fusion.dev_rrf, 3)}</td>
                      <td className="num">{n(r.fusion.dev_tuned, 3)}</td>
                    </tr>
                    <tr>
                      <td className="lead">test ({r.fusion.test_queries} q, held out)</td>
                      <td className="num">{n(r.fusion.test_rrf, 3)}</td>
                      <td className="num">{n(r.fusion.test_tuned, 3)}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
              <p className="hint" style={{ marginTop: '0.9rem' }}>
                {r.fusion.note}
              </p>
              <p className="hint">
                Cross-encoder adds {n(r.reranking.ndcg_at_10_gain, 4)} nDCG@10 for{' '}
                {n(r.reranking.added_p95_ms, 2)} ms p95. Contextual chunking adds{' '}
                {n(r.contextual.delta, 4)} recall@20.
              </p>
            </Panel>
          </div>
        </section>

        {/* ------------------------------------------------------- cache -- */}
        <section className="section" aria-labelledby="cache">
          <div className="section__head">
            <h2 id="cache">Semantic cache</h2>
            <p>
              Sweeping the similarity threshold trades hit rate against false hits — up to the point where
              it stops trading anything at all. The query set is deliberately adversarial, so absolute hit
              rates read low; the shape of the tradeoff is the result, not a hit-rate number to advertise.
            </p>
          </div>
          <Panel
            title="τ sweep — hit rate against false-hit rate"
            provenance={c.provenance}
            source="semantic_cache.json"
            caption={
              <>
                Operating point <strong style={{ color: 'var(--ink)' }}>τ = {c.operatingPoint.tau}</strong>,
                selected by {c.operatingPoint.selectionRule}. It reports{' '}
                <strong style={{ color: 'var(--red-ink)' }}>
                  meets_threshold = {String(c.operatingPoint.meetsThreshold)}
                </strong>{' '}
                against the {pct(c.operatingPoint.threshold, 0)} bar, rather than moving the bar.
              </>
            }
            foot={
              <>
                <p style={{ maxWidth: '90ch', margin: '0 0 0.6em' }}>{c.description}</p>
                <p style={{ maxWidth: '90ch', margin: 0 }}>{c.reasoning}</p>
              </>
            }
          >
            <TauSweep />
          </Panel>
        </section>

        {/* ------------------------------------------------------ ledger -- */}
        <section className="section" aria-labelledby="engine">
          <div className="section__head">
            <h2 id="engine">Inference engine</h2>
            <p>
              Each row is one benchmark, each figure traced to its bench name. CPU only — no GPU was
              available, and the vLLM comparison says so instead of estimating.
            </p>
          </div>
          <Panel
            title="Engine ledger"
            provenance="measured"
            source="inference.json"
            foot={<>Hardware: {data.hardware.inference}</>}
          >
            <div className="ledger">
              {data.ledger.map((row) => (
                <div className="ledger__row" key={row.id} data-status={row.status}>
                  <div>
                    <div className="ledger__label">{row.label}</div>
                    <div className="ledger__meta">
                      {row.bench}
                      {row.scope ? ` · ${row.scope}` : ''}
                    </div>
                  </div>
                  <div className="ledger__value">{ledgerValue(row)}</div>
                  {row.note ? <p className="ledger__note">{row.note}</p> : null}
                </div>
              ))}
            </div>
          </Panel>
        </section>

        {/* ----------------------------------------------------- not run -- */}
        <section className="section" aria-labelledby="notrun">
          <div className="section__head">
            <h2 id="notrun">Not run</h2>
            <p>
              {data.comparisonMatrix.dodReason}. The harnesses exist and are tested; they run the moment a
              checkpoint does.
            </p>
          </div>
          <Panel
            title="§9 5e comparison matrix"
            provenance="not-run"
            source="phase5_comparison_matrix.json"
            caption="Six arms × four metrics. Nothing here is a zero, a dash or an empty box — an unrun experiment must never be mistakable for a negative result."
            foot={<>Environment: {data.comparisonMatrix.hardware}</>}
          >
            <Matrix />
          </Panel>

          <div className="grid-2">
            <Panel
              title="Judge calibration"
              provenance={data.judge.provenance}
              source="judge_calibration.json"
              caption={
                <>
                  The lexical fallback judge scores κ = {withCi(data.judge.kappa)} on n = {data.judge.n},
                  below the {data.judge.threshold} trust bar.
                </>
              }
            >
              <div className="tablewrap">
                <table className="data">
                  <tbody>
                    <tr>
                      <td className="lead">Judge</td>
                      <td className="num">{data.judge.judge}</td>
                    </tr>
                    <tr>
                      <td className="lead">Pairwise agreement</td>
                      <td className="num">{withCi(data.judge.agreement)}</td>
                    </tr>
                    <tr>
                      <td className="lead">Trusted</td>
                      <td className="num" style={{ color: 'var(--red-ink)' }}>
                        {String(data.judge.trusted)}
                      </td>
                    </tr>
                    <tr>
                      <td className="lead">Auto-suppressed metrics</td>
                      <td className="num void-cell">{data.judge.suppressed.join(', ')}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
              <p className="hint" style={{ marginTop: '0.9rem' }}>
                The harness refuses the judge and suppresses judged metrics rather than reporting them with a
                caveat.
              </p>
            </Panel>

            <Panel title="Tokenizer" provenance={data.tokenizer.provenance} source="tokenizer.json">
              <div className="tablewrap">
                <table className="data">
                  <thead>
                    <tr>
                      <th scope="col">tokenizer</th>
                      <th scope="col" className="num">
                        bytes/token
                      </th>
                      <th scope="col" className="num">
                        encode MB/s
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.tokenizer.rows.map((t) => (
                      <tr key={t.tokenizer}>
                        <td className="lead">{t.tokenizer}</td>
                        <td className="num">
                          {t.error ? <span className="void-cell">not run</span> : n(t.bytesPerToken, 3)}
                        </td>
                        <td className="num">
                          {t.error ? <span className="void-cell">not run</span> : n(t.encodeMbS, 2)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="hint" style={{ marginTop: '0.9rem' }}>
                Merge loop, naive → incremental: {n(data.tokenizer.mergeLoop.speedup_x, 2)}× (
                {n(data.tokenizer.mergeLoop.naive_seconds, 2)}s → {n(data.tokenizer.mergeLoop.incremental_seconds, 2)}s
                at vocab {int(data.tokenizer.mergeLoop.vocab_size)}). {data.tokenizer.note}
              </p>
            </Panel>
          </div>
        </section>

        {/* --------------------------------------------------- negatives -- */}
        <section className="section" aria-labelledby="neg">
          <div className="section__head">
            <h2 id="neg">Kept negatives</h2>
            <p>Results that went the wrong way, kept deliberately. A losing result is a deliverable.</p>
          </div>
          <div className="panel">
            <div className="panel__body">
              <ul className="notes notes--neg">
                {NEGATIVES.map((x) => (
                  <li key={x}>{x}</li>
                ))}
              </ul>
            </div>
          </div>
          <div className="panel">
            <div className="panel__head">
              <h3 className="panel__title">Benchmark caveats, verbatim</h3>
              <span className="panel__src">inference.json :: notes</span>
            </div>
            <div className="panel__body">
              <ul className="notes">
                {data.inferenceNotes.map((x) => (
                  <li key={x}>{x}</li>
                ))}
              </ul>
            </div>
          </div>
          <p style={{ marginTop: '1.5rem' }}>
            <Link className="link" href="/architecture">
              Request path and where each stage runs →
            </Link>
          </p>
        </section>
      </div>
    </>
  );
}
