import type { Metadata } from 'next';
import Link from 'next/link';
import { Panel } from '@/components/Panel';
import { RequestPath } from '@/components/RequestPath';
import { AttnBackends } from '@/components/charts/AttnBackends';
import { data } from '@/lib/data';
import { int, n, withCi } from '@/lib/fmt';

export const metadata: Metadata = {
  title: 'Request path — LocalMind',
  description: 'Where each stage runs, and why the split is the argument.',
};

const PLACEMENT = [
  ['Tokenizer training, data prep', 'Laptop CPU', 'Embarrassingly parallel, no GPU benefit'],
  ['Pretraining, distillation', 'Kaggle 2× T4', 'The only free GPU allowance; 30 h/week'],
  ['ColQwen2 page indexing', 'Kaggle T4', '~2 GPU-h batch job; vectors then queried on the laptop'],
  ['Router / rewriter / grader', 'Laptop CPU', '31 MB at int8. This is the thesis.'],
  ['Answer generation', 'Laptop (Ollama)', '4B Q4_K_M; needs the capacity a 31M model lacks'],
  ['Retrieval, agent, eval', 'Laptop + Docker', 'CPU-bound; latency costs stay honestly visible'],
];

const CONSTRAINTS = [
  [
    'T4 is SM 7.5',
    'No bf16, no FlashAttention-2. Forces fp16 + GradScaler, and promotes QK-norm and z-loss from nice-to-have to load-bearing.',
  ],
  [
    '12-hour hard session cap',
    'Makes bit-exact resumability architectural rather than a feature. Drives WSD over cosine and the memory-mapped resumable loader.',
  ],
  [
    '16 GB laptop',
    'Drives compose profiles, GQA over MHA, binary quantisation with rescoring in the vector index, and the decision not to run Airflow.',
  ],
];

export default function Architecture() {
  const kv = data.model.kvVariants;
  const mha = kv.find((k) => k.variant === 'MHA');
  const gqa = kv.find((k) => k.variant.startsWith('GQA'));

  return (
    <div className="shell">
      <section className="section" aria-labelledby="path">
        <div className="section__head">
          <h2 id="path">Request path</h2>
          <p>
            Five stages. The 31M model owns three of them and never writes the answer; a 4B model writes
            the answer and never routes. The split is the argument the project is making.
          </p>
        </div>
        <Panel
          title="route → retrieve → grade → generate → verify"
          provenance="not-run"
          source="localmind/agent/state.py :: Node"
          caption="Stage latency is annotated from a live gateway response when one is available. No artifact carries per-stage p50s, so none is shown by default."
          foot={
            <>
              The agent never imports torch. It depends on a three-method Protocol, which is what makes it
              testable offline and makes “swap the 31M model for the 4B control” a one-line ablation.
            </>
          }
        >
          <RequestPath />
        </Panel>

        <div className="rule-hatch" style={{ margin: '2rem 0 1.5rem' }} aria-hidden="true" />

        <Panel
          title="Where each component runs"
          provenance="measured"
          source="docs/architecture.md"
          caption="Three external limits did more to determine this architecture than any preference."
        >
          <div className="tablewrap">
            <table className="data">
              <thead>
                <tr>
                  <th scope="col">component</th>
                  <th scope="col">runs on</th>
                  <th scope="col">why there</th>
                </tr>
              </thead>
              <tbody>
                {PLACEMENT.map(([a, b, c]) => (
                  <tr key={a}>
                    <td className="lead">{a}</td>
                    <td>{b}</td>
                    <td style={{ color: 'var(--ink-2)' }}>{c}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <ul className="notes" style={{ marginTop: '1.25rem' }}>
            {CONSTRAINTS.map(([h, b]) => (
              <li key={h}>
                <strong style={{ color: 'var(--ink)' }}>{h}.</strong> {b}
              </li>
            ))}
          </ul>
        </Panel>

        <div className="grid-2">
          <Panel title="KV cache per attention variant" provenance={data.model.provenance} source="model.json :: kv_cache">
            <div className="tablewrap">
              <table className="data">
                <thead>
                  <tr>
                    <th scope="col">variant</th>
                    <th scope="col" className="num">
                      kv heads
                    </th>
                    <th scope="col" className="num">
                      KB/token
                    </th>
                    <th scope="col" className="num">
                      MB/seq
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {kv.map((k) => (
                    <tr key={k.variant}>
                      <td className="lead">{k.variant}</td>
                      <td className="num">{k.nKvHeads}</td>
                      <td className="num">{k.kbPerToken}</td>
                      <td className="num">{k.mbPerSequence}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="hint" style={{ marginTop: '0.9rem' }}>
              GQA 4:1 cuts the cache from {mha?.kbPerToken} KB/token to {gqa?.kbPerToken} KB/token at{' '}
              {mha?.contextLen} context — {gqa?.vsMha}× less memory per sequence, which is what makes the
              paged allocator&rsquo;s headroom usable on a 16 GB laptop.
            </p>
          </Panel>

          <Panel title="Model facts" provenance={data.model.provenance} source="model.json">
            <div className="tablewrap">
              <table className="data">
                <tbody>
                  <tr>
                    <td className="lead">Parameters, excl. norms</td>
                    <td className="num">{int(data.model.config.params_excl_norms)}</td>
                  </tr>
                  <tr>
                    <td className="lead">Parameters, incl. norms</td>
                    <td className="num">{int(data.model.config.params_incl_norms)}</td>
                  </tr>
                  {data.model.throughput.map((t) => (
                    <tr key={t.mode}>
                      <td className="lead">Throughput, {t.mode}</td>
                      <td className="num">{withCi(t.tokensPerS, 1)} tok/s</td>
                    </tr>
                  ))}
                  <tr>
                    <td className="lead">MFU vs measured device peak</td>
                    <td className="num">
                      {n((data.model.throughput[0]?.mfu ?? 0) * 100, 1)}%
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
            <p className="hint" style={{ marginTop: '0.9rem' }}>
              {data.model.hardware}
            </p>
          </Panel>
        </div>

        <Panel
          title="Attention backends"
          provenance={data.model.provenance}
          source="model.json :: attn_backend"
          caption={
            <>
              All three backends agree to <strong style={{ color: 'var(--ink)' }}>1e-3 in fp32</strong>; what
              differs is cost. Peak memory for the naive kernel grows with the score matrix — {' '}
              {n(data.model.attnBackends.find((a) => a.backend === 'naive' && a.seqLen === 4096)?.peakMb, 0)} MB at
              seq 4096 against{' '}
              {n(
                data.model.attnBackends.find((a) => a.backend === 'sdpa_efficient' && a.seqLen === 4096)?.peakMb,
                0
              )}{' '}
              MB for the tiled kernel.
            </>
          }
          foot={
            <ul className="notes" style={{ margin: 0 }}>
              {data.model.notes.map((x) => (
                <li key={x}>{x}</li>
              ))}
            </ul>
          }
        >
          <AttnBackends />
        </Panel>

        <p style={{ marginTop: '1.5rem' }}>
          <Link className="link" href="/">
            ← Back to measured results
          </Link>
        </p>
      </section>
    </div>
  );
}
