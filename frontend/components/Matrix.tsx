import { data } from '@/lib/data';

const HEAD: Record<string, string> = {
  router_acc: 'router acc',
  grader_f1: 'grader F1',
  rewrite_win_rate: 'rewrite win',
  p50_latency_ms: 'p50 latency',
};

/**
 * The §9 5e comparison matrix. Zero of twenty-four cells are measured.
 *
 * Every cell renders as a hatched void carrying the word NOT RUN. None of them
 * renders as 0, as a dash that could be read as a low score, or as an empty box
 * that could be read as an omission. An unrun experiment must never be
 * mistakable for a negative result - which is why the definition of done here
 * evaluates to "not-evaluable" rather than "failed".
 */
export function Matrix() {
  const m = data.comparisonMatrix;
  return (
    <div className="matrix">
      <table className="matrix__t">
        <caption className="hint" style={{ textAlign: 'left', paddingBottom: '0.6rem' }}>
          {m.cellsMeasured} of {m.cellsTotal} cells measured · DoD status{' '}
          <strong style={{ color: 'var(--red-ink)' }}>{m.dodStatus}</strong>
        </caption>
        <thead>
          <tr>
            <th scope="col">arm</th>
            {m.columns.map((c) => (
              <th key={c} scope="col">
                {HEAD[c] ?? c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {m.rows.map((r) => (
            <tr key={r.name}>
              <th scope="row">
                {r.name}
                <span>
                  {r.hardware} · params updated {r.paramsUpdated}
                </span>
              </th>
              {r.cells.map((c) => (
                <td key={c.column}>
                  <div className="cell-void" title={`${c.column}: ${c.status}`}>
                    not run
                  </div>
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
