'use client';

import { useEffect, useState } from 'react';
import { data } from '@/lib/data';
import { TIMINGS_KEY, type Timing } from '@/lib/api';

/**
 * The request path, annotated with real per-stage latency.
 *
 * No artifact carries per-stage timings yet, so every stage reads "awaiting
 * run" until the gateway returns some. When a query on /query comes back with
 * a timings array, it is cached and shown here, labelled as a single live
 * sample rather than a benchmark - one request is not a p50.
 */
export function RequestPath() {
  const [timings, setTimings] = useState<Timing[] | null>(null);

  useEffect(() => {
    try {
      const raw = window.localStorage.getItem(TIMINGS_KEY);
      if (raw) setTimings(JSON.parse(raw) as Timing[]);
    } catch {
      /* storage disabled - stages simply stay unannotated */
    }
  }, []);

  const msFor = (node: string): number | null => {
    if (!timings) return null;
    const hits = timings.filter((t) => t.node === node);
    if (!hits.length) return null;
    return hits.reduce((a, b) => a + b.ms, 0);
  };

  return (
    <>
      <div className="path">
        {data.requestPath.map((s, i) => {
          const ms = msFor(s.node);
          return (
            <div className="stage" key={s.node}>
              <div className="stage__idx" aria-hidden="true">
                {String(i + 1).padStart(2, '0')}
              </div>
              <div>
                <div className="stage__name">{s.label}</div>
                <div className="stage__by">{s.by}</div>
                <p className="stage__desc">{s.desc}</p>
              </div>
              <div className="stage__t" data-has={ms !== null}>
                {ms !== null ? `${ms.toFixed(1)} ms — last request` : 'latency: awaiting run'}
              </div>
            </div>
          );
        })}
      </div>
      <p className="hint" style={{ marginTop: '1rem' }}>
        {timings
          ? 'Timings above are one live sample from the most recent query, not a benchmark. Three seeds and a bootstrap CI are required before any of it becomes a reported figure.'
          : 'No per-stage latency exists in any artifact yet, so none is shown. Run a query against a live gateway on the Query page and the stages annotate themselves from the response.'}
      </p>
    </>
  );
}
