'use client';

import type { ReactNode } from 'react';
import type { Provenance } from '@/lib/data';
import { useEvidence } from './Evidence';

export function Stamp({ p }: { p: Provenance }) {
  const label = p === 'not-run' ? 'not run' : p;
  return (
    <span className="stamp" data-p={p} title={LEGEND[p]}>
      {label}
    </span>
  );
}

const LEGEND: Record<Provenance, string> = {
  measured: 'Real timing or memory on the named machine. Reproducible via just.',
  synthetic:
    'Real code and real measurement, on a synthetic corpus and/or a deterministic stand-in model. The harness is validated; quality on a real corpus is not.',
  'not-run': 'Requires a GPU, a network, or a trained checkpoint. No value is invented.',
};

export function Panel({
  title,
  provenance,
  source,
  caption,
  foot,
  children,
}: {
  title: string;
  provenance: Provenance;
  source?: string;
  caption?: ReactNode;
  foot?: ReactNode;
  children: ReactNode;
}) {
  const { on } = useEvidence();

  if (!on[provenance]) {
    return (
      <section className="panel panel--filtered" aria-label={`${title} (hidden by evidence filter)`}>
        <div className="stub">
          <span>{title}</span>
          <span className="stub__rule" aria-hidden="true" />
          <span>hidden — {provenance.replace('-', ' ')}</span>
        </div>
      </section>
    );
  }

  return (
    <section className="panel">
      <header className="panel__head">
        <h3 className="panel__title">{title}</h3>
        <Stamp p={provenance} />
        {source ? <span className="panel__src">{source}</span> : null}
      </header>
      <div className="panel__body">
        {caption ? <p className="panel__caption">{caption}</p> : null}
        {children}
      </div>
      {foot ? <div className="panel__foot">{foot}</div> : null}
    </section>
  );
}
