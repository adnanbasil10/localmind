'use client';

import { createContext, useContext, useMemo, useState, type ReactNode } from 'react';
import { usePathname } from 'next/navigation';
import type { Provenance } from '@/lib/data';

/**
 * The evidence filter.
 *
 * Every figure in this repo is labelled measured / synthetic / not-run. This
 * control lets a reader strip the page down to only the class of evidence they
 * are willing to accept. Filtered panels collapse to a struck stub - they are
 * never quietly removed, because "this exists and you chose not to look at it"
 * and "this does not exist" are different statements.
 */

const KINDS: Provenance[] = ['measured', 'synthetic', 'not-run'];

type State = { on: Record<Provenance, boolean>; toggle: (k: Provenance) => void };

const Ctx = createContext<State>({
  on: { measured: true, synthetic: true, 'not-run': true },
  toggle: () => {},
});

export function EvidenceProvider({ children }: { children: ReactNode }) {
  const [on, setOn] = useState<Record<Provenance, boolean>>({
    measured: true,
    synthetic: true,
    'not-run': true,
  });
  const value = useMemo<State>(
    () => ({ on, toggle: (k) => setOn((p) => ({ ...p, [k]: !p[k] })) }),
    [on]
  );
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export const useEvidence = () => useContext(Ctx);

export function EvidenceBar() {
  const { on, toggle } = useEvidence();
  const path = usePathname();
  const hidden = KINDS.filter((k) => !on[k]);

  // The bar controls panels. /query has none, so showing it there would be a
  // control that does nothing.
  if (path === '/query') return null;

  return (
    <div className="evidence">
      <div className="evidence__in">
        <span className="evidence__label" id="evidence-label">
          Evidence filter
        </span>
        <div className="evidence__toggles" role="group" aria-labelledby="evidence-label">
          {KINDS.map((k) => (
            <button
              key={k}
              type="button"
              className="toggle"
              data-kind={k}
              aria-pressed={on[k]}
              onClick={() => toggle(k)}
            >
              <span className="toggle__box" aria-hidden="true" />
              {k.replace('-', ' ')}
            </button>
          ))}
        </div>
        <span className="evidence__count" aria-live="polite">
          {hidden.length === 0
            ? 'showing all evidence classes'
            : `hiding ${hidden.map((h) => h.replace('-', ' ')).join(' + ')}`}
        </span>
      </div>
    </div>
  );
}
