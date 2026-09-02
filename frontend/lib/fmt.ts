import type { LedgerRow, Stat } from './data';

/** Fixed series order. Never cycled: a fifth series is a design error, not a new hue. */
export const SERIES = {
  d1: 'var(--d1)', // #bf8419 amber
  d2: 'var(--d2)', // #5e90d6 steel
  d3: 'var(--d3)', // #31a98c teal
} as const;

export const n = (v: number | null | undefined, d = 2): string =>
  v === null || v === undefined || Number.isNaN(v) ? '--' : v.toFixed(d);

export const pct = (v: number | null | undefined, d = 1): string =>
  v === null || v === undefined ? '--' : `${(v * 100).toFixed(d)}%`;

export const int = (v: number | null | undefined): string =>
  v === null || v === undefined ? '--' : v.toLocaleString('en-US');

/** "0.713 [0.657, 0.768]" - a bare number is never reported (CONVENTIONS.md rule 5). */
export const withCi = (s: Stat | null | undefined, d = 3): string => {
  if (!s || s.mean === null) return '--';
  if (s.lo === null || s.hi === null) return s.mean.toFixed(d);
  return `${s.mean.toFixed(d)} [${s.lo.toFixed(d)}, ${s.hi.toFixed(d)}]`;
};

export const ledgerValue = (r: LedgerRow): string => {
  if (r.value === null) return 'not run';
  switch (r.format) {
    case 'x':
      return `${r.value.toFixed(2)}×`;
    case 'pct':
      return `${(r.value * 100).toFixed(1)}%`;
    case 'signed-pct':
      return `${r.value > 0 ? '+' : '−'}${Math.abs(r.value).toFixed(1)}%`;
    case 'mb':
      return `${r.value.toFixed(2)} MB`;
    case 'zero':
      return '0';
    default:
      return 'not run';
  }
};

/** Nice round ticks covering [0, max]. */
export function ticks(max: number, count = 4): number[] {
  if (max <= 0) return [0];
  const rawStep = max / count;
  const mag = Math.pow(10, Math.floor(Math.log10(rawStep)));
  const norm = rawStep / mag;
  const step = (norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 5 ? 5 : 10) * mag;
  const out: number[] = [];
  for (let t = 0; t <= max + step * 1e-9; t += step) out.push(+t.toFixed(10));
  return out;
}
