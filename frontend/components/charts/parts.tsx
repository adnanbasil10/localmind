'use client';

import { useCallback, useRef, useState, type ReactNode } from 'react';

/* -------------------------------------------------------------- tooltip -- */

export interface TipState {
  x: number;
  y: number;
  w: number;
  node: ReactNode;
}

export function useTip() {
  const wrap = useRef<HTMLDivElement | null>(null);
  const [tip, setTip] = useState<TipState | null>(null);

  const show = useCallback((e: { clientX: number; clientY: number }, node: ReactNode) => {
    const r = wrap.current?.getBoundingClientRect();
    if (!r) return;
    setTip({ x: e.clientX - r.left, y: e.clientY - r.top, w: r.width, node });
  }, []);

  const hide = useCallback(() => setTip(null), []);
  return { wrap, tip, show, hide };
}

export function Tip({ tip }: { tip: TipState | null }) {
  if (!tip) return null;
  const flip = tip.x > tip.w * 0.55;
  return (
    <div
      className="tip"
      role="status"
      style={{
        left: flip ? undefined : tip.x + 14,
        right: flip ? tip.w - tip.x + 14 : undefined,
        top: Math.max(0, tip.y - 12),
      }}
    >
      {tip.node}
    </div>
  );
}

export function TipRow({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <span className="tip__row">
      <span>
        {color ? <i className="tip__sw" style={{ background: color }} aria-hidden="true" /> : null}
        {label}
      </span>
      <b>{value}</b>
    </span>
  );
}

/* --------------------------------------------------------------- legend -- */

export function Legend({ items }: { items: { label: string; color: string; shape?: 'bar' | 'line' }[] }) {
  return (
    <div className="legend">
      {items.map((i) => (
        <span className="legend__item" key={i.label}>
          <span
            className="legend__swatch"
            aria-hidden="true"
            style={
              i.shape === 'line'
                ? { background: 'transparent', borderTop: `2px solid ${i.color}`, height: 2 }
                : { background: i.color }
            }
          />
          {i.label}
        </span>
      ))}
    </div>
  );
}

/* ---------------------------------------------------------------- marks -- */

/** Horizontal bar anchored at x0 with a 4px rounded data-end. */
export function barPath(x0: number, y: number, w: number, h: number): string {
  const r = Math.min(4, Math.max(0, w), h / 2);
  if (w <= 0.5) return `M${x0} ${y} h0.5 v${h} h-0.5 Z`;
  return `M${x0} ${y} h${w - r} a${r} ${r} 0 0 1 ${r} ${r} v${h - 2 * r} a${r} ${r} 0 0 1 ${-r} ${r} h${-(w - r)} Z`;
}

export function line(points: [number, number][]): string {
  return points.map((p, i) => `${i === 0 ? 'M' : 'L'}${p[0].toFixed(2)} ${p[1].toFixed(2)}`).join(' ');
}

/** Diagonal hatch used wherever a value does not exist. Never a zero, never a gap. */
export function HatchDef({ id }: { id: string }) {
  return (
    <defs>
      <pattern id={id} width="8" height="8" patternTransform="rotate(-45)" patternUnits="userSpaceOnUse">
        <rect width="8" height="8" fill="transparent" />
        <line x1="0" y1="0" x2="0" y2="8" stroke="var(--red)" strokeWidth="3" opacity="0.55" />
      </pattern>
    </defs>
  );
}
