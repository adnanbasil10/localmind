'use client';

import { data } from '@/lib/data';
import { ticks } from '@/lib/fmt';
import { Legend, Tip, TipRow, barPath, useTip } from './parts';

/**
 * How many sequences fit in a fixed KV budget, contiguous vs paged.
 * Grouped bars: the ordered category is sequence length, the measure is a count.
 * Linear axis - the 16x ratio is the finding, so it must be seen at true scale.
 */
export function Concurrency() {
  const mc = data.maxConcurrent;
  const { wrap, tip, show, hide } = useTip();

  const W = 780;
  const padL = 78;
  const padR = 62;
  const padT = 26;
  const rowH = 62;
  const barH = 20;
  const gap = 2; // 2px surface gap between adjacent fills
  const H = padT + mc.rows.length * rowH + 26;
  const plotW = W - padL - padR;
  const max = Math.max(...mc.rows.map((r) => r.paged));
  const xs = (v: number) => (v / max) * plotW;
  const tk = ticks(max, 4);

  return (
    <div className="chartwrap" ref={wrap}>
      <Legend
        items={[
          { label: 'contiguous KV', color: 'var(--d2)' },
          { label: 'paged KV', color: 'var(--d1)' },
        ]}
      />
      <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img"
        aria-label={`Maximum concurrent sequences at a ${mc.budgetMb} megabyte KV budget. ${mc.rows
          .map((r) => `At sequence length ${r.seqLen}: contiguous ${r.contiguous}, paged ${r.paged}.`)
          .join(' ')}`}>
        {tk.map((t) => (
          <g key={t}>
            <line className="grid-line" x1={padL + xs(t)} x2={padL + xs(t)} y1={padT - 8} y2={H - 24} />
            <text x={padL + xs(t)} y={H - 10} textAnchor="middle">
              {t}
            </text>
          </g>
        ))}
        <text x={padL} y={H - 10} textAnchor="middle" opacity="0">
          0
        </text>

        {mc.rows.map((r, i) => {
          const yTop = padT + i * rowH;
          const yA = yTop + 4;
          const yB = yA + barH + gap;
          const tipNode = (
            <>
              <span className="tip__h">seq_len {r.seqLen}</span>
              <TipRow label="contiguous" value={`${r.contiguous} seq`} color="var(--d2)" />
              <TipRow label="paged" value={`${r.paged} seq`} color="var(--d1)" />
              <TipRow label="gain" value={`${r.gain.toFixed(2)}×`} />
            </>
          );
          return (
            <g key={r.seqLen}>
              <text className="t-cat" x={padL - 12} y={yA + barH + 2} textAnchor="end">
                seq_len {r.seqLen}
              </text>
              <path d={barPath(padL, yA, xs(r.contiguous), barH)} fill="var(--d2)" />
              <path d={barPath(padL, yB, xs(r.paged), barH)} fill="var(--d1)" />
              <text className="t-value" x={padL + xs(r.contiguous) + 8} y={yA + barH - 5}>
                {r.contiguous}
              </text>
              <text className="t-value" x={padL + xs(r.paged) + 8} y={yB + barH - 5}>
                {r.paged}
              </text>
              <rect
                className="hit"
                x={0}
                y={yTop}
                width={W}
                height={rowH}
                onPointerMove={(e) => show(e, tipNode)}
                onPointerLeave={hide}
              />
            </g>
          );
        })}
        <line className="axis-line" x1={padL} x2={padL} y1={padT - 8} y2={H - 24} />
      </svg>
      <Tip tip={tip} />
    </div>
  );
}
