'use client';

import { data } from '@/lib/data';
import { Tip, TipRow, barPath, useTip } from './parts';

/**
 * nDCG@10 by retrieval configuration, with bootstrap 95% CI whiskers.
 * One series, so no legend box: the panel title names the measure.
 * Configs are kept in build order (single arms, then fusions, then reranking)
 * because that order is the argument; sorting by score would destroy it.
 */
export function Ndcg() {
  const r = data.retrieval;
  const { wrap, tip, show, hide } = useTip();

  const W = 780;
  const padL = 186;
  const padR = 78;
  const padT = 12;
  const rowH = 34;
  const barH = 16;
  const H = padT + r.rows.length * rowH + 32;
  const plotW = W - padL - padR;
  const xMax = 1;
  const xs = (v: number) => (v / xMax) * plotW;
  const tk = [0, 0.25, 0.5, 0.75, 1];

  return (
    <div className="chartwrap" ref={wrap}>
      <svg
        className="chart"
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={`nDCG at 10 by retrieval configuration. ${r.rows
          .map((x) => `${x.config}: ${(x.ndcg.mean ?? 0).toFixed(3)}.`)
          .join(' ')}`}
      >
        {tk.map((t) => (
          <g key={t}>
            <line className="grid-line" x1={padL + xs(t)} x2={padL + xs(t)} y1={padT} y2={H - 26} />
            <text x={padL + xs(t)} y={H - 10} textAnchor="middle">
              {t.toFixed(2)}
            </text>
          </g>
        ))}
        <text x={padL + plotW / 2} y={H - 26 + 34} opacity="0">
          .
        </text>

        {r.rows.map((row, i) => {
          const y = padT + i * rowH + (rowH - barH) / 2;
          const m = row.ndcg.mean ?? 0;
          const lo = row.ndcg.lo ?? m;
          const hi = row.ndcg.hi ?? m;
          const best = m === Math.max(...r.rows.map((x) => x.ndcg.mean ?? 0));
          const node = (
            <>
              <span className="tip__h">{row.config}</span>
              <TipRow label="nDCG@10" value={m.toFixed(3)} color="var(--d1)" />
              <TipRow label="95% CI" value={`${lo.toFixed(3)} – ${hi.toFixed(3)}`} />
              <TipRow label="recall@5" value={(row.recall5.mean ?? 0).toFixed(3)} />
              <TipRow label="recall@20" value={(row.recall20.mean ?? 0).toFixed(3)} />
              <TipRow label="p50 / p95" value={`${(row.p50.mean ?? 0).toFixed(2)} / ${(row.p95.mean ?? 0).toFixed(2)} ms`} />
            </>
          );
          return (
            <g key={row.config}>
              <text className="t-cat" x={padL - 12} y={y + barH - 3} textAnchor="end">
                {row.config}
              </text>
              <path d={barPath(padL, y, xs(m), barH)} fill="var(--d1)" opacity={best ? 1 : 0.72} />
              {/* bootstrap 95% CI - a bare number is never reported. A 2px surface
                  casing keeps the whisker legible where it crosses the fill. */}
              <g>
                <path
                  d={`M${padL + xs(lo)} ${y + 1} v${barH - 2} M${padL + xs(lo)} ${y + barH / 2} H${
                    padL + xs(hi)
                  } M${padL + xs(hi)} ${y + 1} v${barH - 2}`}
                  stroke="var(--panel)"
                  strokeWidth="4"
                  fill="none"
                />
                <path
                  d={`M${padL + xs(lo)} ${y + 2} v${barH - 4} M${padL + xs(lo)} ${y + barH / 2} H${
                    padL + xs(hi)
                  } M${padL + xs(hi)} ${y + 2} v${barH - 4}`}
                  stroke="var(--ink)"
                  strokeWidth="1.5"
                  fill="none"
                />
              </g>
              <text className="t-value" x={padL + xs(hi) + 8} y={y + barH - 3}>
                {m.toFixed(3)}
              </text>
              <rect
                className="hit"
                x={0}
                y={padT + i * rowH}
                width={W}
                height={rowH}
                onPointerMove={(e) => show(e, node)}
                onPointerLeave={hide}
              />
            </g>
          );
        })}
        <line className="axis-line" x1={padL} x2={padL} y1={padT} y2={H - 26} />
      </svg>
      <Tip tip={tip} />
    </div>
  );
}
