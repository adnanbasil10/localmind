'use client';

import { data } from '@/lib/data';
import { Legend, Tip, TipRow, useTip } from './parts';

/**
 * Reserved-KV waste, contiguous vs paged, for every measured run.
 * A paired dot plot: one row per (workload mix, seed), the connector carries the
 * magnitude of the collapse. Nine runs are shown individually rather than
 * averaged, because the seed spread is part of the result.
 */
export function Waste() {
  const f = data.fragmentation;
  const { wrap, tip, show, hide } = useTip();

  const W = 780;
  const padL = 152;
  const padR = 58;
  const padT = 20;
  const rowH = 26;
  const H = padT + f.rows.length * rowH + 34;
  const plotW = W - padL - padR;
  const xs = (v: number) => padL + v * plotW;
  const tk = [0, 0.25, 0.5, 0.75, 1];

  return (
    <div className="chartwrap" ref={wrap}>
      <Legend
        items={[
          { label: 'contiguous', color: 'var(--d2)' },
          { label: 'paged (block 16)', color: 'var(--d1)' },
        ]}
      />
      <svg
        className="chart"
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={`Reserved KV waste per run. Contiguous ranges from ${(
          Math.min(...f.rows.map((r) => r.contiguousWaste)) * 100
        ).toFixed(1)} to ${(Math.max(...f.rows.map((r) => r.contiguousWaste)) * 100).toFixed(
          1
        )} percent; paged from ${(Math.min(...f.rows.map((r) => r.pagedWaste)) * 100).toFixed(1)} to ${(
          Math.max(...f.rows.map((r) => r.pagedWaste)) * 100
        ).toFixed(1)} percent.`}
      >
        {tk.map((t) => (
          <g key={t}>
            <line className="grid-line" x1={xs(t)} x2={xs(t)} y1={padT - 6} y2={H - 26} />
            <text x={xs(t)} y={H - 10} textAnchor="middle">
              {(t * 100).toFixed(0)}%
            </text>
          </g>
        ))}

        {f.rows.map((r, i) => {
          const y = padT + i * rowH + rowH / 2;
          const node = (
            <>
              <span className="tip__h">
                {r.mix} · seed {r.seed}
              </span>
              <TipRow label="contiguous" value={`${(r.contiguousWaste * 100).toFixed(1)}%`} color="var(--d2)" />
              <TipRow label="paged" value={`${(r.pagedWaste * 100).toFixed(1)}%`} color="var(--d1)" />
              <TipRow label="reserved" value={`${r.contiguousReservedMb.toFixed(1)} → ${r.pagedReservedMb.toFixed(1)} MB`} />
              <TipRow label="mean seq_len" value={r.meanSeqLen.toFixed(1)} />
            </>
          );
          return (
            <g key={`${r.mix}-${r.seed}`}>
              <text className="t-cat" x={padL - 12} y={y + 3.5} textAnchor="end">
                {r.mix} · s{r.seed}
              </text>
              <line
                x1={xs(r.pagedWaste)}
                x2={xs(r.contiguousWaste)}
                y1={y}
                y2={y}
                stroke="var(--rule-2)"
                strokeWidth="2"
              />
              {/* 2px surface ring so overlapping marks stay separable */}
              <circle cx={xs(r.contiguousWaste)} cy={y} r="5.5" fill="var(--panel)" />
              <circle cx={xs(r.contiguousWaste)} cy={y} r="4.5" fill="var(--d2)" />
              <circle cx={xs(r.pagedWaste)} cy={y} r="5.5" fill="var(--panel)" />
              <circle cx={xs(r.pagedWaste)} cy={y} r="4.5" fill="var(--d1)" />
              <text className="t-value" x={xs(r.contiguousWaste) + 10} y={y + 3.5}>
                {(r.contiguousWaste * 100).toFixed(1)}
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
        <line className="axis-line" x1={padL} x2={padL} y1={padT - 6} y2={H - 26} />
      </svg>
      <Tip tip={tip} />
    </div>
  );
}
