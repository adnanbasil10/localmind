'use client';

import { data } from '@/lib/data';
import { ticks } from '@/lib/fmt';
import { Legend, Tip, TipRow, barPath, useTip } from './parts';

const COLOR: Record<string, string> = {
  naive: 'var(--d2)',
  sdpa_math: 'var(--d3)',
  sdpa_efficient: 'var(--d1)',
};

const ORDER = ['naive', 'sdpa_math', 'sdpa_efficient'];

/**
 * Attention backend latency by sequence length, all three backends on CPU.
 * The three agree to 1e-3 in fp32; what differs is cost. Note the kernel names:
 * sdpa_efficient pins ATen's tiled CPU kernel, NOT FlashAttention-2, which needs
 * SM 8.0+ and is banned repo-wide.
 */
export function AttnBackends() {
  const rows = data.model.attnBackends;
  const { wrap, tip, show, hide } = useTip();
  const seqLens = [...new Set(rows.map((r) => r.seqLen))].sort((a, b) => a - b);

  const W = 780;
  const padL = 72;
  const padR = 74;
  const padT = 22;
  const barH = 15;
  const gap = 2;
  const groupH = ORDER.length * (barH + gap) + 22;
  const H = padT + seqLens.length * groupH + 26;
  const plotW = W - padL - padR;
  const max = Math.max(...rows.map((r) => r.latencyMs.mean ?? 0));
  const xs = (v: number) => (v / max) * plotW;
  const tk = ticks(max, 4);

  return (
    <div className="chartwrap" ref={wrap}>
      <Legend items={ORDER.map((b) => ({ label: b, color: COLOR[b] }))} />
      <svg
        className="chart"
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={`Attention backend latency in milliseconds by sequence length. ${rows
          .map((r) => `${r.backend} at ${r.seqLen}: ${(r.latencyMs.mean ?? 0).toFixed(0)} ms.`)
          .join(' ')}`}
      >
        {tk.map((t) => (
          <g key={t}>
            <line className="grid-line" x1={padL + xs(t)} x2={padL + xs(t)} y1={padT - 8} y2={H - 24} />
            <text x={padL + xs(t)} y={H - 10} textAnchor="middle">
              {t}
            </text>
          </g>
        ))}
        <text x={padL + plotW / 2} y={padT - 12} textAnchor="middle">
          latency (ms) · fp32 · CPU · batch 1
        </text>

        {seqLens.map((sl, gi) => {
          const gTop = padT + gi * groupH;
          return (
            <g key={sl}>
              <text className="t-cat" x={padL - 12} y={gTop + groupH / 2 - 4} textAnchor="end">
                {sl}
              </text>
              {ORDER.map((b, bi) => {
                const row = rows.find((r) => r.seqLen === sl && r.backend === b);
                if (!row) return null;
                const y = gTop + bi * (barH + gap);
                const v = row.latencyMs.mean ?? 0;
                const node = (
                  <>
                    <span className="tip__h">
                      {b} · seq {sl}
                    </span>
                    <TipRow label="latency" value={`${v.toFixed(1)} ms`} color={COLOR[b]} />
                    <TipRow
                      label="95% CI"
                      value={`${(row.latencyMs.lo ?? 0).toFixed(0)}–${(row.latencyMs.hi ?? 0).toFixed(0)} ms`}
                    />
                    <TipRow label="peak mem" value={`${row.peakMb.toFixed(1)} MB`} />
                    <TipRow label="kernel" value={row.observedKernel} />
                  </>
                );
                return (
                  <g key={b}>
                    <path d={barPath(padL, y, xs(v), barH)} fill={COLOR[b]} />
                    <text className="t-value" x={padL + xs(v) + 8} y={y + barH - 3}>
                      {v.toFixed(0)}
                    </text>
                    <rect
                      className="hit"
                      x={0}
                      y={y - 1}
                      width={W}
                      height={barH + 2}
                      onPointerMove={(e) => show(e, node)}
                      onPointerLeave={hide}
                    />
                  </g>
                );
              })}
            </g>
          );
        })}
        <line className="axis-line" x1={padL} x2={padL} y1={padT - 8} y2={H - 24} />
      </svg>
      <Tip tip={tip} />
    </div>
  );
}
