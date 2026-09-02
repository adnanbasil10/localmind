'use client';

import { useState } from 'react';
import { data } from '@/lib/data';
import { Legend, Tip, TipRow, line, useTip } from './parts';

/**
 * Semantic cache: hit rate and false-hit rate as tau is swept 0.80 -> 0.99.
 * Both series are rates on [0,1], so they share one axis. A second y-scale would
 * be the single most common chart lie and is not used here.
 *
 * The story is the floor: false-hit rate stops improving at tau = 0.91 because
 * the residual error is a literal cache-key collision, not a similarity problem.
 * The floor rule is the only red mark on this chart - red means "a limit you
 * cannot move", not "a series".
 */
export function TauSweep() {
  const c = data.semanticCache;
  const { wrap, tip, show, hide } = useTip();
  const [idx, setIdx] = useState<number | null>(null);

  const W = 780;
  const padL = 46;
  const padR = 122;
  const padT = 18;
  const padB = 40;
  const H = 320;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;

  const taus = c.rows.map((r) => r.tau);
  const t0 = Math.min(...taus);
  const t1 = Math.max(...taus);
  const yMax = 0.6;
  const xs = (t: number) => padL + ((t - t0) / (t1 - t0)) * plotW;
  const ys = (v: number) => padT + plotH - (v / yMax) * plotH;

  const hitPts: [number, number][] = c.rows.map((r) => [xs(r.tau), ys(r.hit.mean ?? 0)]);
  const falsePts: [number, number][] = c.rows.map((r) => [xs(r.tau), ys(r.falseTotal.mean ?? 0)]);
  const opX = xs(c.operatingPoint.tau);
  const yTicks = [0, 0.15, 0.3, 0.45, 0.6];

  return (
    <div className="chartwrap" ref={wrap}>
      <Legend
        items={[
          { label: 'hit rate', color: 'var(--d1)', shape: 'line' },
          { label: 'false-hit rate (of all queries)', color: 'var(--d2)', shape: 'line' },
        ]}
      />
      <svg
        className="chart"
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={`Semantic cache tau sweep. Hit rate falls from ${(
          (c.rows[0].hit.mean ?? 0) * 100
        ).toFixed(1)} percent at tau ${t0} to ${((c.rows[c.rows.length - 1].hit.mean ?? 0) * 100).toFixed(
          1
        )} percent at tau ${t1}. False-hit rate floors at ${(c.floor * 100).toFixed(1)} percent from tau ${
          c.operatingPoint.tau
        } onward.`}
      >
        {yTicks.map((t) => (
          <g key={t}>
            <line className="grid-line" x1={padL} x2={padL + plotW} y1={ys(t)} y2={ys(t)} />
            <text x={padL - 8} y={ys(t) + 3.5} textAnchor="end">
              {(t * 100).toFixed(0)}%
            </text>
          </g>
        ))}
        {c.rows
          .filter((_, i) => i % 3 === 0 && i <= c.rows.length - 2)
          .map((r) => (
            <text key={r.tau} x={xs(r.tau)} y={H - 18} textAnchor="middle">
              {r.tau.toFixed(2)}
            </text>
          ))}
        <text x={padL + plotW / 2} y={H - 3} textAnchor="middle" fill="var(--ink-3)">
          τ (similarity threshold)
        </text>

        {/* irreducible floor - a limit, not a measurement of a series */}
        <line
          x1={padL}
          x2={padL + plotW}
          y1={ys(c.floor)}
          y2={ys(c.floor)}
          stroke="var(--red)"
          strokeWidth="1"
          strokeDasharray="3 4"
        />
        <text x={padL + plotW + 8} y={ys(c.floor) + 3.5} fill="var(--red-ink)">
          floor {(c.floor * 100).toFixed(1)}%
        </text>

        {/* operating point */}
        <line x1={opX} x2={opX} y1={padT} y2={padT + plotH} stroke="var(--rule-2)" strokeWidth="1" />
        <text x={opX + 6} y={padT + 10} className="t-cat">
          τ = {c.operatingPoint.tau}
        </text>

        <path d={line(falsePts)} fill="none" stroke="var(--d2)" strokeWidth="2" />
        <path d={line(hitPts)} fill="none" stroke="var(--d1)" strokeWidth="2" />

        {c.rows.map((r, i) => (
          <g key={r.tau} opacity={idx === null || idx === i ? 1 : 0.55}>
            <circle cx={hitPts[i][0]} cy={hitPts[i][1]} r={idx === i ? 5 : 3.2} fill="var(--panel)" />
            <circle cx={hitPts[i][0]} cy={hitPts[i][1]} r={idx === i ? 4 : 2.4} fill="var(--d1)" />
            <circle cx={falsePts[i][0]} cy={falsePts[i][1]} r={idx === i ? 5 : 3.2} fill="var(--panel)" />
            <circle cx={falsePts[i][0]} cy={falsePts[i][1]} r={idx === i ? 4 : 2.4} fill="var(--d2)" />
          </g>
        ))}

        {idx !== null ? (
          <line
            x1={xs(c.rows[idx].tau)}
            x2={xs(c.rows[idx].tau)}
            y1={padT}
            y2={padT + plotH}
            stroke="var(--ink-3)"
            strokeWidth="1"
          />
        ) : null}

        {c.rows.map((r, i) => {
          const w = plotW / (c.rows.length - 1);
          return (
            <rect
              key={r.tau}
              className="hit"
              x={xs(r.tau) - w / 2}
              y={padT}
              width={w}
              height={plotH}
              onPointerMove={(e) => {
                setIdx(i);
                show(
                  e,
                  <>
                    <span className="tip__h tip__h--raw">τ = {r.tau.toFixed(2)}</span>
                    <TipRow
                      label="hit rate"
                      value={`${((r.hit.mean ?? 0) * 100).toFixed(1)}%`}
                      color="var(--d1)"
                    />
                    <TipRow
                      label="false-hit / all"
                      value={`${((r.falseTotal.mean ?? 0) * 100).toFixed(1)}%`}
                      color="var(--d2)"
                    />
                    <TipRow
                      label="false-hit / hits"
                      value={`${((r.falseOfHits.mean ?? 0) * 100).toFixed(1)}%`}
                    />
                    <TipRow
                      label="95% CI (hit)"
                      value={`${((r.hit.lo ?? 0) * 100).toFixed(1)}–${((r.hit.hi ?? 0) * 100).toFixed(1)}%`}
                    />
                  </>
                );
              }}
              onPointerLeave={() => {
                setIdx(null);
                hide();
              }}
            />
          );
        })}
        <line className="axis-line" x1={padL} x2={padL} y1={padT} y2={padT + plotH} />
        <line className="axis-line" x1={padL} x2={padL + plotW} y1={padT + plotH} y2={padT + plotH} />
      </svg>
      <Tip tip={tip} />
    </div>
  );
}
