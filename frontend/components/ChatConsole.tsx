'use client';

import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import {
  API_BASE,
  GatewayError,
  TIMINGS_KEY,
  chat,
  ping,
  type ChatResult,
  type Source,
} from '@/lib/api';

type Conn = 'checking' | 'online' | 'offline';

interface Turn {
  id: number;
  role: 'user' | 'agent' | 'error';
  text: string;
  result?: ChatResult;
  detail?: string;
}

/** Split an answer on [n] markers so each one becomes a focusable control. */
function renderWithCitations(
  text: string,
  onPick: (marker: number) => void,
  active: number | null
): ReactNode[] {
  const out: ReactNode[] = [];
  const re = /\[(\d{1,2})\]/g;
  let last = 0;
  let match: RegExpExecArray | null;
  let k = 0;
  while ((match = re.exec(text)) !== null) {
    if (match.index > last) out.push(text.slice(last, match.index));
    const marker = Number(match[1]);
    out.push(
      <button
        key={`c${k++}`}
        type="button"
        className="cite"
        onClick={() => onPick(marker)}
        aria-label={`Jump to source ${marker}`}
        style={active === marker ? { background: 'var(--red)' } : undefined}
      >
        {marker}
      </button>
    );
    last = match.index + match[0].length;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

export function ChatConsole() {
  const [conn, setConn] = useState<Conn>('checking');
  const [turns, setTurns] = useState<Turn[]>([]);
  const [q, setQ] = useState('');
  const [busy, setBusy] = useState(false);
  const [active, setActive] = useState<number | null>(null);
  const sessionId = useMemo(() => `ui-${Math.random().toString(36).slice(2, 10)}`, []);
  const nextId = useRef(1);

  const check = useCallback(async () => {
    setConn('checking');
    setConn((await ping()) ? 'online' : 'offline');
  }, []);

  useEffect(() => {
    void check();
  }, [check]);

  const latest = [...turns].reverse().find((t) => t.role === 'agent');
  const sources: Source[] = latest?.result?.sources ?? [];

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    const query = q.trim();
    if (!query || busy) return;
    setQ('');
    setBusy(true);
    setActive(null);
    setTurns((t) => [...t, { id: nextId.current++, role: 'user', text: query }]);
    try {
      const result = await chat(query, sessionId);
      setConn('online');
      setTurns((t) => [
        ...t,
        { id: nextId.current++, role: 'agent', text: result.answer, result },
      ]);
      if (result.timings.length) {
        try {
          window.localStorage.setItem(TIMINGS_KEY, JSON.stringify(result.timings));
        } catch {
          /* storage disabled - the architecture page just stays unannotated */
        }
      }
    } catch (err) {
      const ge = err instanceof GatewayError ? err : null;
      if (ge?.kind === 'offline' || ge?.kind === 'timeout') setConn('offline');
      setTurns((t) => [
        ...t,
        {
          id: nextId.current++,
          role: 'error',
          text: ge?.message ?? 'The request failed for an unknown reason.',
          detail: ge?.detail,
        },
      ]);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="chat">
      <div>
        <div className="console">
          <div className="console__log">
            {turns.length === 0 ? (
              conn === 'offline' ? (
                <Dead onRetry={check} />
              ) : (
                <Idle />
              )
            ) : null}

            {turns.map((t) => (
              <div className={`turn turn--${t.role}`} key={t.id}>
                <div className="turn__who">
                  {t.role === 'user' ? 'query' : t.role === 'agent' ? 'agent' : 'gateway error'}
                  {t.role === 'agent' && t.result ? (
                    <>
                      <span>· status {t.result.status}</span>
                      {t.result.elapsedS !== null ? <span>· {t.result.elapsedS.toFixed(2)} s</span> : null}
                      {t.result.steps !== null ? <span>· {t.result.steps} steps</span> : null}
                    </>
                  ) : null}
                </div>
                <div className="turn__body">
                  {t.role === 'agent' ? (
                    <>
                      {t.text
                        .split(/\n{2,}/)
                        .filter(Boolean)
                        .map((para, i) => (
                          <p key={i}>{renderWithCitations(para, setActive, active)}</p>
                        ))}
                      {t.result?.status === 'refused' && t.result.refusalReason ? (
                        <p style={{ color: 'var(--red-ink)' }}>Refused: {t.result.refusalReason}</p>
                      ) : null}
                      {t.result && t.result.citations.length === 0 ? (
                        <p className="hint">
                          The gateway returned no citations for this answer. An uncited answer is not a
                          grounded answer.
                        </p>
                      ) : null}
                    </>
                  ) : (
                    <>
                      <p>{t.text}</p>
                      {t.detail ? <p className="hint">{t.detail}</p> : null}
                    </>
                  )}
                </div>
              </div>
            ))}
          </div>

          <form className="console__form" onSubmit={submit}>
            <span className="console__prompt" aria-hidden="true">
              &gt;
            </span>
            <input
              className="console__input"
              value={q}
              onChange={(e) => setQ(e.target.value)}
              placeholder={conn === 'offline' ? 'gateway unreachable — send anyway to retry' : 'ask the corpus'}
              aria-label="Query"
              disabled={busy}
              autoComplete="off"
            />
            <button className="btn" type="submit" disabled={busy || q.trim() === ''}>
              {busy ? 'running' : 'send'}
            </button>
          </form>
        </div>

        <div style={{ display: 'flex', gap: '1rem', alignItems: 'center', flexWrap: 'wrap', marginTop: '0.7rem' }}>
          <span className="conn" data-s={conn}>
            <span className="conn__dot" aria-hidden="true" />
            {conn === 'checking' ? 'probing gateway' : conn === 'online' ? 'gateway online' : 'gateway unreachable'}
          </span>
          <span className="hint">{API_BASE}</span>
          <button type="button" className="btn btn--ghost" onClick={check} disabled={conn === 'checking'}>
            re-probe
          </button>
        </div>
      </div>

      <aside aria-label="Citations">
        <div className="panel">
          <div className="panel__head">
            <h3 className="panel__title">Citations</h3>
            <span className="panel__src">{sources.length} source(s)</span>
          </div>
          <div className="panel__body">
            {sources.length === 0 ? (
              <p className="hint" style={{ margin: 0 }}>
                Sources appear here when an answer comes back. Each carries its chunk id, its retrieval
                score and its origin, so a claim can be traced to the text it came from.
              </p>
            ) : (
              sources.map((s, i) => {
                const marker =
                  latest?.result?.citations.find((c) => c.chunk_id === s.chunk_id)?.marker ?? i + 1;
                return (
                  <div
                    className={`source${active === marker ? ' source--active' : ''}`}
                    key={`${s.chunk_id}-${i}`}
                  >
                    <div className="source__h">
                      <span className="source__m">{marker}</span>
                      <span className="source__id">
                        {s.doc_id || s.chunk_id || 'unidentified chunk'} · {s.source} · score{' '}
                        {s.score.toFixed(3)}
                      </span>
                    </div>
                    <p className="source__t">{s.text || '(the gateway returned no text for this chunk)'}</p>
                    {s.uri ? (
                      <p className="source__id" style={{ marginTop: '0.35rem' }}>
                        {s.uri}
                      </p>
                    ) : null}
                  </div>
                );
              })
            )}
          </div>
        </div>

        <div className="panel">
          <div className="panel__head">
            <h3 className="panel__title">Gateway contract</h3>
            <span className="panel__src">localmind/agent/state.py :: AgentResult</span>
          </div>
          <div className="panel__body">
            <pre className="hint" style={{ margin: 0, whiteSpace: 'pre-wrap' }}>
              {`POST ${API_BASE}/chat
  { query, session_id }
→ { answer, status, refusal_reason,
    citations[{ marker, chunk_id, doc_id, uri }],
    sources[{ chunk_id, doc_id, text, score,
              source, uri }],
    steps, elapsed_s, tokens_used, timings[] }

GET  ${API_BASE}/health → 200`}
            </pre>
            <p className="hint" style={{ marginTop: '0.9rem', marginBottom: 0 }}>
              This UI is written against that contract and parses it defensively: a field the gateway names
              differently degrades to absent rather than throwing, and a request that does not answer inside
              45 s fails with a reason instead of spinning.
            </p>
          </div>
        </div>
      </aside>
    </div>
  );
}

function Idle() {
  return (
    <div className="hint">
      <p style={{ marginBottom: '0.5rem' }}>
        POST /chat → route · retrieve · grade · generate · verify. The answer comes back with its
        citations, and every cited chunk is listed beside it.
      </p>
      <p style={{ margin: 0 }}>
        Retrieved text is untrusted by contract. It is wrapped before it reaches a prompt, and a chunk
        the injection classifier flags is quarantined rather than silently dropped.
      </p>
    </div>
  );
}

function Dead({ onRetry }: { onRetry: () => void }) {
  return (
    <div className="deadstate">
      <h3>Gateway unreachable</h3>
      <p>
        Nothing is listening on <code>{API_BASE}</code>. The RAG gateway (<code>localmind/api</code>) is
        built separately from this UI and is not running.
      </p>
      <ol>
        <li>
          <code>just up core</code> — postgres+pgvector, redis, api
        </li>
        <li>
          Confirm <code>{API_BASE}/health</code> returns 200
        </li>
        <li>
          Point elsewhere with <code>NEXT_PUBLIC_API_BASE</code> if the gateway moved
        </li>
      </ol>
      <p style={{ marginTop: '0.75rem', marginBottom: 0 }}>
        <button type="button" className="btn btn--ghost" onClick={onRetry}>
          probe again
        </button>
      </p>
    </div>
  );
}
