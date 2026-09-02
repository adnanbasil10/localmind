/**
 * Client for the RAG gateway (localmind/api, built concurrently).
 *
 * Contract, from localmind/agent/state.py :: AgentResult, which is what the
 * gateway hands back:
 *
 *   POST {base}/chat   { query, session_id? }
 *     -> { answer, status, refusal_reason, citations[], sources[], steps,
 *          elapsed_s, tokens_used, timings[]? }
 *
 *   GET  {base}/health -> 200
 *
 * The gateway does not exist yet in this checkout. Everything below is written
 * to fail loudly and specifically rather than hang: a fixed timeout, a typed
 * failure, and shape-tolerant parsing so a field the gateway names slightly
 * differently degrades to "absent" instead of throwing.
 */

export const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? 'http://localhost:8000';
export const TIMINGS_KEY = 'localmind:last-timings';

export interface Citation {
  marker: number;
  chunk_id: string;
  doc_id: string;
  uri: string;
}

export interface Source {
  chunk_id: string;
  doc_id: string;
  text: string;
  score: number;
  source: string;
  uri: string;
}

export interface Timing {
  node: string;
  ms: number;
}

export interface ChatResult {
  answer: string;
  status: string;
  refusalReason: string;
  citations: Citation[];
  sources: Source[];
  steps: number | null;
  elapsedS: number | null;
  tokensUsed: number | null;
  timings: Timing[];
}

export class GatewayError extends Error {
  constructor(
    message: string,
    readonly kind: 'offline' | 'timeout' | 'http' | 'shape',
    readonly detail?: string
  ) {
    super(message);
    this.name = 'GatewayError';
  }
}

const str = (v: unknown, fallback = ''): string => (typeof v === 'string' ? v : fallback);
const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);
const arr = (v: unknown): unknown[] => (Array.isArray(v) ? v : []);

function parse(raw: unknown): ChatResult {
  if (!raw || typeof raw !== 'object') {
    throw new GatewayError('The gateway returned a body that is not an object.', 'shape');
  }
  const o = raw as Record<string, unknown>;
  const answer = str(o.answer) || str(o.text) || str(o.content) || str(o.response);
  const status = str(o.status, 'answered');

  const citations: Citation[] = arr(o.citations).map((c, i) => {
    const x = (c ?? {}) as Record<string, unknown>;
    return {
      marker: num(x.marker) ?? i + 1,
      chunk_id: str(x.chunk_id),
      doc_id: str(x.doc_id),
      uri: str(x.uri),
    };
  });

  const rawSources = arr(o.sources).length ? arr(o.sources) : arr(o.documents ?? o.chunks);
  const sources: Source[] = rawSources.map((s) => {
    const x = (s ?? {}) as Record<string, unknown>;
    return {
      chunk_id: str(x.chunk_id) || str(x.id),
      doc_id: str(x.doc_id),
      text: str(x.text) || str(x.content) || str(x.snippet),
      score: num(x.score) ?? 0,
      source: str(x.source, 'documents'),
      uri: str(x.uri) || str(x.url),
    };
  });

  const timings: Timing[] = arr(o.timings)
    .map((t) => {
      const x = (t ?? {}) as Record<string, unknown>;
      return { node: str(x.node), ms: num(x.ms) ?? 0 };
    })
    .filter((t) => t.node !== '');

  if (!answer && status !== 'refused' && status !== 'error') {
    throw new GatewayError(
      'The gateway replied without an answer field.',
      'shape',
      Object.keys(o).join(', ')
    );
  }

  return {
    answer,
    status,
    refusalReason: str(o.refusal_reason) || str(o.refusalReason),
    citations,
    sources,
    steps: num(o.steps),
    elapsedS: num(o.elapsed_s),
    tokensUsed: num(o.tokens_used),
    timings,
  };
}

async function withTimeout<T>(ms: number, fn: (signal: AbortSignal) => Promise<T>): Promise<T> {
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), ms);
  try {
    return await fn(ctl.signal);
  } finally {
    clearTimeout(timer);
  }
}

export async function ping(): Promise<boolean> {
  try {
    return await withTimeout(2500, async (signal) => {
      const res = await fetch(`${API_BASE}/health`, { signal, cache: 'no-store' });
      return res.ok;
    });
  } catch {
    return false;
  }
}

export async function chat(query: string, sessionId: string): Promise<ChatResult> {
  let res: Response;
  try {
    res = await withTimeout(45000, (signal) =>
      fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, session_id: sessionId }),
        signal,
        cache: 'no-store',
      })
    );
  } catch (e) {
    const aborted = e instanceof DOMException && e.name === 'AbortError';
    throw new GatewayError(
      aborted
        ? `No response from ${API_BASE}/chat within 45 s.`
        : `Cannot reach the RAG gateway at ${API_BASE}.`,
      aborted ? 'timeout' : 'offline'
    );
  }

  if (!res.ok) {
    let detail = '';
    try {
      detail = (await res.text()).slice(0, 400);
    } catch {
      /* body already consumed or unreadable */
    }
    throw new GatewayError(`Gateway returned HTTP ${res.status}.`, 'http', detail);
  }

  let body: unknown;
  try {
    body = await res.json();
  } catch {
    throw new GatewayError('Gateway returned a body that is not JSON.', 'shape');
  }
  return parse(body);
}
