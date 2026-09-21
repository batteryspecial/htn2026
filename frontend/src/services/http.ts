/** fetch with a timeout and error messages that name the likely cause. */

export class HttpError extends Error {
  constructor(readonly status: number, message: string, readonly body?: unknown) {
    super(message);
    this.name = 'HttpError';
  }
}

export function friendlyError(e: unknown, what = 'the service'): string {
  if (e instanceof DOMException && e.name === 'AbortError') return 'request timed out';
  if (e instanceof HttpError) return e.message;
  if (e instanceof TypeError) {
    return `could not reach ${what} (server down, wrong URL, or CORS not enabled on it)`;
  }
  return e instanceof Error ? e.message : String(e);
}

export interface RequestOptions extends RequestInit {
  timeoutMs?: number;
}

/** One fetch, one timeout, JSON in and out. Non-2xx throws an HttpError. */
export async function request<T>(url: string, opts: RequestOptions = {}): Promise<T> {
  const { timeoutMs = 8000, ...init } = opts;
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);

  try {
    const res = await fetch(url, { ...init, signal: ctrl.signal });
    const raw = await res.text();

    let payload: unknown = null;
    if (raw) {
      try { payload = JSON.parse(raw); } catch { payload = raw; }
    }

    if (!res.ok) {
      const detail = detailOf(payload) ?? raw.slice(0, 200) ?? res.statusText;
      throw new HttpError(res.status, `${res.status}: ${detail}`, payload);
    }
    return payload as T;
  } finally {
    clearTimeout(timer);
  }
}

export function postJson<T>(url: string, body: unknown, opts: RequestOptions = {}): Promise<T> {
  return request<T>(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...(opts.headers ?? {}) },
    body: JSON.stringify(body),
    ...opts,
  });
}

function detailOf(payload: unknown): string | null {
  if (typeof payload === 'string') return payload.slice(0, 200);
  if (payload && typeof payload === 'object') {
    const p = payload as Record<string, unknown>;
    const d = p.detail ?? p.message ?? p.reason;
    if (typeof d === 'string') return d;
    if (d) return JSON.stringify(d).slice(0, 200);
  }
  return null;
}
