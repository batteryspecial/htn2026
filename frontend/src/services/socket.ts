/**
 * A WebSocket that reconnects with backoff and parses JSON for you.
 *
 * Every socket in this app is a read-only firehose the server owns
 * (`/ws/state`, `/ws/events`, `/ws/status`, later `/ws/trace`), so there is no
 * send path here on purpose. Add one when something needs to talk back.
 */

export interface SocketHandlers<T> {
  onMessage?: (data: T) => void;
  onOpen?: () => void;
  onClose?: () => void;
}

export class ReconnectingSocket<T = unknown> {
  private ws: WebSocket | null = null;
  private backoff = 1000;
  private retry: ReturnType<typeof setTimeout> | null = null;
  private closed = false;

  constructor(
    private readonly url: string,
    private readonly handlers: SocketHandlers<T> = {},
  ) {}

  connect(): void {
    if (this.closed) return;

    let ws: WebSocket;
    try {
      ws = new WebSocket(this.url);
    } catch {
      this.scheduleReconnect();
      return;
    }
    this.ws = ws;

    ws.onopen = () => {
      this.backoff = 1000;
      this.handlers.onOpen?.();
    };

    ws.onmessage = (evt) => {
      let data: T;
      try {
        data = JSON.parse(evt.data as string) as T;
      } catch {
        return;
      }
      this.handlers.onMessage?.(data);
    };

    ws.onclose = () => {
      this.handlers.onClose?.();
      this.scheduleReconnect();
    };

    ws.onerror = () => {
      try { ws.close(); } catch { /* onclose handles the retry */ }
    };
  }

  private scheduleReconnect(): void {
    if (this.closed) return;
    this.retry = setTimeout(() => this.connect(), this.backoff);
    this.backoff = Math.min(this.backoff * 1.6, 5000);
  }

  close(): void {
    this.closed = true;
    if (this.retry) clearTimeout(this.retry);
    if (this.ws) {
      try { this.ws.close(); } catch { /* already gone */ }
    }
  }
}
