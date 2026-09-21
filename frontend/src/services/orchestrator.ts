/** HTTP turns and trace streaming for the agent layer on :8000. */

import { REQUEST_TIMEOUT_MS, toWebSocketUrl } from '../config/settings';
import type { InstalledBehavior } from '../contracts/liveRun';
import { friendlyError, postJson, request } from './http';
import type { Reachability } from './perception';

/** What POST /chat answers with. */
export interface ChatResponse {
  turn: string;
  reply: string;
  seconds: number;
  ok: boolean;
  outcome?: 'applied' | 'reply' | 'error';
  behaviors?: InstalledBehavior[];
}

/** Every kind the orchestrator publishes on /ws/trace. */
export type TraceKind =
  | 'user' | 'thought' | 'tool_call' | 'tool_result'
  | 'say' | 'reply' | 'event' | 'error' | 'timer';

/** One line of the agent trace. */
export interface TraceEntry {
  id: string;
  ts: number;
  kind: TraceKind;
  label: string;
  detail?: string;
  /** Which turn it belongs to, so the panel can group. */
  turn?: string;
  data?: Record<string, unknown>;
}

export class OrchestratorClient {
  constructor(private base: string, private timeoutMs = REQUEST_TIMEOUT_MS) {}

  configure(base: string, timeoutMs: number): void {
    this.base = base.replace(/\/+$/, '');
    this.timeoutMs = timeoutMs;
  }

  url(path: string): string {
    return this.base + path;
  }

  /**
   * POST /chat — one operator turn, with optional photos.
   *
   * Uploads are registered with perception as references before the agent
   * thinks, so "track that human" with a photo works in one message.
   * Multipart, so no base64 bloat on the wire.
   */
  async chat(text: string, images: File[] = [], turn?: string): Promise<ChatResponse> {
    const form = new FormData();
    form.append('text', text);
    if (turn) form.append('turn', turn);
    for (const image of images) form.append('images', image);

    try {
      return await request<ChatResponse>(this.url('/chat'), {
        method: 'POST',
        body: form,
        timeoutMs: this.timeoutMs,
      });
    } catch (e) {
      throw new Error(friendlyError(e, 'the orchestrator'));
    }
  }

  async health(): Promise<Reachability> {
    try {
      const health = await request<{
        has_key: boolean;
        perception: { up: boolean };
        model_connection: { state: 'unknown' | 'ok' | 'error'; detail: string };
      }>(
        this.url('/health'), { timeoutMs: 3000 },
      );
      const modelOk = health.model_connection?.state === 'ok';
      return {
        up: health.has_key && health.perception.up && modelOk,
        detail: !health.has_key
          ? 'API key missing'
          : !health.perception.up
            ? 'Perception offline'
            : health.model_connection?.detail ?? 'Model connection not verified',
      };
    } catch (e) {
      return { up: false, detail: friendlyError(e, 'the orchestrator') };
    }
  }

  stop(turn?: string): Promise<{ stopped: boolean }> {
    return postJson(this.url('/stop'), { turn }, { timeoutMs: 30000 });
  }

  /** The agent's tool calls, its say() output, and the switch timer. */
  traceSocketUrl(): string {
    return toWebSocketUrl(this.base, '/ws/trace');
  }
}
