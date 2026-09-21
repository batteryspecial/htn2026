/**
 * Client for perception (:8001) — the reflex layer.
 *
 * Every route here is from `perception/README.md`'s endpoint table, and every
 * one is the current contract: `BehaviorSpec` in, `StateView` and `Event` out.
 * The orchestrator will normally sit in front of this; until it does, the
 * console drives the pipeline itself. Same contract either way.
 */

import { toWebSocketUrl } from '../config/settings';
import type {
  BehaviorCreated, BehaviorView, CameraEntry, CountResult, Health, LookResult,
  ModelEntry, Selector, StateView,
} from '../contracts/behavior';
import type { BehaviorSpec } from '../contracts/behavior';
import { friendlyError, postJson, request } from './http';

export interface Reachability {
  up: boolean;
  detail: string;
  health?: Health;
}

export class PerceptionClient {
  constructor(private base: string) {}

  /** Settings can change at any time; the client follows rather than being rebuilt. */
  setBase(base: string): void {
    this.base = base.replace(/\/+$/, '');
  }

  url(path: string): string {
    return this.base + path;
  }

  /* ------------------------------------------------------------ behaviours */

  /** POST /behaviors -> 201 {id}. A 422 means an unknown kind, class or param. */
  addBehavior(spec: BehaviorSpec): Promise<BehaviorCreated> {
    return postJson<BehaviorCreated>(this.url('/behaviors'), spec);
  }

  listBehaviors(): Promise<{ behaviors: BehaviorView[]; kinds: { available: string[]; planned: string[] } }> {
    return request(this.url('/behaviors'));
  }

  removeBehavior(id: string): Promise<{ removed: string }> {
    return request(this.url(`/behaviors/${id}`), { method: 'DELETE' });
  }

  /** DELETE /behaviors — stop everything. This is what the STOP button calls. */
  clearBehaviors(): Promise<{ cleared: boolean }> {
    return request(this.url('/behaviors'), { method: 'DELETE' });
  }

  /** Install a whole program, replacing what was running. */
  async applyProgram(behaviors: BehaviorSpec[]): Promise<string[]> {
    await this.clearBehaviors();
    const ids: string[] = [];
    for (const behavior of behaviors) {
      ids.push((await this.addBehavior(behavior)).id);
    }
    return ids;
  }

  /* ----------------------------------------------------------- inspection */

  async health(): Promise<Reachability> {
    try {
      const health = await request<Health>(this.url('/health'), { timeoutMs: 1500 });
      return { up: true, detail: health.detail ?? health.status, health };
    } catch (e) {
      return { up: false, detail: friendlyError(e, 'the pipeline') };
    }
  }

  /** GET /models. Returns [] rather than throwing: the dropdown degrades. */
  async models(): Promise<ModelEntry[]> {
    try {
      const m = await request<{ models?: ModelEntry[] }>(this.url('/models'), { timeoutMs: 3000 });
      return m?.models ?? [];
    } catch {
      return [];
    }
  }

  setModel(name: string, role?: string): Promise<unknown> {
    return postJson(this.url('/model'), { name, role: role ?? null });
  }

  /**
   * GET /cameras. Degrades to [] like `models()` does.
   *
   * The list comes from the pipeline, not from `navigator.mediaDevices`: the
   * camera is opened by OpenCV on whichever machine runs perception, and the
   * browser's own device list names cameras that process cannot open and
   * numbers them differently.
   *
   * `refresh` re-scans and opens each idle device to read its real
   * resolution — seconds, not milliseconds — so it is opt-in.
   */
  async cameras(refresh = false): Promise<CameraEntry[]> {
    try {
      const body = await request<{ cameras?: CameraEntry[] }>(
        this.url(`/cameras${refresh ? '?refresh=1' : ''}`),
        { timeoutMs: refresh ? 30000 : 4000 },
      );
      return body?.cameras ?? [];
    } catch {
      return [];
    }
  }

  /** POST /camera. By name where we have one — indices shuffle on replug. */
  setCamera(camera: { name?: string; source?: string }): Promise<unknown> {
    return postJson(this.url('/camera'), {
      name: camera.name ?? null,
      source: camera.source ?? null,
    }, { timeoutMs: 30000 });
  }

  state(): Promise<StateView> {
    return request<StateView>(this.url('/state'));
  }

  /** POST /hud — the instruction the operator typed, burned onto the frame. */
  setHud(text: string): Promise<unknown> {
    return postJson(this.url('/hud'), { text });
  }

  /* -------------------------------------------------------------- queries */

  count(selector: Selector, windowS = 1.0): Promise<CountResult> {
    return postJson<CountResult>(this.url('/query/count'), { selector, window_s: windowS });
  }

  look(selector?: Selector | null): Promise<LookResult> {
    return postJson<LookResult>(this.url('/query/look'), { selector: selector ?? null });
  }

  /* ----------------------------------------------------------- media urls */

  /** The enriched MJPEG. Cache-busted so a reconnect actually reopens the stream. */
  videoUrl(bust?: number): string {
    const u = this.url('/video');
    return bust ? `${u}${u.includes('?') ? '&' : '?'}_t=${bust}` : u;
  }

  /** Deliberately un-annotated: a vision model should see the world, not our drawings. */
  snapshotUrl(): string {
    return this.url('/snapshot');
  }

  /** The crop an event fired on. */
  eventSnapshotUrl(eventId: string): string {
    return this.url(`/snapshots/${eventId}.jpg`);
  }

  /* -------------------------------------------------------------- sockets */

  /** Per-frame StateView. Level-triggered, so a dropped message self-corrects. */
  stateSocketUrl(): string {
    return toWebSocketUrl(this.base, '/ws/state');
  }

  /** Things that happened, in the pipeline's own vocabulary. Replays a backlog on connect. */
  eventsSocketUrl(): string {
    return toWebSocketUrl(this.base, '/ws/events');
  }

}
