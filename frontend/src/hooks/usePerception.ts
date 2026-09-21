/**
 * Live reads off the pipeline: per-frame state, the event log, health, and the
 * model and camera registries. One hook per channel, all of them self-healing.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import type {
  BehaviorView, CameraEntry, Health, ModelEntry, PerceptionEvent, StateView, TrackView,
} from '../contracts/behavior';
import type { PerceptionClient, Reachability } from '../services/perception';
import { ReconnectingSocket } from '../services/socket';

/** WS /ws/state — one StateView per frame. Level-triggered: drops self-correct. */
export function usePerceptionState(perception: PerceptionClient, base: string): StateView | null {
  const [state, setState] = useState<StateView | null>(null);

  useEffect(() => {
    setState(null);
    const socket = new ReconnectingSocket<StateView>(perception.stateSocketUrl(), {
      onMessage: setState,
      onClose: () => setState(null),
    });
    socket.connect();
    return () => socket.close();
  }, [perception, base]);

  return state;
}

/**
 * The behaviour whose state is worth showing. With several running, the one
 * that has actually matched something wins.
 */
export function leadBehavior(state: StateView | null): BehaviorView | null {
  const behaviors = state?.behaviors ?? [];
  return behaviors.find((b) => b.state === 'TRACKING')
    ?? behaviors.find((b) => b.track_ids.length > 0)
    ?? behaviors[0]
    ?? null;
}

/** The track a behaviour is on, or the first one in frame if it has none. */
export function leadTrack(state: StateView | null, behavior: BehaviorView | null): TrackView | null {
  if (!state) return null;
  const wanted = behavior?.track_ids ?? [];
  if (wanted.length) return state.tracks.find((t) => wanted.includes(t.track_id)) ?? null;
  return behavior ? null : state.tracks[0] ?? null;
}

export interface EventLog {
  events: PerceptionEvent[];
  clear: () => void;
}

/** WS /ws/events — the pipeline's own vocabulary. Replays a backlog on connect. */
export function usePerceptionEvents(
  perception: PerceptionClient,
  base: string,
  limit = 50,
): EventLog {
  const [events, setEvents] = useState<PerceptionEvent[]>([]);
  const seen = useRef(new Set<string>());

  useEffect(() => {
    seen.current.clear();
    const socket = new ReconnectingSocket<PerceptionEvent>(perception.eventsSocketUrl(), {
      onMessage: (ev) => {
        if (!ev?.type) return;
        const key = ev.id || `${ev.ts}:${ev.type}:${ev.behavior_id ?? ''}`;
        if (seen.current.has(key)) return;
        seen.current.add(key);
        if (seen.current.size > limit * 4) {
          seen.current = new Set([...seen.current].slice(-limit * 2));
        }
        setEvents((prev) => [ev, ...prev].slice(0, limit));
      },
    });
    socket.connect();
    return () => socket.close();
  }, [perception, base, limit]);

  const clear = useCallback(() => setEvents([]), []);
  return { events, clear };
}

/** GET /health on a timer: fps, resident model and device. */
export function usePerceptionHealth(
  perception: PerceptionClient,
  base: string,
  intervalMs = 2000,
): Reachability | null {
  const [reach, setReach] = useState<Reachability | null>(null);

  useEffect(() => {
    let alive = true;
    const poll = async () => {
      const r = await perception.health();
      if (alive) setReach(r);
    };
    poll();
    const timer = setInterval(poll, intervalMs);
    return () => { alive = false; clearInterval(timer); };
  }, [perception, base, intervalMs]);

  return reach;
}

export const healthOf = (reach: Reachability | null): Health | null =>
  (reach?.up ? reach.health ?? null : null);

/** GET /models. Refetched when the pipeline URL moves, and on demand. */
export function useModels(perception: PerceptionClient, base: string) {
  const [models, setModels] = useState<ModelEntry[]>([]);
  const mounted = useRef(true);

  const refresh = useCallback(async () => {
    const list = await perception.models();
    if (mounted.current) setModels(list);
  }, [perception]);

  useEffect(() => {
    mounted.current = true;
    refresh();
    return () => { mounted.current = false; };
  }, [refresh, base]);

  return { models, names: models.map((m) => m.name).filter(Boolean), refresh };
}

/**
 * Which cameras the pipeline can open, from `GET /cameras`.
 *
 * Server-side enumeration on purpose. The camera belongs to whichever machine
 * runs perception; `navigator.mediaDevices` would describe this browser's
 * machine, which is not the same thing and not the same indices.
 *
 * `refresh` re-scans and opens each idle device to read its resolution, which
 * takes seconds — so it is a button, not a poll.
 */
export function useCameras(perception: PerceptionClient, base: string) {
  const [cameras, setCameras] = useState<CameraEntry[]>([]);
  const mounted = useRef(true);

  const refresh = useCallback(async (rescan = false) => {
    const list = await perception.cameras(rescan);
    if (mounted.current) setCameras(list);
  }, [perception]);

  useEffect(() => {
    mounted.current = true;
    refresh();
    return () => { mounted.current = false; };
  }, [refresh, base]);

  return { cameras, active: cameras.find((c) => c.active) ?? null, refresh };
}

/**
 * Which kinds this pipeline actually has, from `GET /behaviors`.
 *
 * The contract declares eight; `keyboard` is not built. Asking rather than
 * hardcoding means a spec naming an unbuilt kind is caught here instead of as
 * a 422 — and a kind landing later needs no frontend change.
 */
export function useBehaviorKinds(perception: PerceptionClient, base: string): string[] {
  const [kinds, setKinds] = useState<string[]>([]);

  useEffect(() => {
    let alive = true;
    perception.listBehaviors()
      .then((r) => { if (alive) setKinds(r.kinds?.available ?? []); })
      .catch(() => { /* the validator falls back to its own list */ });
    return () => { alive = false; };
  }, [perception, base]);

  return kinds;
}
