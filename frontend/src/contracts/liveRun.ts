import type { BehaviorSpec, BehaviorView, StateView } from './behavior';
import type { Program } from './program';

/** Receipts are returned over HTTP as well as the optional trace socket. */
export interface InstalledBehavior {
  id: string;
  spec: BehaviorSpec;
}

export function readInstallations(value: unknown): InstalledBehavior[] {
  if (!Array.isArray(value)) return [];
  return value.filter((item): item is InstalledBehavior => (
    item && typeof item.id === 'string' && item.id.length > 0
    && item.spec && typeof item.spec.kind === 'string'
    && Array.isArray(item.spec.subject?.detect)
    && item.spec.subject.detect.length > 0
  ));
}

export function programFromInstallations(items: InstalledBehavior[]): Program | null {
  return items.length ? { behaviors: items.map((item) => item.spec) } : null;
}

/** A completed conversation is not evidence that a behavior was installed. */
export function turnBadge(ok: boolean, program: Program | null) {
  if (!ok) return { kind: 'fail' as const, text: 'ERROR' };
  return program
    ? { kind: 'pending' as const, text: 'COMPILED' }
    : { kind: 'idle' as const, text: 'REPLY' };
}

/** A receipt is accepted work; pipeline state proves it was installed. */
export function installationsApplied(ids: string[], state: StateView | null): boolean {
  if (!ids.length || !state) return false;
  return ids.every((id) => state.behaviors.some((item) => item.id === id));
}

function isActive(behavior: BehaviorView): boolean {
  if (behavior.state === 'ACTIVE') return behavior.matches > 0;
  return ['TRACKING', 'EDGE', 'ARMED', 'FIRED', 'COOLDOWN', 'GUIDING', 'REACHED', 'LOCKED']
    .includes(behavior.state);
}

/** Never let another turn's target acquisition complete this turn's strip. */
export function installationsActive(ids: string[], state: StateView | null): boolean {
  if (!ids.length || !state?.camera_ok) return false;
  return ids.every((id) => {
    const behavior = state.behaviors.find((item) => item.id === id);
    return !!behavior && isActive(behavior);
  });
}

/** The pipeline remains authoritative after a reply, page reload, or DROP. */
export function liveObjective(state: StateView | null): string | null {
  if (!state) return 'Pipeline state unavailable';
  if (!state.behaviors.length) return null;
  return state.behaviors.map((behavior) => {
    const label = behavior.label && behavior.label !== behavior.kind ? behavior.label : null;
    return `${behavior.kind}: ${label || behavior.spec}`;
  }).join(' · ');
}
