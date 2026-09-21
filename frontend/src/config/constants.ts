/** Fixed vocabulary the UI draws from. Kept in one file so a rename is one edit. */

import type { BehaviorState } from '../contracts/behavior';

/**
 * The stage strip, in the order it fills.
 *
 * `received` and `compiled` are ours — the pipeline knows nothing about the
 * LLM compile. `applied` is the 201 from POST /behaviors. `active` is the
 * pipeline's `acquired` event, and it is the number the project is measured on.
 *
 * Optional stages only light when the pipeline has to load a model. Against
 * the behaviour layer they never fire at all, which is why they are not in the
 * default list — `server/legacy.py::STAGE` forwards `acquired` and the two
 * failure events and nothing else.
 */
export const STAGES = ['received', 'compiled', 'applied', 'active'] as const;
export type Stage = (typeof STAGES)[number];

export const OPTIONAL_STAGES: readonly string[] = ['model_loading', 'model_loaded', 'prepared'];

/** Stages that mean the run is over and went wrong. */
export const BAD_STAGES: readonly string[] = ['rejected', 'failed', 'no_target'];

/** Badge colour per behaviour state, plus the service statuses GET /health reports. */
export const STATE_CLASS: Record<string, string> = {
  // behaviour states, from BehaviorState
  TRACKING: 'pass', ACQUIRING: 'pending', EDGE: 'warn', LOST: 'warn',
  SEARCHING: 'pending', ACTIVE: 'pass', ARMED: 'pass', ARMING: 'pending',
  FIRED: 'warn', COOLDOWN: 'pending', GUIDING: 'pending', REACHED: 'pass',
  LOCKED: 'pass', PAUSED: 'warn', FAILED: 'fail',
  // service status from GET /health
  OK: 'pass', BOOTING: 'idle', NO_CAMERA: 'fail', FAULT: 'fail',
  IDLE: 'idle',
};

export const stateClass = (state: BehaviorState | string | null | undefined): string =>
  (state && STATE_CLASS[state]) || 'idle';

/**
 * Pipeline events that mean something to the stage strip.
 *
 * Deliberately the same three `server/legacy.py::STAGE` forwards. Everything
 * else in the event vocabulary — `count_changed`, `crossed`, `armed` — has no
 * stage equivalent, and inventing one would put unknown entries on the strip.
 * Those belong in the events panel, where they are shown in full.
 */
export const EVENT_STAGE: Record<string, string> = {
  acquired: 'active',
  behavior_failed: 'failed',
  camera_lost: 'failed',
};

/** Event types that are worth an alert rather than a log line. */
export const ALERT_EVENTS: readonly string[] = [
  'near', 'missing', 'moved', 'hand_raised', 'behavior_failed', 'camera_lost',
];

/** One-click instructions, straight off the demo list in README.md. */
export const EXAMPLE_INSTRUCTIONS: readonly string[] = [
  'track the pencil',
  'track the pencil and the eraser',
  'follow the person in red shoes',
  'follow the dog',
];

export const HISTORY_KEY = 'retask.history.v1';
export const HISTORY_MAX = 8;

/** Stop counting rather than run forever when the pipeline never acquires. */
export const ACQUIRE_TIMEOUT_MS = 25000;
