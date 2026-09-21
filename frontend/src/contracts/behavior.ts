/**
 * TypeScript mirror of `linker/schemas.py` — the shared wire contract.
 *
 * `linker/schemas.py` is the source of truth and is owned jointly. This file
 * must be changed only to follow it. Everything here is normalized: positions
 * are [-1, 1] offsets from frame centre, sizes are frame fractions.
 */

/* ---------------------------------------------------------------- selector */

/** Which matches a behaviour acts on. */
export type Pick = 'all' | 'largest' | 'most_centered' | 'ref';

/** Containment: keep a subject only if another class sits low inside it. */
export interface Relate {
  contains: string;
  /** Fraction of the keep-box height, from the bottom, the contained centre must fall within. */
  lower_frac?: number;
}

/** What counts as a subject. One shape for the whole demo list. */
export interface Selector {
  /** 1-8 detector prompts: YOLOE text, or labels from a fixed vocabulary. */
  detect: string[];
  /** CLIP checks on the subject's crop. A subject must match every include… */
  include?: string[];
  /** …and no exclude. */
  exclude?: string[];
  /** A reference registered via POST /references. */
  ref_id?: string | null;
  relate?: Relate | null;
  pick?: Pick;
  /** Contrastive pass mark for include/exclude. Errs toward matching too little. */
  min_score?: number;
  /** Cosine a crop must reach to match the reference. */
  ref_min_sim?: number;
}

/* -------------------------------------------------------------- behaviours */

export type BehaviorKind =
  | 'highlight'
  | 'track'
  | 'watch'
  | 'count_line'
  | 'privacy'
  | 'pan_to'
  | 'pose_trigger'
  | 'keyboard';

/**
 * Every state any kind can be in. Which are reachable is per kind:
 *   highlight / count_line / privacy / pose_trigger : ACTIVE
 *   track    : ACQUIRING -> TRACKING <-> EDGE -> LOST -> SEARCHING
 *   watch    : ARMING -> ARMED -> FIRED -> COOLDOWN -> ARMED
 *   pan_to   : GUIDING -> REACHED
 *   keyboard : SEARCHING <-> LOCKED
 * PAUSED is reachable from anywhere: the active model cannot see the subject.
 */
export type BehaviorState =
  | 'ACTIVE' | 'PAUSED' | 'FAILED'
  | 'ACQUIRING' | 'TRACKING' | 'EDGE' | 'LOST' | 'SEARCHING'
  | 'ARMING' | 'ARMED' | 'FIRED' | 'COOLDOWN'
  | 'GUIDING' | 'REACHED' | 'LOCKED';

/** How a behaviour wants to look. Primitives only — the agent never writes drawing code. */
export interface RenderSpec {
  color?: string | null;
  mask?: boolean;
  trail?: boolean;
  boxes?: boolean;
  label?: string | null;
}

/** Body of POST /behaviors. The id is assigned by the server. */
export interface BehaviorSpec {
  kind: BehaviorKind;
  subject: Selector;
  /** Kind-specific settings, validated per kind rather than in the contract. */
  params?: Record<string, unknown>;
  render?: RenderSpec;
  /** Whether events from this behaviour should wake the agent. */
  notify?: boolean;
}

/** Response to POST /behaviors. */
export interface BehaviorCreated {
  id: string;
}

/** A behaviour as seen from outside. */
export interface BehaviorView {
  id: string;
  kind: BehaviorKind;
  state: BehaviorState;
  /** One-line summary, for the HUD and the trace. */
  spec: string;
  label?: string | null;
  matches: number;
  track_ids: number[];
  since: number;
  /** Why PAUSED, why FAILED. */
  detail?: string | null;
  notify: boolean;
  data: Record<string, unknown>;
}

/* ------------------------------------------------------------------ events */

export type EventType =
  | 'count_changed'
  | 'acquired' | 'lost' | 'reacquired'
  | 'armed' | 'missing' | 'moved' | 'near' | 'appeared'
  | 'crossed' | 'reached' | 'hand_raised'
  | 'keyboard_locked' | 'keyboard_lost' | 'step'
  | 'model_switched' | 'camera_lost' | 'camera_ok'
  | 'behavior_failed';

/** Something happened. Edge-triggered, for the agent and the UI. */
export interface PerceptionEvent {
  id: string;
  ts: number;
  /** Null for system events. */
  behavior_id?: string | null;
  type: EventType;
  detail: string;
  data: Record<string, unknown>;
  snapshot_url?: string | null;
  /** False means "show it, do not wake the agent". */
  notify: boolean;
}

/* ------------------------------------------------------------ observations */

/** One tracked thing, normalized. */
export interface TrackView {
  track_id: number;
  label: string;
  conf: number;
  cx: number;
  cy: number;
  area: number;
  attributes: Record<string, number>;
}

/** GET /state, and one message per frame on WS /ws/state. */
export interface StateView {
  ts: number;
  model?: string | null;
  fps: number;
  camera_ok: boolean;
  hud?: string | null;
  behaviors: BehaviorView[];
  tracks: TrackView[];
  refs: string[];
}

/** GET /health. Deliberately small. */
export interface Health {
  status: 'booting' | 'ok' | 'no_camera' | 'fault';
  fps: number;
  model?: string | null;
  camera_ok: boolean;
  behaviors: number;
  device: string;
  detail?: string | null;
}

/* ----------------------------------------------------------------- queries */

export interface CountQuery {
  selector: Selector;
  window_s?: number;
}

export interface CountResult {
  count: number;
  samples: number;
  window_s: number;
}

export interface LookQuery {
  selector?: Selector | null;
}

export interface LookResult {
  ts: number;
  tracks: TrackView[];
}

/** POST /hud. The instruction the operator typed, shown on the frame. */
export interface HudText {
  text: string;
}

/** POST /model. The role is inferred from the model unless given. */
export interface ModelChoice {
  name: string;
  role?: string | null;
}

/** An entry from GET /models. Not in schemas.py — it is the registry's own view. */
/**
 * One camera, from `GET /cameras` on the pipeline.
 *
 * Enumerated server-side, because that is where the camera is opened. An
 * `index` of -1 means the source is not a camera device at all — a file or a
 * stream URL, which `Capture` supports and no device list contains.
 */
export interface CameraEntry {
  index: number;
  name: string;
  /** Which OpenCV backend the index is valid for: dshow, msmf, v4l2. */
  backend?: string;
  active?: boolean;
  available?: boolean;
  /** What the driver actually gave, not what was asked for. */
  width?: number | null;
  height?: number | null;
  fps?: number | null;
  detail?: string | null;
}

export interface ModelEntry {
  name: string;
  open_vocab?: boolean;
  classes?: string[] | null;
  loaded?: boolean;
  available?: boolean;
  detail?: string | null;
}
