/**
 * A **program** is what one instruction compiles to: the behaviours that
 * should be running afterwards.
 *
 * It is not a wire type. Perception has no notion of a program — each
 * behaviour is posted on its own to `POST /behaviors`. The grouping exists
 * because an instruction like "guard the table: laptop, phone, wallet" is
 * three behaviours that must be installed and torn down together.
 */

import type { BehaviorSpec, Selector } from './behavior';

export interface Program {
  /** Installed in order, between the same two frames. */
  behaviors: BehaviorSpec[];
  /** The agent's one-line reading of the instruction, for the HUD and speech. */
  summary?: string;
}

/** Kinds `POST /behaviors` will accept. `keyboard` is declared but not built. */
export const BUILT_KINDS = [
  'highlight', 'track', 'watch', 'count_line', 'privacy', 'pan_to', 'pose_trigger',
] as const;

export const PLANNED_KINDS = ['keyboard'] as const;

export const PICKS = ['all', 'largest', 'most_centered', 'ref'] as const;

export const TRIGGER_TYPES = ['missing', 'moved', 'near', 'appeared'] as const;

/** Contract key order, so the JSON always reads the same way on a projector. */
export const BEHAVIOR_KEYS = ['kind', 'subject', 'params', 'render', 'notify'] as const;
export const SELECTOR_KEYS = [
  'detect', 'include', 'exclude', 'relate', 'ref_id', 'pick', 'min_score', 'ref_min_sim',
] as const;
export const RELATE_KEYS = ['contains', 'lower_frac'] as const;
export const RENDER_KEYS = ['color', 'mask', 'trail', 'boxes', 'label'] as const;

/** Pipeline-side defaults, from linker/schemas.py. A spec omitting these gets these. */
export const DEFAULTS = {
  pick: 'all' as const,
  min_score: 0.75,
  ref_min_sim: 0.75,
  lower_frac: 0.4,
  cooldown_s: 5.0,
  missing_after_s: 2.0,
  moved_min_shift: 0.15,
  near_margin: 0.1,
  line: [[0.5, 0.0], [0.5, 1.0]] as number[][],
};

/** The one-line form of a selector, matching `Selector.summary()` server-side. */
export function selectorSummary(s: Selector | undefined): string {
  if (!s) return '?';
  let bits = (s.detect ?? []).join('+');
  if (s.include?.length) bits += ` ✓${s.include.join(',')}`;
  if (s.exclude?.length) bits += ` ✗${s.exclude.join(',')}`;
  if (s.relate) bits += ` ⊃${s.relate.contains}`;
  if (s.ref_id) bits += ` ref:${s.ref_id}`;
  return bits;
}

const VERB: Record<string, string> = {
  highlight: 'highlighting',
  track: 'following',
  watch: 'watching',
  count_line: 'counting',
  privacy: 'blurring',
  pan_to: 'panning to',
  pose_trigger: 'watching for a gesture from',
  keyboard: 'reading',
};

/** What one behaviour does, in words. */
export function describeBehavior(b: BehaviorSpec): string {
  const what = b.render?.label || selectorSummary(b.subject);
  if (b.kind === 'pan_to') {
    const deg = Number(b.params?.deg ?? 0);
    return `panning ${Math.abs(deg)}° ${deg < 0 ? 'left' : 'right'}`;
  }
  return `${VERB[b.kind] ?? b.kind} ${what}`;
}

/** Spoken and on-screen summary of a whole program. */
export function summarize(program: Program | null | undefined): string {
  if (!program?.behaviors?.length) return 'no objective';
  if (program.summary) return program.summary;
  return program.behaviors.map(describeBehavior).join(', and ');
}

/** Stable key order for display. Does not change meaning. */
export function reorder(program: Program): BehaviorSpec[] {
  const pickFirst = <T extends object>(obj: T, keys: readonly string[]): T => {
    const src = obj as Record<string, unknown>;
    const out: Record<string, unknown> = {};
    for (const k of keys) if (src[k] !== undefined) out[k] = src[k];
    for (const k of Object.keys(src)) if (out[k] === undefined) out[k] = src[k];
    return out as T;
  };

  return program.behaviors.map((b) => {
    const out = pickFirst(b, BEHAVIOR_KEYS);
    if (out.subject) out.subject = pickFirst(out.subject, SELECTOR_KEYS);
    if (out.subject?.relate) out.subject.relate = pickFirst(out.subject.relate, RELATE_KEYS);
    if (out.render) out.render = pickFirst(out.render, RENDER_KEYS);
    return out;
  });
}
