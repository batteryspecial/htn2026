/**
 * Client-side mirror of the pipeline's validation, so a bad program shows up
 * here in red rather than as a 422 from `POST /behaviors`.
 *
 * Two layers, matching the two the server applies:
 *   - the pydantic contract in `linker/schemas.py` (extra="forbid", bounds)
 *   - the per-kind `params` check `_params_error` runs by building the
 *     behaviour and throwing it away
 *
 * Warnings are things the pipeline accepts but that mean the behaviour will
 * not do what was asked. They never block a post.
 */

import type { Selector } from './behavior';
import {
  BUILT_KINDS, PICKS, PLANNED_KINDS, type Program, TRIGGER_TYPES,
} from './program';

export interface Problem {
  path: string;
  msg: string;
}

export interface ValidationResult {
  ok: boolean;
  errors: Problem[];
  warnings: Problem[];
}

type Dict = Record<string, unknown>;

const isObject = (v: unknown): v is Dict =>
  v !== null && typeof v === 'object' && !Array.isArray(v);

const isNonEmptyString = (v: unknown): v is string =>
  typeof v === 'string' && v.trim().length > 0;

const SELECTOR_KEYS = [
  'detect', 'include', 'exclude', 'ref_id', 'relate', 'pick', 'min_score', 'ref_min_sim',
];
const RELATE_KEYS = ['contains', 'lower_frac'];
const RENDER_KEYS = ['color', 'mask', 'trail', 'boxes', 'label'];
const BEHAVIOR_KEYS = ['kind', 'subject', 'params', 'render', 'notify'];

/** Params each kind understands. Anything else is silently ignored server-side. */
const KIND_PARAMS: Record<string, string[]> = {
  highlight: [],
  track: [],
  watch: ['triggers', 'cooldown_s'],
  count_line: ['line'],
  privacy: ['mode', 'keep_ref'],
  pan_to: ['deg', 'hfov_deg', 'tolerance_deg'],
  pose_trigger: ['gesture', 'cooldown_s', 'hold_frames'],
};

class Collector {
  readonly errors: Problem[] = [];
  readonly warnings: Problem[] = [];

  err(path: string, msg: string) { this.errors.push({ path, msg }); }
  warn(path: string, msg: string) { this.warnings.push({ path, msg }); }

  forbidExtra(obj: Dict, allowed: readonly string[], path: string) {
    for (const k of Object.keys(obj)) {
      if (!allowed.includes(k)) {
        this.err(`${path}.${k}`,
          'not in the contract — pydantic sets extra="forbid", so this is a 422');
      }
    }
  }

  number(value: unknown, path: string, lo: number, hi: number, exclusiveLow = false) {
    if (typeof value !== 'number' || Number.isNaN(value)) {
      this.err(path, 'must be a number');
      return;
    }
    if (exclusiveLow ? value <= lo : value < lo) {
      this.err(path, `must be ${exclusiveLow ? 'greater than' : 'at least'} ${lo}`);
    } else if (value > hi) {
      this.err(path, `must be at most ${hi}`);
    }
  }

  stringList(value: unknown, path: string, min: number, max: number) {
    if (!Array.isArray(value)) {
      this.err(path, `must be an array of ${min}-${max} strings`);
      return;
    }
    if (value.length < min || value.length > max) {
      this.err(path, `must hold ${min}-${max} entries (got ${value.length})`);
    }
    value.forEach((v, i) => {
      if (!isNonEmptyString(v)) this.err(`${path}[${i}]`, 'must be a non-empty string');
    });
  }
}

/* -------------------------------------------------------------- selector */

function validateSelector(s: unknown, path: string, c: Collector): void {
  if (!isObject(s)) { c.err(path, 'must be an object'); return; }

  c.stringList(s.detect, `${path}.detect`, 1, 8);
  if (s.include !== undefined) c.stringList(s.include, `${path}.include`, 0, 4);
  if (s.exclude !== undefined) c.stringList(s.exclude, `${path}.exclude`, 0, 4);

  if (s.ref_id !== undefined && s.ref_id !== null && !isNonEmptyString(s.ref_id)) {
    c.err(`${path}.ref_id`, 'must be a reference id from POST /references, or null');
  }

  if (s.relate !== undefined && s.relate !== null) {
    if (!isObject(s.relate)) {
      c.err(`${path}.relate`, 'must be an object');
    } else {
      if (!isNonEmptyString(s.relate.contains)) {
        c.err(`${path}.relate.contains`, 'required non-empty string');
      }
      if (s.relate.lower_frac !== undefined) {
        c.number(s.relate.lower_frac, `${path}.relate.lower_frac`, 0, 1, true);
      }
      c.forbidExtra(s.relate, RELATE_KEYS, `${path}.relate`);
    }
  }

  if (s.pick !== undefined && !PICKS.includes(s.pick as never)) {
    c.err(`${path}.pick`, `must be one of ${PICKS.join(' | ')}`);
  }

  if (s.min_score !== undefined) c.number(s.min_score, `${path}.min_score`, 0, 1);
  if (s.ref_min_sim !== undefined) c.number(s.ref_min_sim, `${path}.ref_min_sim`, 0, 1);

  c.forbidExtra(s, SELECTOR_KEYS, path);

  /* --- accepted, but will not do what was asked --- */

  // `pick: "ref"` latches onto one instance. Without a ref_id it latches onto
  // whatever it saw first, which is right for "follow him" and wrong for a
  // photo upload that never happened.
  if (s.pick === 'ref' && !s.ref_id) {
    c.warn(`${path}.pick`,
      '"ref" with no ref_id latches onto the first match — correct for "follow him", '
      + 'but a photo upload should set ref_id from POST /references');
  }
}

/* ---------------------------------------------------------------- params */

function validateWatchParams(params: Dict, path: string, c: Collector): void {
  const triggers = params.triggers;
  if (triggers === undefined) return; // defaults to a `missing` trigger

  if (!Array.isArray(triggers) || !triggers.length) {
    c.err(`${path}.triggers`, "watch needs a non-empty 'triggers' list");
    return;
  }

  triggers.forEach((t, i) => {
    const at = `${path}.triggers[${i}]`;
    if (!isObject(t)) { c.err(at, 'must be an object'); return; }

    if (!TRIGGER_TYPES.includes(t.type as never)) {
      c.err(`${at}.type`, `unknown trigger ${JSON.stringify(t.type)}; `
        + `expected one of ${[...TRIGGER_TYPES].sort().join(', ')}`);
      return;
    }

    if (t.type === 'near' || t.type === 'appeared') {
      if (t.other === undefined) {
        c.err(`${at}.other`, `trigger "${t.type}" needs an 'other' selector`);
      } else {
        validateSelector(t.other, `${at}.other`, c);
      }
    }

    if (t.after_s !== undefined) c.number(t.after_s, `${at}.after_s`, 0, 3600);
    if (t.min_shift !== undefined) c.number(t.min_shift, `${at}.min_shift`, 0, 2);
    if (t.margin !== undefined) c.number(t.margin, `${at}.margin`, 0, 1);
  });
}

function validateParams(kind: string, params: unknown, path: string, c: Collector): void {
  if (params === undefined) return;
  if (!isObject(params)) { c.err(path, 'must be an object'); return; }

  const known = KIND_PARAMS[kind];
  if (known) {
    for (const k of Object.keys(params)) {
      if (!known.includes(k)) {
        c.warn(`${path}.${k}`, known.length
          ? `${kind} does not read "${k}" — it understands ${known.join(', ')}`
          : `${kind} takes no params, so "${k}" is ignored`);
      }
    }
  }

  switch (kind) {
    case 'watch':
      validateWatchParams(params, path, c);
      if (params.cooldown_s !== undefined) {
        c.number(params.cooldown_s, `${path}.cooldown_s`, 0, 3600);
      }
      break;

    case 'count_line': {
      const line = params.line;
      if (line === undefined) break;
      const flat = Array.isArray(line) && line.length === 2
        && line.every((p) => Array.isArray(p) && p.length === 2
          && p.every((n) => typeof n === 'number'));
      if (!flat) {
        c.err(`${path}.line`, "count_line needs 'line': [[x1,y1],[x2,y2]] in 0..1 coordinates");
      } else {
        const [[ax, ay], [bx, by]] = line as number[][];
        if (ax === bx && ay === by) {
          c.err(`${path}.line`, 'count_line needs two different points');
        }
        for (const [n, name] of [[ax, 'x1'], [ay, 'y1'], [bx, 'x2'], [by, 'y2']] as const) {
          if (n < 0 || n > 1) {
            c.warn(`${path}.line`, `${name} is ${n} — the line is normalized, so 0..1`);
          }
        }
      }
      break;
    }

    case 'privacy':
      if (params.mode !== undefined && params.mode !== 'blur' && params.mode !== 'pixelate') {
        c.err(`${path}.mode`,
          `privacy mode must be 'blur' or 'pixelate', not ${JSON.stringify(params.mode)}`);
      }
      if (params.keep_ref !== undefined && params.keep_ref !== null
          && !isNonEmptyString(params.keep_ref)) {
        c.err(`${path}.keep_ref`, 'must be a reference id from POST /references, or null');
      }
      break;

    case 'pan_to':
      if (params.deg !== undefined) c.number(params.deg, `${path}.deg`, -360, 360);
      if (params.hfov_deg !== undefined) c.number(params.hfov_deg, `${path}.hfov_deg`, 1, 180);
      if (params.tolerance_deg !== undefined) {
        c.number(params.tolerance_deg, `${path}.tolerance_deg`, 0, 90);
      }
      if (params.deg === undefined || params.deg === 0) {
        c.warn(`${path}.deg`, 'pan_to with no angle reaches its target immediately');
      }
      break;

    case 'pose_trigger':
      if (params.gesture !== undefined && !isNonEmptyString(params.gesture)) {
        c.err(`${path}.gesture`, 'must be a non-empty string');
      }
      if (params.cooldown_s !== undefined) {
        c.number(params.cooldown_s, `${path}.cooldown_s`, 0, 3600);
      }
      if (params.hold_frames !== undefined) {
        c.number(params.hold_frames, `${path}.hold_frames`, 1, 300);
      }
      break;

    default:
      break;
  }
}

/* -------------------------------------------------------------- behaviour */

function validateRender(r: unknown, path: string, c: Collector): void {
  if (r === undefined || r === null) return;
  if (!isObject(r)) { c.err(path, 'must be an object'); return; }

  if (r.color !== undefined && r.color !== null) {
    if (typeof r.color !== 'string') c.err(`${path}.color`, 'must be a colour string');
    else if (!/^#[0-9a-fA-F]{6}$/.test(r.color)) {
      c.warn(`${path}.color`, `"${r.color}" is not #RRGGBB — the renderer may fall back`);
    }
  }
  for (const flag of ['mask', 'trail', 'boxes'] as const) {
    if (r[flag] !== undefined && typeof r[flag] !== 'boolean') {
      c.err(`${path}.${flag}`, 'must be true or false');
    }
  }
  if (r.label !== undefined && r.label !== null && typeof r.label !== 'string') {
    c.err(`${path}.label`, 'must be a string');
  }
  c.forbidExtra(r, RENDER_KEYS, path);
}

function validateBehavior(b: unknown, path: string, c: Collector): void {
  if (!isObject(b)) { c.err(path, 'must be an object'); return; }

  const kind = b.kind;
  if (!isNonEmptyString(kind)) {
    c.err(`${path}.kind`, `required — one of ${BUILT_KINDS.join(', ')}`);
  } else if (PLANNED_KINDS.includes(kind as never)) {
    c.err(`${path}.kind`, `"${kind}" is declared in the contract but not implemented yet; `
      + `available: ${[...BUILT_KINDS].sort().join(', ')}`);
  } else if (!BUILT_KINDS.includes(kind as never)) {
    c.err(`${path}.kind`, `"${kind}" is not a behaviour kind; `
      + `available: ${[...BUILT_KINDS].sort().join(', ')}`);
  }

  if (b.subject === undefined) c.err(`${path}.subject`, 'required');
  else validateSelector(b.subject, `${path}.subject`, c);

  if (isNonEmptyString(kind)) validateParams(kind, b.params, `${path}.params`, c);
  validateRender(b.render, `${path}.render`, c);

  if (b.notify !== undefined && typeof b.notify !== 'boolean') {
    c.err(`${path}.notify`, 'must be true or false');
  }

  c.forbidExtra(b, BEHAVIOR_KEYS, path);

  /* --- accepted, but will not do what was asked --- */

  const subject = b.subject as Selector | undefined;
  const pick = subject?.pick ?? 'all';

  if ((kind === 'highlight' || kind === 'count_line' || kind === 'privacy') && pick !== 'all') {
    c.warn(`${path}.subject.pick`,
      `${kind} acts on every match, so pick "${pick}" narrows it to one — use "all"`);
  }
  if (kind === 'track' && pick === 'all') {
    c.warn(`${path}.subject.pick`,
      'track follows one thing; "all" gives it nothing to choose — use "largest" or "ref"');
  }
  if (kind === 'privacy' && (b.params as Dict | undefined)?.keep_ref
      && !subject?.ref_id) {
    c.warn(`${path}.params.keep_ref`,
      'keep_ref spares a reference the subject never scores against — set subject.ref_id too');
  }
  if (!isNonEmptyString((b.render as Dict | undefined)?.label)) {
    c.warn(`${path}.render.label`,
      'no label — the behaviours panel and the HUD will show the raw selector');
  }
}

/* ---------------------------------------------------------------- program */

export function validateProgram(program: unknown): ValidationResult {
  const c = new Collector();

  if (!isObject(program) || !Array.isArray(program.behaviors)) {
    return {
      ok: false,
      errors: [{ path: '$', msg: 'not a program — expected { behaviors: [...] }' }],
      warnings: [],
    };
  }

  const behaviors = program.behaviors as unknown[];
  if (!behaviors.length) {
    c.err('behaviors', 'a program must hold at least one behaviour');
  }
  behaviors.forEach((b, i) => validateBehavior(b, `behaviors[${i}]`, c));

  return { ok: c.errors.length === 0, errors: c.errors, warnings: c.warnings };
}

/** Kinds the live pipeline reports as built, when we have asked it. */
export function unavailableKinds(
  program: Program,
  available: string[] | null,
): Problem[] {
  if (!available?.length) return [];
  return program.behaviors
    .map((b, i) => ({ b, i }))
    .filter(({ b }) => b.kind && !available.includes(b.kind))
    .map(({ b, i }) => ({
      path: `behaviors[${i}].kind`,
      msg: `this pipeline does not have "${b.kind}"; it offers ${available.join(', ')}`,
    }));
}
