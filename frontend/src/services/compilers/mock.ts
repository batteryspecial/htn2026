/**
 * Keyword rules, not an LLM.
 *
 * Enough to exercise every UI path and every behaviour kind with no network at
 * all, and to keep the demo alive if the venue WiFi or the laptop running the
 * orchestrator falls over. ~0.5 s, entirely offline.
 *
 * It is deliberately shallow: it classifies the verb, extracts a noun phrase,
 * and fills the params that kind needs. Anything subtler is the real
 * compiler's job.
 */

import type { BehaviorKind, BehaviorSpec, Pick, Selector } from '../../contracts/behavior';
import { describeBehavior, type Program } from '../../contracts/program';
import type { Reachability } from '../perception';
import type { CompileResult, Compiler, StageSink } from './types';

/** Verb patterns, in priority order — the first match wins. */
const KINDS: { kind: BehaviorKind; re: RegExp }[] = [
  { kind: 'privacy', re: /\b(blur|pixelate|anonymi[sz]e|hide (?:the )?face)\b/ },
  { kind: 'count_line', re: /\b(cross|crossing|through the (?:door|line)|past the line)\b/ },
  { kind: 'pan_to', re: /\b(turn|pan|rotate|swivel|look) (?:the camera )?(?:\d+|left|right)/ },
  { kind: 'pose_trigger', re: /\b(rais\w+ (?:their |a )?hand|hand ?s? up|wave|gesture|pose)\b/ },
  { kind: 'watch', re: /\b(watch|guard|keep an eye|alert|warn|notify|tell me (?:when|if)|protect|sentinel)\b/ },
  { kind: 'highlight', re: /\b(detect|highlight|show me|find all|count|how many|every|all the)\b/ },
  { kind: 'track', re: /\b(follow|track|chase|go to|approach|centre on|center on)\b/ },
];

/** Nouns worth recognising as the head of a phrase. */
const NOUNS = [
  'person', 'people', 'human', 'face', 'dog', 'cat', 'duck', 'pencil', 'eraser',
  'pen', 'marker', 'bottle', 'cup', 'mug', 'chair', 'backpack', 'bag', 'phone',
  'book', 'ball', 'laptop', 'keyboard', 'mouse', 'shoe', 'shoes', 'hat', 'box',
  'can', 'car', 'wallet', 'table', 'hoodie', 'jacket',
];

const COLORS = [
  'red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink',
  'black', 'white', 'grey', 'gray', 'brown',
];

/** Kinds that act on every match rather than choosing one. */
const ALL_KINDS: BehaviorKind[] = ['highlight', 'count_line', 'privacy', 'pose_trigger'];

const LEADING_VERB =
  /^(please\s+)?(can you\s+|could you\s+)?(now\s+)?(go\s+)?(and\s+)?(detect|highlight|show me|find|count|watch|guard|follow|track|chase|blur|pixelate|keep an eye on|look at|point at|protect|alert me (?:when|if)|tell me (?:when|if)|notify me (?:when|if))\s+/i;

const ARTICLE = /^(the|a|an|that|this|all|every|any|anybody|anyone|someone|somebody)\s+/i;

/** "…, ignore black jackets" / "…, except mine" — a qualifier, not a second subject. */
const QUALIFIER = /\s*[,;]?\s*\b(but\s+)?(ignore|ignoring|except|excluding|unless|apart from|other than)\b\s*/i;

let counter = 0;

function classify(text: string): BehaviorKind {
  for (const { kind, re } of KINDS) if (re.test(text)) return kind;
  return 'highlight';
}

/** Split "watch the duck, ignore black jackets" into subject and qualifier. */
function splitQualifier(phrase: string): { subject: string; ignore: string | null } {
  const m = phrase.match(QUALIFIER);
  if (!m || m.index === undefined || m.index === 0) return { subject: phrase, ignore: null };
  return {
    subject: phrase.slice(0, m.index).trim(),
    ignore: phrase.slice(m.index + m[0].length).trim() || null,
  };
}

function head(phrase: string): string {
  let out = phrase.trim();
  for (let i = 0; i < 3; i++) {
    const before = out;
    out = out.replace(LEADING_VERB, '').replace(ARTICLE, '').trim();
    if (out === before) break;
  }
  return out;
}

function bareNoun(phrase: string): string {
  const words = phrase.toLowerCase().replace(/[^a-z\s']/g, ' ').split(/\s+/).filter(Boolean);
  for (let i = words.length - 1; i >= 0; i--) {
    const w = words[i];
    if (NOUNS.includes(w)) {
      if (w === 'people' || w === 'human') return 'person';
      if (w === 'shoes') return 'shoe';
      return w;
    }
  }
  return words[words.length - 1] ?? 'object';
}

/**
 * Detection runs on the union of every phrase, so several cost what one costs.
 * The full phrase finds a described object; the bare noun is the fallback.
 */
function detectPhrases(phrase: string): string[] {
  const noun = bareNoun(phrase);
  const full = phrase.toLowerCase().replace(/[^a-z\s']/g, ' ').replace(/\s+/g, ' ').trim();
  const out = [full, noun].filter(Boolean);
  return [...new Set(out)].slice(0, 4);
}

function buildSelector(phrase: string, kind: BehaviorKind): Selector {
  const noun = bareNoun(phrase);
  const isPerson = noun === 'person';
  const pick: Pick = ALL_KINDS.includes(kind)
    ? 'all'
    : /\b(that|him|her|this one|them)\b/.test(phrase) ? 'ref' : 'largest';

  // A colour on a person is an attribute to discriminate with; a colour on an
  // object is part of what makes it findable at all.
  const colour = COLORS.find((c) => phrase.toLowerCase().includes(c));
  if (isPerson && colour) {
    return {
      detect: ['person'],
      include: [`a person wearing ${colour} clothing`],
      exclude: [`a person not wearing ${colour}`],
      pick,
    };
  }

  return { detect: detectPhrases(phrase), pick };
}

function paramsFor(
  kind: BehaviorKind,
  text: string,
  ignore: string | null,
): Record<string, unknown> | undefined {
  switch (kind) {
    case 'count_line':
      return { line: [[0.5, 0.0], [0.5, 1.0]] };

    case 'privacy':
      return { mode: /pixelate/.test(text) ? 'pixelate' : 'blur' };

    case 'pan_to': {
      const degrees = Number(text.match(/(\d+)\s*(?:deg|degree|°)?/)?.[1] ?? 45);
      const left = /\bleft\b/.test(text);
      return { deg: left ? -degrees : degrees };
    }

    case 'watch': {
      if (ignore || /\b(approach|near|close|come|someone|anybody|anyone)\b/.test(text)) {
        // "ignore black jackets" is an exclude on the trigger's own selector.
        // Authorisation is not a separate mechanism.
        const other: Selector = { detect: ['person'], pick: 'all' };
        if (ignore) {
          other.include = ['a person in light coloured clothing'];
          other.exclude = [`a person wearing ${ignore}`];
        }
        return { triggers: [{ type: 'near', other }] };
      }
      if (/\b(moved?|taken|stolen|picked up)\b/.test(text)) {
        return { triggers: [{ type: 'moved', min_shift: 0.15 }] };
      }
      return { triggers: [{ type: 'missing', after_s: 2.0 }] };
    }

    default:
      return undefined;
  }
}

/** "laptop, phone and wallet" -> three subjects. */
function splitSubjects(phrase: string): string[] {
  const parts = phrase
    .split(/\s*,\s*|\s+and\s+/i)
    .map((p) => head(p))
    .filter((p) => p.length > 1);
  return parts.length ? parts.slice(0, 4) : [phrase];
}

export function mockCompile(text: string): Program {
  const clean = String(text ?? '').trim();
  const lower = clean.toLowerCase();
  const kind = classify(lower);

  // Everything after the colon in "guard the table: laptop, phone, wallet".
  const afterColon = clean.includes(':') ? clean.slice(clean.indexOf(':') + 1) : clean;
  const { subject: subjectText, ignore } = splitQualifier(afterColon);
  const subject = head(subjectText);

  // Only a colon really means several subjects. A comma is far more often a
  // qualifier ("watch the duck, ignore black jackets"), and splitting on it
  // turns one instruction into two behaviours guarding nonsense.
  const phrases = kind === 'watch' && clean.includes(':')
    ? splitSubjects(afterColon)
    : [subject];

  counter += 1;

  const behaviors: BehaviorSpec[] = phrases.map((phrase) => {
    const spec: BehaviorSpec = {
      kind,
      subject: buildSelector(phrase, kind),
      render: { label: phrase || bareNoun(phrase) },
    };
    const params = paramsFor(kind, lower, ignore);
    if (params) spec.params = params;
    if (kind === 'track') spec.render = { ...spec.render, trail: true };
    return spec;
  });

  return {
    behaviors,
    summary: behaviors.map(describeBehavior).join(', and '),
  };
}

export class MockCompiler implements Compiler {
  readonly name = 'mock';

  /** Drives the same stage strip the real pipeline does. */
  async compile(text: string, onStage?: StageSink): Promise<CompileResult> {
    if (!String(text ?? '').trim()) throw new Error('empty instruction');

    const instructionId = `mock_i_${Date.now()}_${++counter}`;
    const emit = (stage: string, delay: number) => {
      setTimeout(() => {
        onStage?.({ instruction_id: instructionId, stage, ts: Date.now() / 1000 });
      }, delay);
    };

    emit('received', 10);
    emit('compiled', 420);

    await new Promise((r) => setTimeout(r, 420 + Math.random() * 260));
    return { instruction_id: instructionId, program: mockCompile(text), raw: null };
  }

  async health(): Promise<Reachability> {
    return { up: true, detail: 'mock adapter — no network' };
  }
}
