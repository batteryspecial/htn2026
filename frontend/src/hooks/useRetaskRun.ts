/**
 * One run, start to finish: compile, validate, dispatch, settle.
 *
 * The headline number is instruction sent -> the pipeline acquiring, because
 * that is the metric the project is built around. Compile time is the small
 * line underneath. If the program is applied but nothing is ever acquired the
 * clock stops after 25 s and says so rather than counting forever.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { ACQUIRE_TIMEOUT_MS, BAD_STAGES } from '../config/constants';
import type { Settings } from '../config/settings';
import { summarize, type Program } from '../contracts/program';
import { unavailableKinds, validateProgram } from '../contracts/validateProgram';
import type { Compiler, StageEvent } from '../services/compilers';
import { friendlyError } from '../services/http';
import type { PerceptionClient } from '../services/perception';
import type { HistoryEntry } from './useHistory';
import { seconds as fmt } from '../utils/format';

export type BadgeKind = 'idle' | 'pending' | 'pass' | 'warn' | 'fail';
export type NoticeKind = 'error' | 'warn' | 'ask';

export interface Notice {
  kind: NoticeKind;
  text: string;
  path?: string;
}

export interface TimerView {
  value: number;
  running: boolean;
  failed: boolean;
  label: string;
  sub: string;
}

/** One line of the conversation. Images are object URLs, for display only. */
export interface ChatMessage {
  id: string;
  role: 'you' | 'agent' | 'alert';
  text: string;
  images?: string[];
  seconds?: number;
}

export interface RunView {
  badge: { kind: BadgeKind; text: string };
  messages: ChatMessage[];
  program: Program | null;
  notices: Notice[];
  /** Stages reached, in arrival order. Includes failure stages the strip appends. */
  reached: string[];
  timer: TimerView;
  busy: boolean;
}

interface Pending {
  id: string | null;
  text: string;
  t0: number;
  settled: boolean;
  compiled: boolean;
  compileSeconds: number;
  generation: number;
  acquireTimer: ReturnType<typeof setTimeout> | null;
}

export interface RunDeps {
  settings: Settings;
  compiler: Compiler;
  perception: PerceptionClient;
  /** Kinds this pipeline reports as built, from GET /behaviors. */
  availableKinds: string[];
  speak: (text: string) => void;
  pushHistory: (entry: HistoryEntry) => void;
}

const IDLE_TIMER: TimerView = {
  value: 0, running: false, failed: false, label: 'compile time', sub: '',
};

export function useRetaskRun(deps: RunDeps) {
  const [view, setView] = useState<RunView>({
    badge: { kind: 'idle', text: 'IDLE' },
    messages: [],
    program: null,
    notices: [],
    reached: [],
    timer: IDLE_TIMER,
    busy: false,
  });

  // Everything the callbacks need, read fresh without re-creating them.
  const depsRef = useRef(deps);
  depsRef.current = deps;

  const pending = useRef<Pending | null>(null);
  const generation = useRef(0);
  const ticker = useRef<ReturnType<typeof setInterval> | null>(null);
  // Mirrors view.program. Recording a run must not happen inside a setState
  // updater — those are re-run on a StrictMode double render, and history
  // would gain a duplicate entry every time.
  const programRef = useRef<Program | null>(null);

  const patch = useCallback((p: Partial<RunView>) => {
    setView((prev) => ({ ...prev, ...p }));
  }, []);

  const patchTimer = useCallback((p: Partial<TimerView>) => {
    setView((prev) => ({ ...prev, timer: { ...prev.timer, ...p } }));
  }, []);

  const say = useCallback((message: Omit<ChatMessage, 'id'>) => {
    setView((prev) => ({
      ...prev,
      messages: [...prev.messages, { ...message, id: `m${prev.messages.length + 1}` }],
    }));
  }, []);

  const addNotice = useCallback((notice: Notice) => {
    setView((prev) => ({ ...prev, notices: [...prev.notices, notice] }));
  }, []);

  const markStage = useCallback((stage: string) => {
    setView((prev) => (prev.reached.includes(stage)
      ? prev
      : { ...prev, reached: [...prev.reached, stage] }));
  }, []);

  const stopTicker = useCallback(() => {
    if (ticker.current) clearInterval(ticker.current);
    ticker.current = null;
  }, []);

  const elapsed = () => (pending.current ? (performance.now() - pending.current.t0) / 1000 : 0);

  useEffect(() => () => {
    stopTicker();
    if (pending.current?.acquireTimer) clearTimeout(pending.current.acquireTimer);
  }, [stopTicker]);

  /* ------------------------------------------------------------ settling */

  /** Stop the clock and record the run. */
  const settle = useCallback((value: number, ok: boolean, label: string) => {
    const p = pending.current;
    if (!p || p.settled) return;
    p.settled = true;
    if (p.acquireTimer) clearTimeout(p.acquireTimer);
    stopTicker();

    depsRef.current.pushHistory({
      text: p.text,
      seconds: value,
      compileSeconds: p.compileSeconds,
      program: programRef.current,
      ok,
      acquired: label === 'retask time',
    });

    setView((prev) => ({
      ...prev,
      timer: { ...prev.timer, value, running: false, failed: !ok, label },
    }));
  }, [stopTicker]);

  const fail = useCallback((detail: string) => {
    const p = pending.current;
    if (!p || p.settled) return;
    const value = elapsed();
    p.settled = true;
    if (p.acquireTimer) clearTimeout(p.acquireTimer);
    stopTicker();

    depsRef.current.pushHistory({
      text: p.text,
      seconds: value,
      program: p.compiled ? programRef.current : null,
      ok: false,
    });

    setView((prev) => ({
      ...prev,
      badge: { kind: 'fail', text: 'ERROR' },
      notices: [...prev.notices, { kind: 'error', text: detail || 'compile failed' }],
      timer: { ...prev.timer, value, running: false, failed: true, label: 'failed' },
    }));
    depsRef.current.speak('Failed.');
  }, [stopTicker]);

  const clarify = useCallback((question?: string) => {
    const p = pending.current;
    if (!p) return;
    p.settled = true;
    stopTicker();
    const text = question || 'The orchestrator needs more detail.';
    setView((prev) => ({
      ...prev,
      badge: { kind: 'warn', text: 'CLARIFY' },
      notices: [{ kind: 'ask', text }],
      timer: { ...prev.timer, running: false, label: 'needs clarification' },
    }));
    depsRef.current.speak(question || 'Which one did you mean?');
  }, [stopTicker]);

  /* ----------------------------------------------------------- dispatch */

  /**
   * In live mode the orchestrator installs the behaviours itself, as part of
   * its agent loop. Only the offline mock needs this side to post.
   */
  const dispatchEnabled = () => depsRef.current.settings.mode === 'mock';

  /**
   * Install the program, replacing whatever was running.
   *
   * An instruction means "this is the whole objective now", so the pipeline is
   * cleared first. `applyProgram` does both halves in order, because two
   * behaviours from one instruction must never be half-installed.
   */
  const dispatch = useCallback(async (program: Program) => {
    const { perception } = depsRef.current;
    await perception.applyProgram(program.behaviors);
    // Best effort: the HUD is decoration, and a pipeline without it is fine.
    perception.setHud(summarize(program)).catch(() => {});
  }, []);

  /* ------------------------------------------------------------ delivery */

  const deliverProgram = useCallback((program: Program) => {
    const p = pending.current;
    if (!p || p.settled || p.compiled) return;
    p.compiled = true;
    p.compileSeconds = elapsed();

    const { availableKinds, speak } = depsRef.current;
    const result = validateProgram(program);
    const missing = unavailableKinds(program, availableKinds);

    markStage('compiled');
    programRef.current = program;
    setView((prev) => ({
      ...prev,
      program,
      notices: [
        ...[...result.errors, ...missing].map((e): Notice =>
          ({ kind: 'error', text: e.msg, path: e.path })),
        ...result.warnings.map((w): Notice => ({ kind: 'warn', text: w.msg, path: w.path })),
      ],
      timer: { ...prev.timer, sub: `compiled in ${fmt(p.compileSeconds)}s` },
    }));

    if (!result.ok || missing.length) {
      // Never post a program we already know the pipeline will 422.
      settle(p.compileSeconds, false, 'invalid');
      patch({ badge: { kind: 'fail', text: 'INVALID' } });
      const n = result.errors.length + missing.length;
      speak(`Rejected. ${n} contract ${n === 1 ? 'error' : 'errors'}.`);
      return;
    }

    patch({
      badge: result.warnings.length
        ? { kind: 'warn', text: `VALID · ${result.warnings.length} WARN` }
        : { kind: 'pass', text: 'VALID' },
    });
    speak(summarize(program));

    if (!dispatchEnabled()) {
      settle(p.compileSeconds, true, 'compile time');
      return;
    }

    // Keep the clock running: the headline number is instruction -> acquired.
    patchTimer({ label: 'applying…' });

    dispatch(program).then(() => {
      markStage('applied');

      // Only `track` and `watch` report reaching their subject. A highlight or
      // a privacy blur is doing its job the moment it is installed, so the run
      // is finished rather than left waiting for an event that never comes.
      if (!program.behaviors.some((b) => b.kind === 'track' || b.kind === 'watch')) {
        markStage('active');
        settle(elapsed(), true, 'retask time');
        patch({ badge: { kind: 'pass', text: 'RUNNING' } });
        return;
      }

      patchTimer({ label: 'acquiring…' });
      p.acquireTimer = setTimeout(() => {
        if (pending.current === p && !p.settled) {
          settle(p.compileSeconds, false, 'applied (never acquired)');
          addNotice({
            kind: 'warn',
            text: `no acquire within ${ACQUIRE_TIMEOUT_MS / 1000}s — the pipeline took the `
              + 'objective but never matched anything. Check the match count, not the state: '
              + 'a behaviour can be ACTIVE with zero matches. Try a more specific phrase.',
          });
        }
      }, ACQUIRE_TIMEOUT_MS);
    }).catch((e) => fail(friendlyError(e, 'the pipeline')));
  }, [addNotice, dispatch, fail, markStage, patch, patchTimer, settle]);

  /**
   * Live mode. The agent installed whatever it decided on, so there is no
   * program to validate here — its reply is the result, and the trace panel
   * carries the reasoning. The clock stops on the reply, because that is when
   * the operator learns what happened.
   */
  const deliverReply = useCallback((reply: string) => {
    const p = pending.current;
    if (!p || p.settled) return;
    p.compiled = true;
    p.compileSeconds = elapsed();

    markStage('compiled');
    markStage('applied');
    markStage('active');
    say({ role: 'agent', text: reply, seconds: p.compileSeconds });
    patch({ badge: { kind: 'pass', text: 'DONE' } });
    settle(p.compileSeconds, true, 'retask time');
    depsRef.current.speak(reply);
  }, [markStage, patch, say, settle]);

  /* -------------------------------------------------------------- events */

  const handleStage = useCallback((ev: StageEvent) => {
    const p = pending.current;
    if (!ev?.stage || !p) return;
    // Sockets replay a recent backlog on connect, so events from a run that
    // predates this page load must not light up the strip.
    if (p.id && ev.instruction_id && ev.instruction_id !== p.id) return;

    markStage(ev.stage);

    if (ev.stage === 'compiled') {
      const data = ev.data as Program | undefined;
      if (data && Array.isArray(data.behaviors)) deliverProgram(data);
      return;
    }

    if (ev.stage === 'clarify') {
      clarify(ev.detail);
      return;
    }

    if (ev.stage === 'active') {
      // The retask metric. `no_target` can arrive first and settle the run —
      // acquiring later is still the better outcome, so let it overwrite the
      // timer, not just the badge, or the two end up disagreeing.
      p.settled = false;
      settle(elapsed(), true, 'retask time');
      patch({ badge: { kind: 'pass', text: 'TRACKING' } });
      depsRef.current.speak('Target acquired.');
      return;
    }

    if (ev.stage === 'no_target') {
      // Not a failure: the objective is live, nothing matching is in frame yet.
      addNotice({
        kind: 'warn',
        text: ev.detail || 'nothing matching in frame — the objective is loaded and still looking',
      });
      if (!p.settled) settle(p.compileSeconds || elapsed(), false, 'no target');
      patch({ badge: { kind: 'warn', text: 'NO TARGET' } });
      return;
    }

    if (BAD_STAGES.includes(ev.stage)) {
      fail(ev.detail || `pipeline stage: ${ev.stage}`);
    }
  }, [addNotice, clarify, deliverProgram, fail, markStage, patch, settle]);

  /* ------------------------------------------------------------- compile */

  const compile = useCallback((raw: string, images: File[] = []) => {
    const text = String(raw ?? '').trim();
    if (!text && !images.length) return;

    const p: Pending = {
      id: null,
      text,
      t0: performance.now(),
      settled: false,
      compiled: false,
      compileSeconds: 0,
      generation: ++generation.current,
      acquireTimer: null,
    };
    if (pending.current?.acquireTimer) clearTimeout(pending.current.acquireTimer);
    pending.current = p;
    programRef.current = null;

    setView((prev) => ({
      badge: { kind: 'pending', text: 'THINKING' },
      messages: [...prev.messages, {
        id: `m${prev.messages.length + 1}`,
        role: 'you',
        text: text || `${images.length} image${images.length === 1 ? '' : 's'}`,
        images: images.map((f) => URL.createObjectURL(f)),
      }],
      program: null,
      notices: [],
      reached: ['received'],
      timer: { value: 0, running: true, failed: false, label: 'thinking…', sub: '' },
      busy: true,
    }));

    stopTicker();
    ticker.current = setInterval(() => {
      patchTimer({ value: (performance.now() - p.t0) / 1000 });
    }, 40);

    const { compiler } = depsRef.current;

    compiler.compile(text, (ev) => {
      if (p.generation === generation.current) handleStage(ev);
    }, images).then((res) => {
      if (pending.current !== p || p.settled) return;
      if (res.instruction_id) p.id = res.instruction_id;

      if (res.program) {
        // Mock mode: this side installs the behaviours.
        deliverProgram(res.program);
      } else if (res.reply) {
        // Live mode: the agent already installed what it wanted. The reply is
        // the turn's result, and the trace shows how it got there.
        deliverReply(res.reply);
      } else {
        // Accepted, with the program arriving later over the status socket.
        patch({ badge: { kind: 'pending', text: 'AWAITING SPEC' } });
      }
    }).catch((e: unknown) => {
      fail(e instanceof Error ? e.message : String(e));
    }).finally(() => {
      setView((prev) => ({ ...prev, busy: false }));
    });
  }, [deliverProgram, deliverReply, fail, handleStage, patch, patchTimer, stopTicker]);

  /* --------------------------------------------------------------- stop */

  /** DELETE /behaviors — everything stops. This replaced E-STOP when the car went away. */
  const stop = useCallback(async () => {
    const { perception, speak } = depsRef.current;
    if (pending.current && !pending.current.settled) settle(elapsed(), false, 'stopped');

    try {
      await perception.clearBehaviors();
      perception.setHud('').catch(() => {});
      programRef.current = null;
      setView((prev) => ({
        ...prev,
        program: null,
        badge: { kind: 'idle', text: 'IDLE' },
        notices: [...prev.notices, { kind: 'warn', text: 'pipeline cleared — back to IDLE' }],
      }));
      speak('Stopped.');
    } catch (e) {
      addNotice({
        kind: 'error',
        text: `could not clear the pipeline: ${friendlyError(e, 'the pipeline')}`,
      });
    }
  }, [addNotice, settle]);

  /** An alert the agent spoke between turns, off WS /ws/trace. */
  const alert = useCallback((text: string) => {
    say({ role: 'alert', text });
    depsRef.current.speak(text);
  }, [say]);

  /** Put an old run back on screen without re-running it. */
  const recall = useCallback((entry: HistoryEntry) => {
    if (!entry.program) return;
    pending.current = null;
    programRef.current = entry.program;
    stopTicker();
    const result = validateProgram(entry.program);
    setView((prev) => ({
      ...prev,
      badge: { kind: result.ok ? 'pass' : 'fail', text: result.ok ? 'VALID' : 'INVALID' },
      program: entry.program,
      notices: [],
      reached: [],
      timer: {
        value: entry.seconds, running: false, failed: !entry.ok, label: 'recalled', sub: '',
      },
      busy: false,
    }));
  }, [stopTicker]);

  return { view, compile, stop, recall, handleStage, alert };
}
