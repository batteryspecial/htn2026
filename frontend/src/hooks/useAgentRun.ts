/** Live turns: /chat owns completion; correlated trace entries supply progress. */
import { useCallback, useEffect, useRef, useState } from 'react';
import type { BehaviorSpec, StateView } from '../contracts/behavior';
import {
  installationsActive, installationsApplied, programFromInstallations,
  readInstallations, turnBadge,
} from '../contracts/liveRun';
import type { Program } from '../contracts/program';
import { friendlyError } from '../services/http';
import type { OrchestratorClient, TraceEntry } from '../services/orchestrator';
import type { PerceptionClient } from '../services/perception';
import type { HistoryEntry } from './useHistory';
import type { RunView } from './useRetaskRun';

interface Dependencies {
  orchestrator: OrchestratorClient;
  perception: PerceptionClient;
  speak: (text: string) => void;
  pushHistory: (entry: HistoryEntry) => void;
  pipelineState: StateView | null;
}

interface Turn {
  id: string;
  text: string;
  started: number;
  specs: Map<string, BehaviorSpec>;
  draft: BehaviorSpec | null;
}

const initial: RunView = {
  badge: { kind: 'idle', text: 'IDLE' }, messages: [], program: null,
  notices: [], reached: [], busy: false,
  timer: { value: 0, running: false, failed: false, label: 'agent turn', sub: '' },
};

export function useAgentRun(deps: Dependencies) {
  const [view, setView] = useState<RunView>(initial);
  const current = useRef<Turn | null>(null);
  const stopping = useRef(false);
  const [installedIds, setInstalledIds] = useState<string[]>([]);
  const urls = useRef<string[]>([]);
  const depsRef = useRef(deps);
  depsRef.current = deps;

  useEffect(() => {
    if (!view.busy) return;
    const timer = setInterval(() => {
      const turn = current.current;
      if (turn) setView((prev) => ({ ...prev, timer: { ...prev.timer,
        value: (performance.now() - turn.started) / 1000 } }));
    }, 50);
    return () => clearInterval(timer);
  }, [view.busy]);

  useEffect(() => () => {
    current.current = null;
    urls.current.forEach((url) => URL.revokeObjectURL(url));
  }, []);

  const finish = useCallback((id: string, reply: string, ok: boolean, receipts?: unknown) => {
    const turn = current.current;
    if (!turn || turn.id !== id) return;
    current.current = null; // HTTP and trace can both carry the same reply.
    const seconds = (performance.now() - turn.started) / 1000;
    // The HTTP receipt also works when the trace socket disconnected, or the
    // HTTP response arrived before its tool-result messages.
    const installed = receipts === undefined
      ? [...turn.specs].map(([behaviorId, spec]) => ({ id: behaviorId, spec }))
      : readInstallations(receipts);
    const program = programFromInstallations(installed);
    setInstalledIds(installed.map((item) => item.id));
    depsRef.current.pushHistory({ text: turn.text, seconds, program, ok });
    setView((prev) => ({ ...prev, program, busy: false,
      badge: turnBadge(ok, program),
      reached: program
        ? [...new Set([...prev.reached, 'compiled'])] : ['received'],
      notices: ok ? prev.notices : [...prev.notices, { kind: 'error', text: reply }],
      messages: [...prev.messages, { id: `${id}-reply`, role: 'agent', text: reply, seconds }],
      timer: { ...prev.timer, value: seconds, running: false, failed: !ok,
        label: ok ? 'agent turn' : 'failed' },
    }));
    depsRef.current.speak(reply);
  }, []);

  useEffect(() => {
    if (!installedIds.length) return;
    const installed = deps.pipelineState?.behaviors.filter((item) => installedIds.includes(item.id)) ?? [];
    const applied = installationsApplied(installedIds, deps.pipelineState);
    const active = installationsActive(installedIds, deps.pipelineState);
    setView((prev) => {
      // A live-state update must never erase a terminal agent-turn failure.
      if (prev.timer.failed) return prev;
      const reached = [...prev.reached];
      if (applied && !reached.includes('applied')) reached.push('applied');
      if (active && !reached.includes('active')) reached.push('active');
      let badge: RunView['badge'] = active
        ? { kind: 'pass' as const, text: 'ACTIVE' }
        : applied
          ? { kind: 'pass' as const, text: 'APPLIED' }
          : { kind: 'pending' as const, text: deps.pipelineState ? 'SYNCING' : 'STATE UNKNOWN' };
      if (installed.some((item) => item.state === 'FAILED')) {
        badge = { kind: 'fail' as const, text: 'BEHAVIOR FAILED' };
      } else if (installed.some((item) => item.state === 'PAUSED')) {
        badge = { kind: 'pending' as const, text: 'PAUSED' };
      }
      return {
        ...prev,
        reached,
        badge,
      };
    });
  }, [installedIds, deps.pipelineState]);

  const compile = useCallback((raw: string, images: File[] = []) => {
    const text = raw.trim();
    if ((!text && !images.length) || current.current || stopping.current) return;
    const id = crypto.randomUUID();
    const turn: Turn = { id, text, started: performance.now(), specs: new Map(), draft: null };
    current.current = turn;
    setInstalledIds([]);
    const previews = images.map((file) => URL.createObjectURL(file));
    urls.current.push(...previews);
    setView((prev) => ({ ...initial, busy: true,
      badge: { kind: 'pending', text: 'THINKING' }, reached: ['received'],
      messages: [...prev.messages, { id, role: 'you', text: text || 'Attached image', images: previews }],
      timer: { ...initial.timer, running: true, label: 'thinking…' },
    }));
    depsRef.current.orchestrator.chat(text, images, id)
      .then((result) => finish(id, result.reply, result.ok !== false, result.behaviors))
      .catch((error: unknown) => finish(id, friendlyError(error, 'the orchestrator'), false));
  }, [finish]);

  const handleTrace = useCallback((entry: TraceEntry) => {
    const turn = current.current;
    if (!turn || entry.turn !== turn.id) return;
    if (entry.kind === 'reply') {
      finish(turn.id, entry.label, entry.data?.ok !== false, entry.data?.behaviors);
      return;
    }
    if (entry.kind === 'tool_call' && entry.label === 'start_behavior') {
      turn.draft = entry.data?.spec as BehaviorSpec ?? null;
    }
    if (entry.kind === 'tool_result' && entry.label === 'start_behavior') {
      const spec = (entry.data?.spec as BehaviorSpec | undefined) ?? turn.draft;
      const installed = readInstallations([{ id: entry.data?.behavior_id, spec }]);
      if (entry.data?.error === undefined && entry.data?.ok !== false && installed.length) {
        turn.specs.set(installed[0].id, installed[0].spec);
      }
      turn.draft = null;
    }
    if (entry.kind === 'tool_result' && entry.label === 'clear_behaviors'
      && entry.data?.error === undefined && entry.data?.ok !== false) turn.specs.clear();
    if (entry.kind === 'tool_result' && entry.label === 'stop_behavior') {
      turn.specs.delete(String(entry.data?.behavior_id));
    }
    const program: Program | null = turn.specs.size ? { behaviors: [...turn.specs.values()] } : null;
    setView((prev) => ({ ...prev, program,
      reached: program && !prev.reached.includes('compiled')
        ? [...prev.reached, 'compiled'] : prev.reached,
      timer: { ...prev.timer, sub: entry.kind === 'tool_call' ? entry.label : prev.timer.sub },
    }));
  }, [finish]);

  const stop = useCallback(async () => {
    if (stopping.current) return;
    stopping.current = true;
    const turnId = current.current?.id;
    current.current = null;
    setInstalledIds([]);
    setView((prev) => ({ ...prev, busy: true,
      badge: { kind: 'pending', text: 'STOPPING' },
      timer: { ...prev.timer, running: false, label: 'stopped' },
    }));
    window.speechSynthesis?.cancel();
    try {
      await depsRef.current.orchestrator.stop(turnId);
      setView((prev) => ({ ...prev, program: null, reached: [],
        badge: { kind: 'idle', text: 'IDLE' },
        messages: [...prev.messages, { id: crypto.randomUUID(), role: 'agent', text: 'Stopped.' }],
      }));
    } catch (error) {
      // The pipeline can still be stopped when the agent service is unreachable.
      let detail = friendlyError(error, 'the orchestrator');
      try {
        await depsRef.current.perception.clearBehaviors();
        detail += '. Pipeline cleared directly; agent cancellation could not be confirmed.';
      } catch (fallback) {
        detail += `. ${friendlyError(fallback, 'the pipeline')}`;
      }
      setView((prev) => ({ ...prev, badge: { kind: 'fail', text: 'STOP ERROR' },
        notices: [...prev.notices, { kind: 'error', text: detail }],
        messages: [...prev.messages, { id: crypto.randomUUID(), role: 'agent', text: detail }],
      }));
    } finally {
      stopping.current = false;
      setView((prev) => ({ ...prev, busy: false }));
    }
  }, []);

  const alert = useCallback((text: string) => {
    setView((prev) => ({ ...prev, messages: [...prev.messages,
      { id: crypto.randomUUID(), role: 'alert', text }] }));
    depsRef.current.speak(text);
  }, []);

  const recall = useCallback((entry: HistoryEntry) => {
    if (current.current || stopping.current) return;
    setInstalledIds([]);
    setView((prev) => ({ ...prev, program: entry.program,
      reached: [], notices: [],
      badge: { kind: entry.ok ? 'pass' : 'fail', text: 'RECALLED' },
      timer: { ...initial.timer, value: entry.seconds, label: 'recalled', failed: !entry.ok },
    }));
  }, []);

  return { view, compile, stop, recall, alert, handleTrace };
}
