import { useEffect, useRef } from 'react';
import type { TraceEntry, TraceKind } from '../../services/orchestrator';
import { clockTime } from '../../utils/format';
import { Badge } from '../common/Badge';

/** Colour per kind, reusing the badge palette so the screen reads as one system. */
const TONE: Record<TraceKind, string> = {
  user: 'pending',
  thought: 'idle',
  tool_call: 'warn',
  tool_result: 'pass',
  say: 'pending',
  reply: 'pass',
  event: 'warn',
  error: 'fail',
  timer: 'idle',
};

const LABEL: Record<TraceKind, string> = {
  user: 'you',
  thought: 'thinking',
  tool_call: 'call',
  tool_result: 'result',
  say: 'says',
  reply: 'reply',
  event: 'event',
  error: 'error',
  timer: 'timing',
};

export interface TraceProps {
  entries: TraceEntry[];
  connected: boolean;
  onClear: () => void;
  /** False in mock mode: there is no agent to trace. */
  live: boolean;
}

/** Controls for the title band. Rendered by the tab strip, not here. */
export function TraceAside({ connected, onClear, live }: TraceProps) {
  return (
    <>
      <Badge kind={!live ? 'idle' : connected ? 'pass' : 'fail'}>
        {!live ? 'MOCK' : connected ? 'LIVE' : 'OFFLINE'}
      </Badge>
      <button type="button" className="title-btn" onClick={onClear}>CLEAR</button>
    </>
  );
}

export function TracePanel({ entries, live }: TraceProps) {
  const box = useRef<HTMLDivElement>(null);

  // Pin to the newest entry. A trace that has to be scrolled during a demo is a
  // trace nobody reads.
  //
  // Set scrollTop on the box itself rather than calling scrollIntoView on a tail
  // element: scrollIntoView scrolls every scrollable ancestor, and the accordion
  // around this panel is one of them. An arriving entry would drag the Events and
  // Behaviours sections out of view several times a second.
  useEffect(() => {
    const el = box.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [entries.length]);

  return (
    <div className="panel-scroll" ref={box}>
      {!entries.length && (
        <p className="empty-row">
          {live
            ? 'Nothing yet. Every tool call the agent makes shows up here.'
            : 'Mock mode compiles in the browser, so there is no agent to trace.'}
        </p>
      )}

      <ol className="trace">
        {entries.map((entry) => (
          <li key={entry.id} className={`trace-${entry.kind}`}>
            <span className="row-time">{clockTime(entry.ts)}</span>
            <span className={`trace-kind ${TONE[entry.kind] ?? 'idle'}`}>
              {LABEL[entry.kind] ?? entry.kind}
            </span>
            <span className="trace-body">
              <span className="trace-label">{entry.label}</span>
              {entry.detail && <span className="trace-detail">{entry.detail}</span>}
            </span>
          </li>
        ))}
      </ol>
    </div>
  );
}
