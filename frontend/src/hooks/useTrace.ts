/**
 * The agent's work, as it happens.
 *
 * `WS /ws/trace` on the orchestrator. An agent loop that shows its tool calls
 * is debuggable on stage; one that does not is a black box that either works
 * or does not.
 *
 * Two things fall out of the same stream, so they are read here together:
 * `say` entries are alerts the console speaks, and `reply` entries close a
 * turn. Everything else is for the panel.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import type { TraceEntry } from '../services/orchestrator';
import type { OrchestratorClient } from '../services/orchestrator';
import { ReconnectingSocket } from '../services/socket';

export interface TraceFeed {
  entries: TraceEntry[];
  connected: boolean;
  clear: () => void;
}

export interface TraceOptions {
  /** Spoken aloud and shown as an alert. Fires for `say` only. */
  onSay?: (text: string) => void;
  onEntry?: (entry: TraceEntry) => void;
  limit?: number;
}

export function useTrace(
  orchestrator: OrchestratorClient,
  apiBase: string,
  enabled: boolean,
  options: TraceOptions = {},
): TraceFeed {
  const [entries, setEntries] = useState<TraceEntry[]>([]);
  const [connected, setConnected] = useState(false);

  // Read through a ref so a new callback identity never reopens the socket.
  const optionsRef = useRef(options);
  const seen = useRef(new Set<string>());
  optionsRef.current = options;

  useEffect(() => {
    if (!enabled) {
      setConnected(false);
      return;
    }

    const limit = optionsRef.current.limit ?? 200;
    const openedAt = Date.now() / 1000;
    const socket = new ReconnectingSocket<TraceEntry>(orchestrator.traceSocketUrl(), {
      onOpen: () => setConnected(true),
      onClose: () => setConnected(false),
      onMessage: (entry) => {
        if (!entry?.kind) return;
        const key = `${entry.ts}:${entry.id}`;
        if (seen.current.has(key)) return;
        seen.current.add(key);
        if (seen.current.size > 2000) {
          seen.current = new Set([...seen.current].slice(-1000));
        }
        setEntries((prev) => [...prev, entry].slice(-limit));
        if (entry.kind === 'say' && entry.ts >= openedAt) {
          optionsRef.current.onSay?.(entry.label);
        }
        optionsRef.current.onEntry?.(entry);
      },
    });

    socket.connect();
    return () => socket.close();
  }, [orchestrator, apiBase, enabled]);

  const clear = useCallback(() => setEntries([]), []);
  return { entries, connected, clear };
}
