/** The last few runs, and how long each took. Survives a reload. */

import { useCallback, useState } from 'react';
import { HISTORY_KEY, HISTORY_MAX } from '../config/constants';
import type { Program } from '../contracts/program';
import { readJson, writeJson } from '../utils/storage';

export interface HistoryEntry {
  text: string;
  /** Whatever number the run earned: retask time if acquired, compile time otherwise. */
  seconds: number;
  compileSeconds?: number;
  program: Program | null;
  ok: boolean;
  acquired?: boolean;
}

export function useHistory() {
  const [entries, setEntries] = useState<HistoryEntry[]>(
    () => readJson<HistoryEntry[]>(HISTORY_KEY, []),
  );

  const push = useCallback((entry: HistoryEntry) => {
    setEntries((prev) => {
      const next = [entry, ...prev].slice(0, HISTORY_MAX);
      writeJson(HISTORY_KEY, next);
      return next;
    });
  }, []);

  const clear = useCallback(() => {
    setEntries([]);
    writeJson(HISTORY_KEY, []);
  }, []);

  return { entries, push, clear };
}
