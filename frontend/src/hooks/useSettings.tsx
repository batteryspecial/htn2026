/** Settings in one context, with the clients that depend on them. */

import {
  createContext, useCallback, useContext, useMemo, useRef, useState,
  type ReactNode,
} from 'react';
import {
  loadSettings, normalize, saveSettings, type Runtime,
} from '../config/settings';
import { createCompiler, type Compiler } from '../services/compilers';
import { OrchestratorClient } from '../services/orchestrator';
import { PerceptionClient } from '../services/perception';

interface SettingsContextValue {
  settings: Runtime;
  /** Merge a patch, persist it, and rebuild the clients. */
  update: (patch: Partial<Runtime>) => void;
  compiler: Compiler;
  perception: PerceptionClient;
  orchestrator: OrchestratorClient;
}

const SettingsContext = createContext<SettingsContextValue | null>(null);

export function SettingsProvider({ children }: { children: ReactNode }) {
  const [settings, setSettings] = useState<Runtime>(loadSettings);

  // The clients hold a base URL rather than a settings snapshot, so they are
  // reconfigured in place. Rebuilding them would drop every open socket.
  const perception = useRef(new PerceptionClient(settings.pipelineBase)).current;
  const orchestrator = useRef(
    new OrchestratorClient(settings.apiBase, settings.wsUrl, settings.requestTimeoutMs),
  ).current;

  const update = useCallback((patch: Partial<Runtime>) => {
    setSettings((prev) => {
      const next = normalize({ ...prev, ...patch });
      saveSettings(next);
      return next;
    });
  }, []);

  // The clients are retargeted as the URLs move. Doing it in a memo rather
  // than loose in the render body keeps it to once per actual change.
  useMemo(() => {
    perception.setBase(settings.pipelineBase);
  }, [perception, settings.pipelineBase]);

  useMemo(() => {
    orchestrator.configure(settings.apiBase, settings.wsUrl, settings.requestTimeoutMs);
  }, [orchestrator, settings.apiBase, settings.wsUrl, settings.requestTimeoutMs]);

  // A compiler is a thin wrapper over a settings snapshot, so rebuilding it on
  // change is correct and cheap.
  const compiler = useMemo(() => createCompiler(settings), [settings]);

  const value = useMemo(
    () => ({ settings, update, compiler, perception, orchestrator }),
    [settings, update, compiler, perception, orchestrator],
  );

  return <SettingsContext value={value}>{children}</SettingsContext>;
}

export function useSettings(): SettingsContextValue {
  const ctx = useContext(SettingsContext);
  if (!ctx) throw new Error('useSettings must be used inside <SettingsProvider>');
  return ctx;
}
