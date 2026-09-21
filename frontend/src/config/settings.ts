/**
 * Two kinds of setting, kept apart on purpose.
 *
 * **Deployment** — where the services are. These come from `.env.local` at
 * build time, or from a `?query=param` for a shared link. They are not
 * editable in the UI, because a URL typed into a form at demo time is a
 * demo that breaks in a way nobody can see.
 *
 * **Preferences** — voice, language, which compiler. These belong to the
 * person using the console, so they live in localStorage and are editable.
 *
 * No secret appears in either. The LLM and its API key are the
 * orchestrator's, in `orchestrator/.env`.
 */

import { readJson, writeJson } from '../utils/storage';

/** Where instructions are turned into behaviours. */
export type CompilerMode = 'live' | 'mock';

/** Editable, per-person, persisted in this browser. */
export interface Preferences {
  mode: CompilerMode;
  /** The active detector, from GET /models. Applied with POST /model. */
  detector: string;
  /**
   * The camera, by device *name* from GET /cameras. A name rather than an
   * index because indices shuffle on replug: unplug the webcam and index 2
   * may become the built-in camera. Empty means "whatever the pipeline booted
   * with" — this console does not force a camera on startup.
   */
  camera: string;
  lang: string;
  tts: boolean;
  autoSend: boolean;
}

/** Fixed at build time. Read-only at runtime. */
export interface Deployment {
  /** Orchestrator (:8000) — the agent layer. */
  apiBase: string;
  /** Perception (:8001) — the reflex layer. */
  pipelineBase: string;
  /** The enriched MJPEG. */
  videoUrl: string;
}

export type Settings = Preferences & Deployment;

const STORE_KEY = 'retask.prefs.v2';

export const DEFAULT_PREFERENCES: Preferences = {
  mode: 'live',
  detector: 'yoloe',
  camera: '',
  lang: 'en-US',
  tts: true,
  autoSend: false,
};

/** Frontier models stall occasionally; 15s was cutting it close. */
export const REQUEST_TIMEOUT_MS = 195000;

const trimSlash = (s: unknown) => String(s ?? '').replace(/\/+$/, '');

function deployment(search: string): Deployment {
  const env = import.meta.env;
  const q = new URLSearchParams(search);

  const apiBase = trimSlash(q.get('api') ?? env.VITE_ORCHESTRATOR_URL ?? 'http://localhost:8000');
  const pipelineBase = trimSlash(
    q.get('pipeline') ?? env.VITE_PERCEPTION_URL ?? 'http://localhost:8001',
  );
  const videoUrl = q.get('video') ?? env.VITE_VIDEO_URL ?? `${pipelineBase}/video`;

  return { apiBase, pipelineBase, videoUrl };
}

export function loadSettings(search = window.location.search): Settings {
  const stored = readJson<Partial<Preferences>>(STORE_KEY, {});
  const prefs: Preferences = { ...DEFAULT_PREFERENCES, ...stored };

  // A shared link can force the offline path without anyone editing anything.
  const q = new URLSearchParams(search);
  if (q.get('mock') === '1') prefs.mode = 'mock';
  if (q.get('live') === '1') prefs.mode = 'live';

  return { ...prefs, ...deployment(search) };
}

/** Only preferences are written back. Deployment is not ours to change. */
export function saveSettings(s: Settings): void {
  const { mode, detector, camera, lang, tts, autoSend } = s;
  writeJson(STORE_KEY, { mode, detector, camera, lang, tts, autoSend });
}

/** ws:// for a given http(s):// base. Also handles a proxied relative base. */
export function toWebSocketUrl(base: string, path: string): string {
  const absolute = base.startsWith('http')
    ? base
    : new URL(base || '/', window.location.origin).toString().replace(/\/+$/, '');
  return absolute.replace(/^http/, 'ws') + path;
}
