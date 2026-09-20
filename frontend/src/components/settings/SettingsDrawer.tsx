/**
 * Preferences, not configuration.
 *
 * Service URLs come from `.env.local` at build time and are shown here
 * read-only. A URL typed into a form at demo time is a demo that breaks in a
 * way nobody in the room can see, and there is no longer any secret to paste:
 * the LLM and its key belong to the orchestrator.
 */

import { useEffect, useState } from 'react';
import type { CompilerMode, Preferences, Settings } from '../../config/settings';
import type { ModelEntry } from '../../contracts/behavior';
import { CheckField, SelectField, TextField } from './Field';

export interface SettingsDrawerProps {
  open: boolean;
  settings: Settings;
  models: ModelEntry[];
  onSave: (patch: Partial<Preferences>) => void;
  /** POST /model — the detector is pipeline state, not a field on a spec. */
  onSelectModel: (name: string) => void;
  onClose: () => void;
}

export function SettingsDrawer(
  { open, settings, models, onSave, onSelectModel, onClose }: SettingsDrawerProps,
) {
  const [draft, setDraft] = useState<Preferences>(settings);

  useEffect(() => {
    if (open) setDraft(settings);
  }, [open, settings]);

  if (!open) return null;

  const set = <K extends keyof Preferences>(key: K, value: Preferences[K]) =>
    setDraft((d) => ({ ...d, [key]: value }));

  const modelOptions = models.length
    ? models.map((m) => ({
      value: m.name,
      label: [
        m.name,
        m.open_vocab ? 'open vocab' : m.classes ? `${m.classes.length} classes` : null,
        m.available === false ? 'UNAVAILABLE' : m.loaded ? 'loaded' : null,
      ].filter(Boolean).join(' · '),
      disabled: m.available === false,
    }))
    : [{ value: draft.detector, label: `${draft.detector} (registry unreachable)` }];

  const save = () => {
    onSave({ ...draft, lang: draft.lang.trim() || 'en-US' });
    // A BehaviorSpec carries no model, so switching detector is its own call.
    // Behaviours the new model cannot serve are paused with a reason, not lost.
    if (draft.detector !== settings.detector) onSelectModel(draft.detector);
    onClose();
  };

  return (
    <>
      <button type="button" className="drawer-scrim" aria-label="Close settings" onClick={onClose} />
      <aside className="drawer" aria-label="Settings">
        <h2 className="panel-title">Settings</h2>

        <SelectField<CompilerMode>
          label="Compiler"
          value={draft.mode}
          onChange={(v) => set('mode', v)}
          options={[
            { value: 'live', label: 'Orchestrator — the agent layer' },
            { value: 'mock', label: 'Mock — offline keyword rules' },
          ]}
          hint="Mock needs no network at all. It is the fallback if the venue WiFi
                or the orchestrator's laptop dies mid-demo."
        />

        <SelectField
          label="Detector model"
          value={draft.detector}
          onChange={(v) => set('detector', v)}
          options={modelOptions}
          hint={
            <>
              From <code>GET /models</code>; applied with <code>POST /model</code> on save.
              Behaviours the new model cannot serve are paused with a reason and resume
              on their own.
            </>
          }
        />

        <TextField
          label="Speech language"
          value={draft.lang}
          onChange={(v) => set('lang', v)}
          placeholder="en-US"
        />

        <CheckField
          label="Speak confirmations aloud"
          checked={draft.tts}
          onChange={(v) => set('tts', v)}
        />

        <CheckField
          label="Send automatically after 1.5 s of silence"
          checked={draft.autoSend}
          onChange={(v) => set('autoSend', v)}
        />

        <div className="actions">
          <button type="button" className="btn primary" onClick={save}>Save</button>
          <button type="button" className="btn ghost" onClick={onClose}>Close</button>
        </div>

        <div className="field">
          <span>Services</span>
          <ul className="endpoints">
            <li><code>{settings.apiBase}</code><em>orchestrator</em></li>
            <li><code>{settings.pipelineBase}</code><em>perception</em></li>
            <li><code>{settings.videoUrl}</code><em>video</em></li>
          </ul>
        </div>
      </aside>
    </>
  );
}
