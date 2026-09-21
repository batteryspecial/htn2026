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
import type { CameraEntry, ModelEntry } from '../../contracts/behavior';
import { CheckField, SelectField, TextField } from './Field';

export interface SettingsDrawerProps {
  open: boolean;
  settings: Settings;
  models: ModelEntry[];
  /** GET /cameras — enumerated by the pipeline, which is what opens them. */
  cameras: CameraEntry[];
  onSave: (patch: Partial<Preferences>) => void;
  /** POST /model — the detector is pipeline state, not a field on a spec. */
  onSelectModel: (name: string) => void;
  /** POST /camera — likewise: the camera belongs to the pipeline. */
  onSelectCamera: (name: string) => void;
  /** Re-scan for devices. Opens each idle one, so it takes a few seconds. */
  onRescanCameras: () => void;
  onClose: () => void;
}

export function SettingsDrawer(
  { open, settings, models, cameras, onSave, onSelectModel, onSelectCamera,
    onRescanCameras, onClose }: SettingsDrawerProps,
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

  // Named, never indexed: index 2 is the webcam until something is replugged
  // and then it is the built-in camera, which is a silent way to demo the
  // wrong lens. An entry the pipeline could not open is offered greyed out
  // rather than hidden, so "my webcam is not listed" and "my webcam will not
  // open" look different.
  const cameraOptions = cameras.length
    ? cameras.map((c) => ({
      value: c.name,
      label: [
        c.name,
        c.width && c.height ? `${c.width}×${c.height}` : null,
        c.available === false ? 'WILL NOT OPEN' : null,
        c.active ? 'live' : null,
      ].filter(Boolean).join(' · '),
      disabled: c.available === false,
    }))
    : [{ value: draft.camera, label: draft.camera || 'pipeline unreachable' }];

  const activeCamera = cameras.find((c) => c.active)?.name ?? '';

  const save = () => {
    onSave({ ...draft, lang: draft.lang.trim() || 'en-US' });
    // A BehaviorSpec carries no model, so switching detector is its own call.
    // Behaviours the new model cannot serve are paused with a reason, not lost.
    if (draft.detector !== settings.detector) onSelectModel(draft.detector);
    // Compared against what the pipeline reports as live, not against the
    // stored preference: the preference can be a camera from another machine,
    // or from before a reboot, and re-sending the one already running would
    // drop frames for nothing.
    if (draft.camera && draft.camera !== activeCamera) onSelectCamera(draft.camera);
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

        <SelectField
          label="Camera"
          value={draft.camera || activeCamera}
          onChange={(v) => set('camera', v)}
          options={cameraOptions}
          hint={
            <>
              From <code>GET /cameras</code> on the pipeline, applied with{' '}
              <code>POST /camera</code> on save. Enumerated where the camera is
              actually opened, not in this browser. Behaviours keep running;
              tracking resets, because track ids do not survive a change of lens.
              {' '}
              <button type="button" className="link-btn" onClick={onRescanCameras}>
                Re-scan
              </button>{' '}
              opens each idle device to read its resolution, which takes a few
              seconds.
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
