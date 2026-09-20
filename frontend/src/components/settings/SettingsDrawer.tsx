import { useEffect, useState } from 'react';
import { statusUrl, type CompilerMode, type DispatchMode, type Runtime } from '../../config/settings';
import type { ModelEntry } from '../../contracts/behavior';
import { CheckField, SelectField, TextField } from './Field';

export interface SettingsDrawerProps {
  open: boolean;
  settings: Runtime;
  models: ModelEntry[];
  onSave: (patch: Partial<Runtime>) => void;
  onClose: () => void;
}

/** The drawer edits a draft and commits on Save, so a half-typed URL never applies. */
export function SettingsDrawer({ open, settings, models, onSave, onClose }: SettingsDrawerProps) {
  const [draft, setDraft] = useState<Runtime>(settings);

  useEffect(() => {
    if (open) setDraft(settings);
  }, [open, settings]);

  if (!open) return null;

  const set = <K extends keyof Runtime>(key: K, value: Runtime[K]) =>
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
    : [{ value: draft.specModel, label: `${draft.specModel} (registry unreachable)` }];

  const save = () => {
    const chosen = models.find((m) => m.name === draft.specModel);
    onSave({
      ...draft,
      openaiModel: draft.openaiModel.trim() || 'gpt-4o-mini',
      lang: draft.lang.trim() || 'en-US',
      // The compile prompt words `detect` differently for an open-vocabulary
      // detector than a fixed one, so it has to know which is selected.
      specModelOpenVocab: chosen ? chosen.open_vocab !== false : true,
    });
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
            { value: 'openai', label: 'OpenAI — compile in the browser' },
            { value: 'live', label: 'Orchestrator — the agent layer' },
            { value: 'mock', label: 'Mock — offline keyword rules' },
          ]}
        />

        <TextField
          label="OpenAI model"
          value={draft.openaiModel}
          onChange={(v) => set('openaiModel', v)}
          placeholder="gpt-4o-mini"
          hint={
            <>
              Key status:{' '}
              <code>
                {settings.openaiKey
                  ? `loaded, ${settings.openaiKey.length} chars, ends ${settings.openaiKey.slice(-4)}`
                  : 'missing — set VITE_OPENAI_API_KEY in frontend/.env.local'}
              </code>
            </>
          }
        />

        <TextField
          label="Pipeline base URL"
          value={draft.pipelineBase}
          onChange={(v) => set('pipelineBase', v)}
          placeholder="http://localhost:8001"
          hint={
            <>
              Perception. Supplies <code>/video</code>, <code>/behaviors</code>,{' '}
              <code>/models</code>, <code>/health</code>, <code>/ws/state</code> and{' '}
              <code>/ws/events</code>.
            </>
          }
        />

        <SelectField<DispatchMode>
          label="Dispatch contract"
          value={draft.dispatch}
          onChange={(v) => set('dispatch', v)}
          options={[
            { value: 'behaviors', label: 'BehaviorSpec — POST /behaviors' },
            { value: 'legacy', label: 'TaskSpec — POST /spec (legacy shim)' },
          ]}
          hint={
            <>
              <code>/behaviors</code> is the current contract. <code>/spec</code> goes through{' '}
              <code>server/legacy.py</code>, which maps every target to a <code>track</code>{' '}
              behaviour — keep it only until that file is deleted.
            </>
          }
        />

        <SelectField
          label="Detector model"
          value={draft.specModel}
          onChange={(v) => set('specModel', v)}
          options={modelOptions}
          hint={
            <>
              From <code>GET /models</code>. The pipeline 422s an unknown one.{' '}
              <code>fake</code> runs the whole loop with no weights and no GPU.
            </>
          }
        />

        <CheckField
          label="Post compiled objectives straight to the pipeline"
          checked={draft.sendToPipeline}
          onChange={(v) => set('sendToPipeline', v)}
        />

        <TextField
          label="Video stream URL"
          value={draft.videoUrl}
          onChange={(v) => set('videoUrl', v)}
          placeholder="http://localhost:8001/video"
          hint="The annotated MJPEG. Point it at a phone or webcam stream to test the pane on its own."
        />

        <TextField
          label="API base URL"
          value={draft.apiBase}
          onChange={(v) => set('apiBase', v)}
          placeholder="http://localhost:8000"
          hint="The orchestrator. Swap in a tunnel URL later without touching code."
        />

        <TextField
          label="Status WebSocket"
          value={draft.wsUrl}
          onChange={(v) => set('wsUrl', v)}
          placeholder="(derived from API base)"
          hint={
            <>
              Leave blank to use <code>{statusUrl({ apiBase: draft.apiBase, wsUrl: '' })}</code>.
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
          label="Auto-compile after 1.5 s of silence"
          checked={draft.autoSend}
          onChange={(v) => set('autoSend', v)}
        />

        <div className="actions">
          <button type="button" className="btn primary" onClick={save}>Save</button>
          <button type="button" className="btn ghost" onClick={onClose}>Close</button>
        </div>

        <p className="hint drawer-hint">
          Settings live in this browser&apos;s localStorage. Share a preconfigured link with{' '}
          <code>?api=http://host:8000</code> or force the mock with <code>?mock=1</code>.
        </p>
      </aside>
    </>
  );
}
