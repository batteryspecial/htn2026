import type { CompilerMode } from '../../config/settings';
import type { Reachability } from '../../services/perception';

const MODE_ORDER: CompilerMode[] = ['openai', 'live', 'mock'];
const MODE_LABEL: Record<CompilerMode, string> = {
  openai: 'OPENAI',
  live: 'LIVE',
  mock: 'MOCK',
};

export interface TopBarProps {
  mode: CompilerMode;
  /** What the compiler dot is pointing at: a model name, a host, or the mock. */
  compilerLabel: string;
  compiler: Reachability | null;
  camera: { live: boolean; note: string };
  tts: boolean;
  onCycleMode: () => void;
  onToggleTts: () => void;
  onOpenSettings: () => void;
}

export function TopBar(props: TopBarProps) {
  const { mode, compilerLabel, compiler, camera, tts } = props;

  const compilerDot = mode === 'mock'
    ? 'mock'
    : compiler?.up ? 'up' : compiler ? 'down' : '';

  return (
    <header className="topbar">
      <div className="brand">
        <span className="brand-mark" />
        <span className="brand-text">Retask Console</span>
      </div>

      <div className="topbar-right">
        <button
          type="button"
          className={`pill ${mode === 'live' ? 'live' : ''} ${mode === 'openai' ? 'openai' : ''}`}
          onClick={props.onCycleMode}
          title="Switch between the orchestrator, the in-browser compiler and the offline mock"
        >
          {MODE_LABEL[mode]}
        </button>

        <div className="health" title={compiler?.detail ?? 'compiler reachability'}>
          <span className={`hdot ${compilerDot}`} />
          <span>{compilerLabel}</span>
        </div>

        <div className="health" title={camera.note}>
          <span className={`hdot ${camera.live ? 'up' : 'down'}`} />
          <span>{camera.live ? 'camera' : 'no stream'}</span>
        </div>

        <button
          type="button"
          className="icon-btn"
          aria-pressed={tts}
          onClick={props.onToggleTts}
          title="Voice feedback (V)"
        >
          {tts ? 'VOICE ON' : 'VOICE OFF'}
        </button>

        <button type="button" className="icon-btn" onClick={props.onOpenSettings} title="Settings">
          SETTINGS
        </button>
      </div>
    </header>
  );
}

export const nextMode = (mode: CompilerMode): CompilerMode =>
  MODE_ORDER[(MODE_ORDER.indexOf(mode) + 1) % MODE_ORDER.length];
