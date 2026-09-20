import { useAudioLevel } from '../../hooks/useAudioLevel';

export function MicButton({ supported, listening, status, error, onToggle }: {
  supported: boolean;
  listening: boolean;
  status: string;
  error: boolean;
  onToggle: () => void;
}) {
  const level = useAudioLevel(listening);

  return (
    <div className="mic-row">
      <button
        type="button"
        className="mic"
        aria-pressed={listening}
        disabled={!supported}
        onClick={onToggle}
      >
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M12 15a3.5 3.5 0 0 0 3.5-3.5v-5a3.5 3.5 0 1 0-7 0v5A3.5 3.5 0 0 0 12 15z" />
          <path d="M5.5 11.5a6.5 6.5 0 0 0 13 0" fill="none" strokeWidth="2" />
          <path d="M12 18.5V22" fill="none" strokeWidth="2" />
        </svg>
        <span>{listening ? 'LISTENING' : 'SPEAK'}</span>
      </button>

      <div className="mic-side">
        <div className={`mic-status ${error ? 'error' : ''}`}>{status}</div>
        <div
          className={`level ${listening ? 'on' : ''} ${level.hot ? 'hot' : ''}`}
          aria-hidden="true"
        >
          {level.bars.map((v, i) => (
            // eslint-disable-next-line react/no-array-index-key -- fixed-length meter
            <i key={i} style={{ height: `${(3 + v * 23).toFixed(1)}px` }} />
          ))}
        </div>
      </div>
    </div>
  );
}
