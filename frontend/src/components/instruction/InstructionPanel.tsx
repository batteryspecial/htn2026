import { useCallback, useRef } from 'react';
import { EXAMPLE_INSTRUCTIONS } from '../../config/constants';
import type { SpeechControls } from '../../hooks/useSpeechRecognition';
import { Panel } from '../common/Panel';
import { MicButton } from './MicButton';

export interface InstructionPanelProps {
  text: string;
  onTextChange: (text: string) => void;
  /** True while dictation is still revising the last words. */
  interim: boolean;
  busy: boolean;
  /** Owned by App so Ctrl+M and Esc can reach the same instance. */
  speech: SpeechControls;
  onCompile: (text: string) => void;
  onStop: () => void;
}

export function InstructionPanel(props: InstructionPanelProps) {
  const { text, onTextChange, busy, speech, onCompile } = props;
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const submit = useCallback((value: string) => {
    if (speech.listening) speech.stop();
    if (!value.trim()) {
      inputRef.current?.focus();
      return;
    }
    onCompile(value);
  }, [onCompile, speech]);

  return (
    <Panel className="panel-input" title="Instruction">
      <MicButton
        supported={speech.supported}
        listening={speech.listening}
        status={speech.status}
        error={speech.error}
        onToggle={speech.toggle}
      />

      <textarea
        ref={inputRef}
        className={`instruction-input ${props.interim ? 'interim' : ''}`}
        rows={2}
        spellCheck={false}
        placeholder="follow the person in red shoes"
        value={text}
        onChange={(e) => onTextChange(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            submit(text);
          }
        }}
      />

      <div className="chips">
        {EXAMPLE_INSTRUCTIONS.map((example) => (
          <button
            key={example}
            type="button"
            className="chip"
            onClick={() => { onTextChange(example); submit(example); }}
          >
            {example}
          </button>
        ))}
      </div>

      <div className="actions">
        <button
          type="button"
          className={`btn primary ${busy ? 'busy' : ''}`}
          onClick={() => submit(text)}
        >
          COMPILE
        </button>
        <button
          type="button"
          className="btn stop"
          onClick={props.onStop}
          title="DELETE /behaviors — every behaviour stops"
        >
          STOP
        </button>
        <button
          type="button"
          className="btn ghost"
          onClick={() => { onTextChange(''); inputRef.current?.focus(); }}
        >
          Clear
        </button>
      </div>

      <p className="hint">
        <kbd>Enter</kbd> compile &nbsp;·&nbsp; <kbd>Shift</kbd>+<kbd>Enter</kbd> newline
        &nbsp;·&nbsp; <kbd>Ctrl</kbd>+<kbd>M</kbd> mic &nbsp;·&nbsp; <kbd>Esc</kbd> cancel
      </p>
    </Panel>
  );
}
