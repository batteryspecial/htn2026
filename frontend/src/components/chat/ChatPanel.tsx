import { useCallback, useEffect, useRef, useState, type ReactNode } from 'react';
import { EXAMPLE_INSTRUCTIONS } from '../../config/constants';
import type { ChatMessage, TimerView } from '../../hooks/useRetaskRun';
import type { SpeechControls } from '../../hooks/useSpeechRecognition';
import { Panel } from '../common/Panel';
import { MicButton } from '../instruction/MicButton';
import { RetaskTimer } from '../output/RetaskTimer';
import { Transcript } from './Transcript';

export interface ChatPanelProps {
  messages: ChatMessage[];
  text: string;
  onTextChange: (text: string) => void;
  /** True while dictation is still revising the last words. */
  interim: boolean;
  busy: boolean;
  timer: TimerView;
  speech: SpeechControls;
  /** The pipeline status strip, rendered under the title band. */
  status: ReactNode;
  onSend: (text: string, images: File[]) => void;
  onStop: () => void;
}

export function ChatPanel(props: ChatPanelProps) {
  const { text, onTextChange, busy, speech, onSend } = props;
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const [attached, setAttached] = useState<File[]>([]);
  const [previews, setPreviews] = useState<string[]>([]);

  useEffect(() => {
    const urls = attached.map((file) => URL.createObjectURL(file));
    setPreviews(urls);
    return () => urls.forEach((url) => URL.revokeObjectURL(url));
  }, [attached]);

  const submit = useCallback((value: string) => {
    if (busy) return;
    if (speech.listening) speech.stop();
    if (!value.trim() && !attached.length) {
      inputRef.current?.focus();
      return;
    }
    onSend(value, attached);
    setAttached([]);
    if (fileRef.current) fileRef.current.value = '';
  }, [attached, busy, onSend, speech]);

  return (
    <Panel
      className="panel-chat"
      title="Instruction"
      aside={<RetaskTimer timer={props.timer} compact />}
    >
      {props.status}

      <Transcript messages={props.messages} />

      <div className="composer">
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
          placeholder="watch the yellow duck and tell me if anyone gets close"
          value={text}
          onChange={(e) => onTextChange(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              submit(text);
            }
          }}
        />

        {attached.length > 0 && (
          <ul className="attachments">
            {attached.map((file, i) => (
              <li key={`${file.name}-${i}`}>
                <img src={previews[i]} alt="" />
                <span className="row-main">{file.name}</span>
                <button
                  type="button"
                  className="row-drop"
                  onClick={() => setAttached((prev) => prev.filter((_, j) => j !== i))}
                >
                  REMOVE
                </button>
              </li>
            ))}
          </ul>
        )}

        <div className="chips">
          {EXAMPLE_INSTRUCTIONS.map((example) => (
            <button
              key={example}
              type="button"
              className="chip"
              disabled={busy}
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
            disabled={busy}
            onClick={() => submit(text)}
          >
            SEND
          </button>

          <button
            type="button"
            className="btn ghost"
            onClick={() => fileRef.current?.click()}
            title="A photo becomes a reference, so you can say 'track that one'"
          >
            + IMAGE
          </button>
          <input
            ref={fileRef}
            type="file"
            accept="image/*"
            multiple
            hidden
            onChange={(e) => {
              setAttached((prev) => [...prev, ...Array.from(e.target.files ?? [])]);
            }}
          />

          <button
            type="button"
            className="btn stop"
            onClick={props.onStop}
            title="Cancel the agent turn and stop all behaviors"
          >
            STOP
          </button>
        </div>

        <p className="hint">
          <kbd>Enter</kbd> send &nbsp;·&nbsp; <kbd>Shift</kbd>+<kbd>Enter</kbd> newline
          &nbsp;·&nbsp; <kbd>Ctrl</kbd>+<kbd>M</kbd> mic &nbsp;·&nbsp; <kbd>Esc</kbd> cancel
        </p>
      </div>
    </Panel>
  );
}
