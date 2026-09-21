/**
 * Browser speech-to-text.
 *
 * Chrome and Edge only — Firefox and most of Safari have no implementation, so
 * `supported` is false there and typing still works. Chrome also uploads the
 * audio to Google, which means the mic needs internet even though the rest of
 * this app is local.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

const ERROR_MESSAGES: Record<string, string> = {
  'not-allowed': 'Microphone blocked. Allow it in the address bar, then try again.',
  'service-not-allowed': 'Microphone blocked by policy. Type instead.',
  'no-speech': 'Heard nothing. Try again, closer to the mic.',
  'audio-capture': 'No microphone found.',
  network: 'Speech service unreachable — Chrome sends audio to Google, so this needs internet.',
};

const SpeechRecognitionImpl: SpeechRecognitionCtor | undefined =
  typeof window !== 'undefined'
    ? window.SpeechRecognition ?? window.webkitSpeechRecognition
    : undefined;

export interface SpeechOptions {
  lang: string;
  /** Whatever is in the textarea now — dictation appends to it. */
  currentText: () => string;
  /** Called with the transcript so far. `interim` means it may still change. */
  onTranscript: (text: string, interim: boolean) => void;
  /** Fires after a pause once the transcript has settled, when auto-send is on. */
  onSilence?: (text: string) => void;
  autoSendMs?: number;
}

export interface SpeechControls {
  supported: boolean;
  listening: boolean;
  status: string;
  error: boolean;
  start: () => void;
  stop: () => void;
  toggle: () => void;
}

const IDLE_HINT = 'Click to speak, or just type below.';

export function useSpeechRecognition(opts: SpeechOptions): SpeechControls {
  const supported = !!SpeechRecognitionImpl;
  const [listening, setListening] = useState(false);
  const [status, setStatus] = useState(
    supported ? IDLE_HINT : 'Speech recognition needs Chrome or Edge. Typing works everywhere.',
  );
  const [error, setError] = useState(!supported);

  const recognition = useRef<SpeechRecognition | null>(null);
  const finalBuffer = useRef('');
  const latest = useRef('');
  const silenceTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Handlers are read through a ref so restarting recognition is never needed
  // just because a parent re-rendered.
  const optsRef = useRef(opts);
  optsRef.current = opts;

  const stop = useCallback(() => {
    const rec = recognition.current;
    if (!rec) return;
    setListening(false);
    try { rec.stop(); } catch { /* already stopped */ }
  }, []);

  const start = useCallback(() => {
    if (!SpeechRecognitionImpl || recognition.current) return;

    // Don't transcribe our own confirmations.
    window.speechSynthesis?.cancel();

    const rec = new SpeechRecognitionImpl();
    rec.continuous = true;
    rec.interimResults = true;
    rec.lang = optsRef.current.lang;
    rec.maxAlternatives = 1;

    latest.current = optsRef.current.currentText();
    finalBuffer.current = latest.current.trim();
    if (finalBuffer.current) finalBuffer.current += ' ';

    rec.onstart = () => {
      setListening(true);
      setError(false);
      setStatus('Listening… click again to stop.');
    };

    rec.onresult = (event) => {
      let interim = '';
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const chunk = event.results[i][0].transcript;
        if (event.results[i].isFinal) finalBuffer.current += `${chunk.trim()} `;
        else interim += chunk;
      }

      const text = `${finalBuffer.current}${interim}`.replace(/\s+/g, ' ').trim();
      latest.current = text;
      optsRef.current.onTranscript(text, interim.length > 0);

      const { onSilence, autoSendMs = 1500 } = optsRef.current;
      if (onSilence) {
        if (silenceTimer.current) clearTimeout(silenceTimer.current);
        silenceTimer.current = setTimeout(() => {
          if (latest.current.trim()) {
            stop();
            onSilence(latest.current);
          }
        }, autoSendMs);
      }
    };

    rec.onerror = (event) => {
      if (event.error === 'aborted') return; // our own stop() call
      setListening(false);
      setError(true);
      setStatus(ERROR_MESSAGES[event.error] ?? `Speech error: ${event.error}`);
    };

    rec.onend = () => {
      if (silenceTimer.current) clearTimeout(silenceTimer.current);
      recognition.current = null;
      setListening(false);
      setStatus((prev) =>
        prev.startsWith('Listening')
          ? (latest.current.trim() ? 'Got it — press Enter to compile.' : IDLE_HINT)
          : prev);
    };

    recognition.current = rec;
    try {
      rec.start();
    } catch (e) {
      recognition.current = null;
      setError(true);
      setStatus(`Could not start the mic: ${(e as Error).message}`);
    }
  }, [stop]);

  const toggle = useCallback(() => {
    if (recognition.current) stop(); else start();
  }, [start, stop]);

  useEffect(() => () => {
    if (silenceTimer.current) clearTimeout(silenceTimer.current);
    try { recognition.current?.abort(); } catch { /* nothing to abort */ }
  }, []);

  // A stable object, so anything keyed on these controls (the hotkey binding,
  // the submit callback) is not rebuilt on every parent render.
  return useMemo(
    () => ({ supported, listening, status, error, start, stop, toggle }),
    [supported, listening, status, error, start, stop, toggle],
  );
}

/* ------------------------------------------------------------ dom typings */

type SpeechRecognitionCtor = new () => SpeechRecognition;

interface SpeechRecognition extends EventTarget {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  maxAlternatives: number;
  start(): void;
  stop(): void;
  abort(): void;
  onstart: (() => void) | null;
  onresult: ((event: SpeechRecognitionEvent) => void) | null;
  onerror: ((event: { error: string }) => void) | null;
  onend: (() => void) | null;
}

interface SpeechRecognitionEvent {
  resultIndex: number;
  results: ArrayLike<ArrayLike<{ transcript: string }> & { isFinal: boolean }>;
}

declare global {
  interface Window {
    SpeechRecognition?: SpeechRecognitionCtor;
    webkitSpeechRecognition?: SpeechRecognitionCtor;
  }
}
