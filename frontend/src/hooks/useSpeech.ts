/** Spoken confirmations. One call, honours the VOICE toggle, never throws. */

import { useCallback, useEffect, useRef } from 'react';

export function useSpeech(enabled: boolean, lang: string) {
  const state = useRef({ enabled, lang });
  state.current = { enabled, lang };

  useEffect(() => {
    if (!enabled) window.speechSynthesis?.cancel();
  }, [enabled]);

  return useCallback((text: string) => {
    const { enabled: on, lang: language } = state.current;
    if (!on || !text || !window.speechSynthesis) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 1.05;
    utterance.lang = language;
    window.speechSynthesis.speak(utterance);
  }, []);
}
