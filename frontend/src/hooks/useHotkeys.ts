/**
 * Global keys.
 *
 * SPACEBAR is deliberately unbound. It was reserved for E-STOP; the car is out
 * of scope, so it is free — but leave it free unless something genuinely needs
 * a panic key, because space is what people hit when they mean "stop".
 */

import { useEffect } from 'react';

export interface Hotkeys {
  /** Esc — cancel listening, close the drawer. */
  onEscape: () => void;
  /** Ctrl/Cmd+M — start or stop the mic. */
  onToggleMic: () => void;
  /** V, when not typing — toggle spoken confirmations. */
  onToggleVoice: () => void;
}

const TYPING = /^(INPUT|TEXTAREA|SELECT)$/;

export function useHotkeys(keys: Hotkeys): void {
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const typing = TYPING.test(document.activeElement?.tagName ?? '');

      if (e.key === 'Escape') {
        keys.onEscape();
        return;
      }
      if ((e.ctrlKey || e.metaKey) && (e.key === 'm' || e.key === 'M')) {
        e.preventDefault();
        keys.onToggleMic();
        return;
      }
      if (!typing && (e.key === 'v' || e.key === 'V')) {
        keys.onToggleVoice();
      }
    };

    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, [keys]);
}
