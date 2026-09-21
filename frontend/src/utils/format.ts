/** Small shared formatters. */

export const seconds = (s: number): string => s.toFixed(2);

export const percent = (fraction: number): string => `${(fraction * 100).toFixed(1)}%`;

export const clockTime = (ts: number): string =>
  new Date(ts * 1000).toLocaleTimeString([], { hour12: false });

export const plural = (n: number, word: string): string =>
  `${n} ${word}${n === 1 ? '' : 's'}`;

/** Join the parts that exist with the separator the console uses everywhere. */
export const joinParts = (parts: (string | null | undefined | false)[]): string =>
  parts.filter(Boolean).join('  ·  ');
