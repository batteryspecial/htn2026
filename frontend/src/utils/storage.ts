/** localStorage that never throws. Private mode and a full quota both fail soft. */

export function readJson<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    if (!raw) return fallback;
    return (JSON.parse(raw) as T) ?? fallback;
  } catch {
    return fallback;
  }
}

export function writeJson(key: string, value: unknown): void {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* quota or private mode — losing a preference is not worth an error */
  }
}
