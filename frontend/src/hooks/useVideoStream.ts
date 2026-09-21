/**
 * An MJPEG `<img>` dies quietly.
 *
 * Connection refused fires `error`, which is easy. The hard case is the server
 * going away mid-stream: no error, no load, the last frame just sits there
 * looking perfectly live. Measured — 14 s after killing the stream the pane
 * still said LIVE. Chrome does not emit a load event per part of a multipart
 * stream either, so a per-frame heartbeat does not help.
 *
 * So the host itself is probed every few seconds: connection refused rejects,
 * any HTTP answer (404 included) resolves. That works for the pipeline and for
 * a bare phone IP-cam alike, because it only asks whether the host is
 * listening. Verified across a full kill/restore cycle: offline within ~9 s,
 * back to live automatically.
 */

import { useCallback, useEffect, useRef, useState } from 'react';

const INITIAL_BACKOFF_MS = 1500;
const MAX_BACKOFF_MS = 8000;
const PROBE_INTERVAL_MS = 4000;
const STALL_AFTER_MS = 5000;

export interface VideoStream {
  /** Feed this straight to `<img src>`. */
  src: string;
  live: boolean;
  /** Why it is offline, or the URL when it is live. */
  note: string;
  reconnect: () => void;
  onLoad: () => void;
  onError: () => void;
}

export function useVideoStream(url: string): VideoStream {
  const [src, setSrc] = useState('');
  const [live, setLive] = useState(false);
  const [note, setNote] = useState('');

  const backoff = useRef(INITIAL_BACKOFF_MS);
  const retry = useRef<ReturnType<typeof setTimeout> | null>(null);
  const frames = useRef(0);
  const lastFrameAt = useRef(0);
  const liveRef = useRef(false);
  liveRef.current = live;

  const attach = useCallback(() => {
    if (retry.current) clearTimeout(retry.current);
    frames.current = 0;
    setLive(false);

    if (!url) {
      setSrc('');
      setNote('no video URL set — open SETTINGS');
      return;
    }
    setNote(`connecting to ${url}`);
    setSrc(`${url}${url.includes('?') ? '&' : '?'}_t=${Date.now()}`);
  }, [url]);

  const scheduleRetry = useCallback((reason: string) => {
    setLive(false);
    setNote(reason);
    if (retry.current) clearTimeout(retry.current);
    retry.current = setTimeout(attach, backoff.current);
    backoff.current = Math.min(backoff.current * 1.4, MAX_BACKOFF_MS);
  }, [attach]);

  // A URL change is a fresh start, not a retry.
  useEffect(() => {
    backoff.current = INITIAL_BACKOFF_MS;
    attach();
    return () => { if (retry.current) clearTimeout(retry.current); };
  }, [attach]);

  const onLoad = useCallback(() => {
    frames.current += 1;
    lastFrameAt.current = Date.now();
    backoff.current = INITIAL_BACKOFF_MS;
    setLive(true);
    setNote(url);
  }, [url]);

  const onError = useCallback(() => {
    scheduleRetry(`cannot reach ${url}`);
  }, [scheduleRetry, url]);

  const reconnect = useCallback(() => {
    backoff.current = INITIAL_BACKOFF_MS;
    attach();
  }, [attach]);

  /* Watchdog for browsers that do emit a load event per MJPEG part. Chrome
     does not, so this arms rarely — it only trips after several loads, so a
     single-load stream is never falsely torn down. */
  useEffect(() => {
    const timer = setInterval(() => {
      if (frames.current > 3 && Date.now() - lastFrameAt.current > STALL_AFTER_MS) {
        scheduleRetry('stream stalled — reconnecting');
      }
    }, 2000);
    return () => clearInterval(timer);
  }, [scheduleRetry]);

  /* The probe that actually catches a dead feed. */
  useEffect(() => {
    if (!url) return;

    let origin: string;
    try {
      origin = new URL(url, window.location.href).origin;
    } catch {
      return;
    }

    const timer = setInterval(async () => {
      if (!liveRef.current) return;
      try {
        await fetch(`${origin}/`, {
          mode: 'no-cors',
          cache: 'no-store',
          signal: AbortSignal.timeout(2500),
        });
      } catch {
        if (liveRef.current) scheduleRetry('stream host stopped answering — reconnecting');
      }
    }, PROBE_INTERVAL_MS);

    return () => clearInterval(timer);
  }, [url, scheduleRetry]);

  return { src, live, note, reconnect, onLoad, onError };
}
