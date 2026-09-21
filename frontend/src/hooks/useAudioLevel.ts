/**
 * Input level meter. Cosmetic, and fails soft: if getUserMedia is refused the
 * bars stay flat and recognition still works.
 */

import { useEffect, useRef, useState } from 'react';

const BAR_COUNT = 12;
const FLAT = new Array<number>(BAR_COUNT).fill(0);

export interface AudioLevel {
  /** One 0-1 value per bar. */
  bars: number[];
  /** Above the noise floor — the meter lights up. */
  hot: boolean;
}

export function useAudioLevel(active: boolean): AudioLevel {
  const [bars, setBars] = useState<number[]>(FLAT);
  const [hot, setHot] = useState(false);
  const raf = useRef<number | null>(null);

  useEffect(() => {
    if (!active || !navigator.mediaDevices) {
      setBars(FLAT);
      setHot(false);
      return;
    }

    let stream: MediaStream | null = null;
    let ctx: AudioContext | null = null;
    let cancelled = false;

    navigator.mediaDevices.getUserMedia({ audio: true }).then((s) => {
      if (cancelled) { s.getTracks().forEach((t) => t.stop()); return; }
      stream = s;
      ctx = new AudioContext();
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 256;
      ctx.createMediaStreamSource(s).connect(analyser);

      const data = new Uint8Array(analyser.frequencyBinCount);
      const step = Math.floor(data.length / BAR_COUNT);

      const draw = () => {
        raf.current = requestAnimationFrame(draw);
        analyser.getByteFrequencyData(data);
        const next = Array.from({ length: BAR_COUNT }, (_, i) => data[i * step] / 255);
        setBars(next);
        setHot(next.reduce((a, b) => a + b, 0) / BAR_COUNT > 0.06);
      };
      draw();
    }).catch(() => {
      /* no meter is fine */
    });

    return () => {
      cancelled = true;
      if (raf.current) cancelAnimationFrame(raf.current);
      raf.current = null;
      stream?.getTracks().forEach((t) => t.stop());
      ctx?.close().catch(() => {});
      setBars(FLAT);
      setHot(false);
    };
  }, [active]);

  return { bars, hot };
}
