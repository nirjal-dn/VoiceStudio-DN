import { useEffect } from 'react';

/**
 * Adds wall-clock milliseconds to ``setElapsedMs``'s state while ``running``
 * is true; nothing is added while false. Pass a ``useState`` setter.
 *
 * Each tick computes its delta before calling the setter. A state updater must
 * never read a variable that is reassigned right after the setState call:
 * React may run the updater later, during the next render, when the variable
 * has already moved on. The dictation pill's timer did exactly that and added
 * 0 ms per tick while its waveform re-rendered every 50 ms, so it sat at "0s".
 */
export function useElapsedTimer(running, setElapsedMs, { intervalMs = 100 } = {}) {
  useEffect(() => {
    if (!running) return undefined;
    let previous = Date.now();
    const id = setInterval(() => {
      const now = Date.now();
      const delta = now - previous;
      previous = now;
      setElapsedMs((elapsed) => elapsed + delta);
    }, intervalMs);
    return () => clearInterval(id);
  }, [running, setElapsedMs, intervalMs]);
}
