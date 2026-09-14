import { act, renderHook } from '@testing-library/react';
import { useEffect, useState } from 'react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { useElapsedTimer } from './useElapsedTimer';

beforeEach(() => {
  vi.useFakeTimers({
    toFake: ['setInterval', 'clearInterval', 'setTimeout', 'clearTimeout', 'Date'],
  });
});

afterEach(() => {
  vi.useRealTimers();
});

// Re-render every 50 ms, like the dictation pill's waveform poll. Batched
// renders run queued state updaters late, which is what froze the old timer.
function useTimerUnderRerenders(running) {
  const [elapsed, setElapsed] = useState(0);
  const [, setTick] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setTick((n) => n + 1), 50);
    return () => clearInterval(id);
  }, []);
  useElapsedTimer(running, setElapsed);
  return [elapsed, setElapsed];
}

it('keeps counting while the component re-renders faster than it ticks', () => {
  const { result } = renderHook(() => useTimerUnderRerenders(true));
  act(() => {
    vi.advanceTimersByTime(2000);
  });
  expect(result.current[0]).toBeGreaterThanOrEqual(1900);
});

it('freezes while not running, resumes, and can be reset', () => {
  const { result, rerender } = renderHook(({ running }) => useTimerUnderRerenders(running), {
    initialProps: { running: true },
  });
  act(() => {
    vi.advanceTimersByTime(1000);
  });
  const atPause = result.current[0];
  expect(atPause).toBeGreaterThanOrEqual(900);

  rerender({ running: false });
  act(() => {
    vi.advanceTimersByTime(5000);
  });
  expect(result.current[0]).toBe(atPause);

  rerender({ running: true });
  act(() => {
    vi.advanceTimersByTime(1000);
  });
  expect(result.current[0]).toBeGreaterThanOrEqual(atPause + 900);

  act(() => {
    result.current[1](0);
  });
  expect(result.current[0]).toBeLessThan(200);
});
