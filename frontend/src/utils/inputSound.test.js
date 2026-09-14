import { expect, it } from 'vitest';
import { frameHasSound, SOUND_THRESHOLD } from './inputSound';

it('treats digital silence and a sub-threshold noise floor as no sound', () => {
  expect(frameHasSound(new Float32Array(320))).toBe(false);
  expect(frameHasSound(new Float32Array(320).fill(SOUND_THRESHOLD / 2))).toBe(false);
});

it('detects a single audible sample of either sign', () => {
  const positive = new Float32Array(320);
  positive[200] = 0.02;
  expect(frameHasSound(positive)).toBe(true);
  const negative = new Float32Array(320);
  negative[10] = -0.02;
  expect(frameHasSound(negative)).toBe(true);
});
