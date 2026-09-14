import { expect, it, vi } from 'vitest';
import { publishDictationLive, subscribeDictationLive } from './dictationLive';

it('delivers published snapshots to subscribers until they unsubscribe', () => {
  const received = vi.fn();
  const unsubscribe = subscribeDictationLive(received);
  const snapshot = {
    state: 'recording',
    paused: false,
    text: 'मेरो order status',
    seconds: 3,
    noInput: false,
  };

  publishDictationLive(snapshot);
  expect(received).toHaveBeenCalledWith(snapshot);

  unsubscribe();
  publishDictationLive({ ...snapshot, seconds: 4 });
  expect(received).toHaveBeenCalledTimes(1);
});
