import { useEffect, useState } from 'react';
import { subscribeDictationLive } from '../utils/dictationLive';

/** The dictation pill's latest live snapshot ({ state, paused, text, seconds, noInput }), or null. */
export function useDictationLive() {
  const [live, setLive] = useState(null);
  useEffect(() => subscribeDictationLive(setLive), []);
  return live;
}
