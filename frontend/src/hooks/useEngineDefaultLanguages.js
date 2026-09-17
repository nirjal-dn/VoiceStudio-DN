import { useEffect, useRef } from 'react';
import { useAppStore } from '../store';
import { LANG_CODES } from '../utils/languages';

function activeDefault(family) {
  return family?.backends?.find((b) => b.id === family.active)?.default_language || null;
}

/**
 * While the active TTS or ASR engine declares a default language (the
 * Nepali-only engines do), select it wherever the matching picker still says
 * "Auto": the generation language (a display name) and the dubbing source
 * language (a code). An explicit choice is never overridden.
 *
 * A value this hook applied follows the engine: switching to an engine with a
 * different default moves it there, and switching to an engine without one puts
 * the picker back on "Auto" — so leaving a Nepali engine does not strand
 * "Nepali" on an engine that cannot speak it.
 */
export function useEngineDefaultLanguages(engines) {
  const ttsCode = activeDefault(engines?.tts);
  const asrCode = activeDefault(engines?.asr);
  const appliedLanguage = useRef(null);
  const appliedSource = useRef(null);

  useEffect(() => {
    const label = LANG_CODES.find((l) => l.code === ttsCode)?.label || null;
    const { language, setLanguage } = useAppStore.getState();
    const untouched = !language || language === 'Auto' || language === appliedLanguage.current;
    if (!untouched) {
      appliedLanguage.current = null;
      return;
    }
    const next = label || 'Auto';
    if (next !== (language || 'Auto')) setLanguage(next);
    appliedLanguage.current = label;
  }, [ttsCode]);

  useEffect(() => {
    const { dubSourceLangCode, setDubSourceLangCode } = useAppStore.getState();
    const untouched =
      !dubSourceLangCode ||
      dubSourceLangCode === 'auto' ||
      dubSourceLangCode === appliedSource.current;
    if (!untouched) {
      appliedSource.current = null;
      return;
    }
    const next = asrCode || 'auto';
    if (next !== (dubSourceLangCode || 'auto')) setDubSourceLangCode(next);
    appliedSource.current = asrCode;
  }, [asrCode]);
}
