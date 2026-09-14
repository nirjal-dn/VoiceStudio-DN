import { useEffect } from 'react';
import { useAppStore } from '../store';
import { LANG_CODES } from '../utils/languages';

function activeDefault(family) {
  return family?.backends?.find((b) => b.id === family.active)?.default_language || null;
}

/**
 * While the active TTS or ASR engine declares a default language (the
 * Nepali-only engines do), select it wherever the matching picker still says
 * "Auto": the generation language (a display name) and the dubbing source
 * language (a code). Runs when the active engine changes; an explicit choice is
 * never overridden.
 */
export function useEngineDefaultLanguages(engines) {
  const ttsCode = activeDefault(engines?.tts);
  const asrCode = activeDefault(engines?.asr);

  useEffect(() => {
    const label = LANG_CODES.find((l) => l.code === ttsCode)?.label;
    const { language, setLanguage } = useAppStore.getState();
    if (label && (!language || language === 'Auto')) setLanguage(label);
  }, [ttsCode]);

  useEffect(() => {
    const { dubSourceLangCode, setDubSourceLangCode } = useAppStore.getState();
    if (asrCode && (!dubSourceLangCode || dubSourceLangCode === 'auto')) {
      setDubSourceLangCode(asrCode);
    }
  }, [asrCode]);
}
