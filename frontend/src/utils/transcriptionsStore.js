/**
 * Single source of truth for the dictation-history localStorage store (#23).
 * Relocates the key/event consts + the reader that were duplicated across
 * Transcriptions.jsx and Projects.jsx. No version/rename/rewrite — the storage
 * contract (key, entry shape, 200-cap) is unchanged; this only de-dups the read.
 */
export const TRANSCRIPTIONS_KEY = 'omni_transcriptions';
export const TRANSCRIPTION_EVENT = 'omni:transcription-added';

/**
 * Read the transcription history. Never throws; always returns an array
 * (newest-first, as written). [] on absent/empty/malformed/non-array/blocked.
 * @returns {Array<object>}
 */
export function loadTranscriptions() {
  try {
    const parsed = JSON.parse(localStorage.getItem(TRANSCRIPTIONS_KEY) || '[]');
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function inTauri() {
  return typeof window !== 'undefined' && '__TAURI_INTERNALS__' in window;
}

/**
 * Announce a new entry. In the desktop app the dictation pill lives in its own
 * widget window, so a window event alone never reaches the main window; a
 * Tauri event carries it across windows (same pattern as dictationLive.js).
 */
export function notifyTranscriptionAdded(entry) {
  if (typeof window === 'undefined') return;
  window.dispatchEvent(new CustomEvent(TRANSCRIPTION_EVENT, { detail: entry }));
  if (inTauri()) {
    void import('@tauri-apps/api/event')
      .then(({ emit }) => emit(TRANSCRIPTION_EVENT, entry))
      .catch((error) => console.warn('transcription update not sent:', error));
  }
}

/**
 * Calls ``callback()`` whenever the history may have changed — in this window,
 * another tab/window (``storage`` event), or another Tauri window. Returns an
 * unsubscribe function. Callers re-read with ``loadTranscriptions()``.
 */
export function subscribeTranscriptions(callback) {
  if (typeof window === 'undefined') return () => {};
  const onWindowEvent = () => callback();
  const onStorage = (event) => {
    if (event.key === null || event.key === TRANSCRIPTIONS_KEY) callback();
  };
  window.addEventListener(TRANSCRIPTION_EVENT, onWindowEvent);
  window.addEventListener('storage', onStorage);
  let unlisten = null;
  let closed = false;
  if (inTauri()) {
    import('@tauri-apps/api/event')
      .then(({ listen }) => listen(TRANSCRIPTION_EVENT, () => callback()))
      .then((stop) => {
        if (closed) stop();
        else unlisten = stop;
      })
      .catch((error) => console.warn('transcription updates unavailable:', error));
  }
  return () => {
    closed = true;
    window.removeEventListener(TRANSCRIPTION_EVENT, onWindowEvent);
    window.removeEventListener('storage', onStorage);
    unlisten?.();
  };
}
