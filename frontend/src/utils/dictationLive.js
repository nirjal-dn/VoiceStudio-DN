// The dictation pill's live session, shared with other views such as the
// Transcriptions page's live transcript. In the browser build the pill runs in
// the same page, so a window event carries it; in the desktop app the pill has
// its own widget window, so a Tauri event carries it across windows.
//
// Snapshot: { state, paused, text, seconds, noInput }.

export const DICTATION_LIVE_EVENT = 'voicestudio:dictation-live';

function inTauri() {
  return typeof window !== 'undefined' && '__TAURI_INTERNALS__' in window;
}

export function publishDictationLive(snapshot) {
  if (typeof window === 'undefined') return;
  window.dispatchEvent(new CustomEvent(DICTATION_LIVE_EVENT, { detail: snapshot }));
  if (inTauri()) {
    void import('@tauri-apps/api/event')
      .then(({ emit }) => emit(DICTATION_LIVE_EVENT, snapshot))
      .catch((error) => console.warn('dictation live update not sent:', error));
  }
}

/** Calls ``callback(snapshot)`` for every update; returns an unsubscribe function. */
export function subscribeDictationLive(callback) {
  if (typeof window === 'undefined') return () => {};
  const onWindowEvent = (event) => callback(event.detail);
  window.addEventListener(DICTATION_LIVE_EVENT, onWindowEvent);
  let unlisten = null;
  let closed = false;
  if (inTauri()) {
    import('@tauri-apps/api/event')
      .then(({ listen }) => listen(DICTATION_LIVE_EVENT, (event) => callback(event.payload)))
      .then((stop) => {
        if (closed) stop();
        else unlisten = stop;
      })
      .catch((error) => console.warn('dictation live updates unavailable:', error));
  }
  return () => {
    closed = true;
    window.removeEventListener(DICTATION_LIVE_EVENT, onWindowEvent);
    unlisten?.();
  };
}
