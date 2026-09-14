// Detects a microphone that delivers no sound at all: nothing connected, a
// muted or wrong input device, or a sound card without a capture path. The
// recording otherwise looks healthy (the socket streams frames) while the
// waveform stays flat and nothing is ever transcribed.

/** Samples at or below this magnitude count as silence (-60 dBFS). */
export const SOUND_THRESHOLD = 0.001;

/** How long a recording may go without one audible frame before warning. */
export const NO_INPUT_AFTER_MS = 3000;

/** Whether any sample in a Float32 PCM frame is louder than the threshold. */
export function frameHasSound(frame, threshold = SOUND_THRESHOLD) {
  for (let i = 0; i < frame.length; i += 1) {
    if (frame[i] > threshold || frame[i] < -threshold) return true;
  }
  return false;
}
