import { apiPost } from './client';

export interface TranscriptionResult {
  text: string;
  refined_text?: string;
  segments: Array<{ start?: number | null; end?: number | null; text: string }>;
  language?: string;
  duration_s?: number;
  transcription_time_s?: number;
  engine?: string;
}

export async function transcribeAudio(
  audio: File,
  {
    mode = 'fast',
    language = '',
    refine = false,
    signal,
  }: {
    mode?: 'fast' | 'accurate';
    language?: string;
    refine?: boolean;
    signal?: AbortSignal;
  } = {},
): Promise<TranscriptionResult> {
  const form = new FormData();
  form.append('audio', audio, audio.name || 'audio');
  form.append('mode', mode);
  if (language.trim()) form.append('language', language.trim());
  if (refine) form.append('refine', '1');
  return apiPost<TranscriptionResult>('/transcribe', form, { signal });
}
