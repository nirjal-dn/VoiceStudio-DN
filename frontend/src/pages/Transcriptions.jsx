/**
 * TranscriptionsPage — history of dictation transcriptions.
 *
 * Stores transcriptions in localStorage and displays them in a searchable,
 * timestamped list. Each entry can be copied, deleted, or re-used.
 *
 * Reactivity: addTranscription() dispatches a custom window event so the
 * page updates in realtime without requiring a shared store.
 */
import React, { useState, useCallback, useMemo, useEffect, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import {
  Mic,
  Trash2,
  Search,
  Clock,
  Languages,
  FileText,
  FileAudio,
  FileDown,
  Square,
  Upload,
  X,
  Pencil,
  Clipboard,
  RotateCw,
  ChevronDown,
  Check,
  Save,
} from 'lucide-react';
import { Button } from '../ui';
import WaveformPlayer from '../components/WaveformPlayer';
import { detectPlatform } from '../utils/micError';
import { useDictationReadiness } from '../hooks/useDictationReadiness';
import AsrModelChooser from '../components/AsrModelChooser';
import { toast } from 'react-hot-toast';
import { copyText as copyToClipboard } from '../utils/copyText';
import { toMillis } from '../utils/relativeTime';
import { useEffectiveDictationShortcut } from '../hooks/useEffectiveDictationShortcut';
import { requestDictationCapture } from '../utils/dictationCapture';
import { useDictationLive } from '../hooks/useDictationLive';
import { transcribeAudio } from '../api/transcriptions';
import { asrMissingPayload, toastAsrModelMissing } from '../utils/asrModelMissing';
import {
  loadTranscriptions,
  notifyTranscriptionAdded,
  subscribeTranscriptions,
  TRANSCRIPTIONS_KEY,
} from '../utils/transcriptionsStore';

function saveTranscriptions(list) {
  localStorage.setItem(TRANSCRIPTIONS_KEY, JSON.stringify(list));
}

/** A segment's "12.0s – 15.5s" label, tolerant of missing timings (#1798).
 *
 * Not every ASR path produces a timed segment: an OpenAI-compatible backend
 * answering in `json`/`text` format has no timings at all, and
 * `services/asr_backend.py` records that honestly as `end: None` rather than
 * inventing a number. Calling `.toFixed()` on it threw during render and took
 * the whole Transcriptions view down, so a transcript that merely lacked
 * timings became one the user could not read at all. Render whichever half is
 * known, and nothing when neither is. */
export function segTimeRange(seg) {
  const known = (v) => typeof v === 'number' && Number.isFinite(v);
  const start = known(seg?.start) ? `${seg.start.toFixed(1)}s` : null;
  const end = known(seg?.end) ? `${seg.end.toFixed(1)}s` : null;
  if (start && end) return `${start} – ${end}`;
  return start || end || '';
}

export function addTranscription(entry) {
  const list = loadTranscriptions();
  const newEntry = {
    id: Date.now(),
    text: entry.text || entry.raw_text || '',
    raw_text: entry.raw_text || entry.text || '',
    language: entry.language || 'unknown',
    duration_s: entry.duration_s || 0,
    segments: entry.segments || [],
    refined_text: entry.refined_text || undefined,
    source: entry.source || 'dictation',
    engine: entry.engine || undefined,
    timestamp: new Date().toISOString(),
  };
  list.unshift(newEntry);
  // Keep last 200
  if (list.length > 200) list.length = 200;
  saveTranscriptions(list);
  // Reactive updates, including the main window when the desktop pill wrote it
  notifyTranscriptionAdded(newEntry);
  return newEntry;
}

function formatLiveClock(seconds = 0) {
  const total = Math.max(0, Math.floor(seconds));
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, '0')}`;
}

export default function TranscriptionsPage() {
  const { t } = useTranslation();
  const [transcriptions, setTranscriptions] = useState(loadTranscriptions);
  const [search, setSearch] = useState('');
  const [selectedId, setSelectedId] = useState(null);
  const { info: shortcut } = useEffectiveDictationShortcut();
  const readiness = useDictationReadiness();
  const checkReadiness = readiness.check;
  // What the progress bar names: the model the user picked, else the recommended one.
  const installTarget = readiness.target || readiness.missing?.recommended;
  const [starting, setStarting] = useState(false);
  const [uploadMode, setUploadMode] = useState('accurate');
  const [modeOpen, setModeOpen] = useState(false);
  const modeMenuRef = useRef(null);
  const [uploadLanguage, setUploadLanguage] = useState('');
  const [uploading, setUploading] = useState(false);
  const [selectedAudio, setSelectedAudio] = useState(null);
  const [editing, setEditing] = useState(false);
  const [draftText, setDraftText] = useState('');
  const [showRefined, setShowRefined] = useState(false);
  const uploadAbortRef = useRef(null);
  const captureDisabled = readiness.phase !== 'ready' || starting;
  const emptyDescription = t('transcriptions.empty_desc', { shortcut: shortcut.display });
  const normalizedSearch = search.trim();
  // Live transcript of the dictation in progress, streamed from the capture pill.
  const live = useDictationLive();
  const liveActive = live?.state === 'recording' || live?.state === 'transcribing';
  const stopCapture = useCallback(() => {
    requestDictationCapture('stop').catch((error) =>
      console.warn('Could not stop dictation:', error),
    );
  }, []);

  const startCapture = useCallback(async () => {
    if (captureDisabled) return;
    setStarting(true);
    try {
      if (!(await checkReadiness())) return;
      await requestDictationCapture('start');
    } catch (error) {
      console.warn('Could not start dictation:', error);
      toast.error(t('transcriptions.capture_failed'));
    } finally {
      setStarting(false);
    }
  }, [t, captureDisabled, checkReadiness]);

  const handleAudioUpload = useCallback(
    async (file) => {
      if (!file) return;
      setSelectedAudio(file);
      setUploading(true);
      const controller = new AbortController();
      uploadAbortRef.current = controller;
      try {
        const result = await transcribeAudio(file, {
          mode: uploadMode,
          language: uploadLanguage,
          signal: controller.signal,
        });
        const entry = addTranscription({
          ...result,
          text: result.text,
          raw_text: result.text,
          source: 'upload',
        });
        setSelectedId(entry.id);
        toast.success(t('transcriptions.uploaded', { defaultValue: 'Audio transcribed' }));
      } catch (error) {
        if (error?.name === 'AbortError') {
          toast(t('transcriptions.cancelled', { defaultValue: 'Transcription cancelled' }));
          return;
        }
        const missing = asrMissingPayload(error);
        if (missing) toastAsrModelMissing(missing);
        else toast.error(error?.message || t('transcriptions.upload_failed'));
      } finally {
        uploadAbortRef.current = null;
        setUploading(false);
      }
    },
    [t, uploadMode, uploadLanguage],
  );

  const cancelUpload = useCallback(() => {
    uploadAbortRef.current?.abort();
  }, []);

  const clearSelectedAudio = useCallback(() => {
    if (uploading) {
      uploadAbortRef.current?.abort();
    }
    setSelectedAudio(null);
  }, [uploading]);

  useEffect(() => () => uploadAbortRef.current?.abort(), []);

  useEffect(() => {
    if (!modeOpen) return undefined;
    const closeOnOutsideClick = (event) => {
      if (!modeMenuRef.current?.contains(event.target)) setModeOpen(false);
    };
    const closeOnEscape = (event) => {
      if (event.key === 'Escape') setModeOpen(false);
    };
    document.addEventListener('mousedown', closeOnOutsideClick);
    document.addEventListener('keydown', closeOnEscape);
    return () => {
      document.removeEventListener('mousedown', closeOnOutsideClick);
      document.removeEventListener('keydown', closeOnEscape);
    };
  }, [modeOpen]);

  // Listen for new transcriptions, including ones the desktop pill's own
  // window adds.
  useEffect(() => subscribeTranscriptions(() => setTranscriptions(loadTranscriptions())), []);

  useEffect(() => {
    if (!selectedId && transcriptions.length > 0) {
      setSelectedId(transcriptions[0].id);
    } else if (selectedId && !transcriptions.some((entry) => entry.id === selectedId)) {
      setSelectedId(transcriptions[0]?.id || null);
    }
  }, [selectedId, transcriptions]);

  const filtered = useMemo(() => {
    if (!normalizedSearch) return transcriptions;
    const q = normalizedSearch.toLowerCase();
    return transcriptions.filter(
      (t) => t.text.toLowerCase().includes(q) || (t.language || '').toLowerCase().includes(q),
    );
  }, [transcriptions, normalizedSearch]);

  const selected = useMemo(
    () => transcriptions.find((t) => t.id === selectedId),
    [transcriptions, selectedId],
  );

  useEffect(() => {
    setDraftText(selected?.text || '');
    setEditing(false);
    setShowRefined(false);
  }, [selectedId, selected?.text]);

  const saveEditedTranscript = useCallback(() => {
    if (!selected) return;
    const next = loadTranscriptions().map((entry) =>
      entry.id === selected.id ? { ...entry, text: draftText, edited: true } : entry,
    );
    saveTranscriptions(next);
    setTranscriptions(next);
    setEditing(false);
    notifyTranscriptionAdded(next.find((entry) => entry.id === selected.id));
    toast.success(t('transcriptions.saved', { defaultValue: 'Transcript saved' }));
  }, [draftText, selected, t]);

  const copyText = useCallback(
    (text) => {
      copyToClipboard(text).then(
        (copied) =>
          copied
            ? toast.success(t('transcriptions.copied'))
            : toast.error(t('transcriptions.copy_failed')),
        () => toast.error(t('transcriptions.copy_failed')),
      );
    },
    [t],
  );

  const deleteEntry = useCallback(
    (id) => {
      // Filter the stored list, not this view's copy: another window may have
      // added entries this page has not re-read yet.
      const next = loadTranscriptions().filter((t) => t.id !== id);
      setTranscriptions(next);
      saveTranscriptions(next);
      if (selectedId === id) setSelectedId(null);
    },
    [selectedId],
  );

  const clearAll = useCallback(() => {
    setTranscriptions([]);
    saveTranscriptions([]);
    setSelectedId(null);
  }, []);

  const exportAll = useCallback(() => {
    const text = transcriptions
      .map((t) => `[${new Date(t.timestamp).toLocaleString()}] (${t.language})\n${t.text}\n`)
      .join('\n---\n\n');
    const blob = new Blob([text], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `transcriptions_${new Date().toISOString().slice(0, 10)}.txt`;
    a.click();
    URL.revokeObjectURL(url);
    toast.success(t('transcriptions.exported'));
  }, [t, transcriptions]);

  const downloadText = useCallback((name, content, type) => {
    const url = URL.createObjectURL(new Blob([content], { type }));
    const a = document.createElement('a');
    a.href = url;
    a.download = name;
    a.click();
    URL.revokeObjectURL(url);
  }, []);

  const exportSegments = useCallback(
    (format) => {
      if (!selected?.segments?.length) return;
      const stamp = (seconds) => {
        const ms = Math.max(0, Math.round((Number(seconds) || 0) * 1000));
        const h = String(Math.floor(ms / 3600000)).padStart(2, '0');
        const m = String(Math.floor((ms % 3600000) / 60000)).padStart(2, '0');
        const s = String(Math.floor((ms % 60000) / 1000)).padStart(2, '0');
        const milli = String(ms % 1000).padStart(3, '0');
        return format === 'srt' ? `${h}:${m}:${s},${milli}` : `${h}:${m}:${s}.${milli}`;
      };
      const content = selected.segments
        .map((segment, index) => {
          const start = stamp(segment.start);
          const end = stamp(segment.end ?? segment.start);
          return format === 'srt'
            ? `${index + 1}\n${start} --> ${end}\n${segment.text}\n`
            : `${index + 1}\n${start} --> ${end}\n${segment.text}\n`;
        })
        .join('\n');
      downloadText(
        `transcription.${format}`,
        format === 'vtt' ? `WEBVTT\n\n${content}` : content,
        'text/plain;charset=utf-8',
      );
    },
    [downloadText, selected],
  );

  // toMillis keeps this unit-safe (ISO strings today; seconds/ms tolerated)
  // and guards unparseable stamps, which used to render "Invalid Date".
  const formatTime = (iso) => {
    const ms = toMillis(iso);
    if (ms == null) return null;
    const d = new Date(ms);
    const diff = Date.now() - ms;
    if (diff < 60000) return t('transcriptions.just_now');
    if (diff < 3600000) return t('transcriptions.m_ago', { count: Math.floor(diff / 60000) });
    if (diff < 86400000) return t('transcriptions.h_ago', { count: Math.floor(diff / 3600000) });
    return d.toLocaleDateString(undefined, {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  return (
    <div
      className="txn-page flex h-full flex-col gap-5 bg-bg px-6 py-5 font-sans"
      role="region"
      aria-label={t('transcriptions.title')}
    >
      {/* Header */}
      <div className="txn-header flex flex-wrap items-end justify-between gap-4">
        <div className="txn-header__left">
          <div className="mb-1 flex items-center gap-2 text-xs font-medium uppercase tracking-[0.12em] text-brand">
            <FileText size={14} />
            {t('transcriptions.title')}
          </div>
          <h1 className="txn-header__title m-0 text-2xl font-semibold tracking-[-0.02em] text-fg">
            {t('transcriptions.workspace_title', { defaultValue: 'Transcription workspace' })}
          </h1>
          <p className="m-1.5 max-w-[620px] text-sm text-fg-muted">
            {t('transcriptions.workspace_desc', {
              defaultValue: 'Turn recordings into searchable, editable transcripts.',
            })}
          </p>
        </div>
        <div className="txn-header__right flex flex-wrap items-center justify-end gap-2">
          {transcriptions.length > 0 && (
            <Button
              size="sm"
              variant="primary"
              leading={<Mic size={13} />}
              disabled={captureDisabled}
              onClick={startCapture}
            >
              {t('transcriptions.capture')}
            </Button>
          )}
          <div className="txn-search relative flex items-center">
            <Search
              size={14}
              className="txn-search__icon pointer-events-none absolute left-3 text-fg-subtle"
            />
            <input
              className="w-[230px] rounded-lg border border-border bg-bg-elev-1 py-2 pl-9 pr-3 text-sm text-fg outline-none transition-colors placeholder:text-fg-subtle focus:border-brand"
              placeholder={t('transcriptions.search_placeholder')}
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              aria-label={t('transcriptions.search_placeholder')}
            />
          </div>
          {transcriptions.length > 0 && (
            <>
              <Button
                size="sm"
                variant="ghost"
                leading={<FileDown size={13} />}
                onClick={exportAll}
                title={t('transcriptions.export_title')}
              >
                {t('transcriptions.export')}
              </Button>
              <Button
                size="sm"
                variant="ghost"
                leading={<Trash2 size={13} />}
                onClick={clearAll}
                title={t('transcriptions.clear_title')}
              >
                {t('transcriptions.clear')}
              </Button>
            </>
          )}
        </div>
      </div>

      <section
        className="txn-upload flex flex-wrap items-center gap-4 rounded-xl border border-border bg-bg-elev-1 p-4 shadow-sm"
        aria-label={t('transcriptions.upload_options', { defaultValue: 'Audio transcription' })}
        onDragOver={(event) => {
          event.preventDefault();
          event.stopPropagation();
        }}
        onDrop={(event) => {
          event.preventDefault();
          event.stopPropagation();
          // The whole box is the drop zone — not just the small label. Dropping
          // a file onto the hint area used to be silently swallowed here.
          const file = event.dataTransfer.files?.[0];
          if (file && !uploading) void handleAudioUpload(file);
        }}
      >
        <input
          id="transcription-audio-upload"
          type="file"
          accept="audio/*,.mp3,.wav,.m4a,.flac,.ogg,.aac,.webm"
          className="sr-only"
          disabled={uploading}
          onChange={(event) => {
            const file = event.target.files?.[0];
            event.target.value = '';
            void handleAudioUpload(file);
          }}
        />
        <div className="min-w-[240px] flex-1">
          <div className="flex items-center gap-2 text-sm font-semibold text-fg">
            <Upload size={15} aria-hidden="true" />
            {t('transcriptions.upload_options', { defaultValue: 'Transcribe an audio file' })}
          </div>
          <p className="m-0 mt-1 text-xs leading-5 text-fg-muted">
            {selectedAudio
              ? selectedAudio.name
              : t('transcriptions.upload_hint', {
                  defaultValue: 'Choose an audio file to generate a complete transcript.',
                })}
          </p>
        </div>
        <label
          htmlFor="transcription-audio-upload"
          className={`inline-flex min-h-10 cursor-pointer items-center gap-2 rounded-lg border border-dashed border-brand px-4 py-2 text-xs font-semibold text-brand transition-colors hover:bg-brand/10 ${
            uploading ? 'pointer-events-none opacity-60' : ''
          }`}
          onDragEnter={(event) => {
            event.preventDefault();
            event.stopPropagation();
          }}
          onDragOver={(event) => {
            event.preventDefault();
            event.stopPropagation();
          }}
          onDrop={(event) => {
            event.preventDefault();
            event.stopPropagation();
            const file = event.dataTransfer.files?.[0];
            if (!file) return;
            void handleAudioUpload(file);
          }}
        >
          <Upload size={13} aria-hidden="true" />
          {uploading
            ? t('transcriptions.transcribing_upload', { defaultValue: 'Transcribing…' })
            : t('transcriptions.choose_audio', { defaultValue: 'Choose or drop audio' })}
        </label>
        {uploading && (
          <Button size="sm" variant="ghost" leading={<Square size={12} />} onClick={cancelUpload}>
            {t('common.cancel', { defaultValue: 'Cancel' })}
          </Button>
        )}
        {selectedAudio && (
          <div className="flex min-w-[260px] flex-[1.4] items-center gap-2">
            <WaveformPlayer
              src={selectedAudio}
              source="transcription-preview"
              height={44}
              className="min-w-0 flex-1"
            />
            <Button
              variant="icon"
              iconSize="sm"
              onClick={clearSelectedAudio}
              title={t('transcriptions.clear_audio', { defaultValue: 'Clear selected audio' })}
              aria-label={t('transcriptions.clear_audio', { defaultValue: 'Clear selected audio' })}
            >
              <X size={14} />
            </Button>
          </div>
        )}
        <div className="flex items-center gap-2 text-xs text-fg-muted">
          <span id="transcription-mode-label">
            {t('transcriptions.mode', { defaultValue: 'Mode' })}
          </span>
          <div ref={modeMenuRef} className="relative">
            <button
              type="button"
              id="transcription-mode"
              className="input-base flex h-8 min-w-[104px] items-center justify-between gap-[6px] rounded-lg border-0 px-2 py-1 text-left text-[0.65rem] font-medium text-[var(--text-primary)] transition-colors hover:bg-[var(--color-bg-elev-1)] focus-visible:outline-2 focus-visible:outline-[var(--chrome-accent)]"
              aria-haspopup="listbox"
              aria-expanded={modeOpen}
              aria-labelledby="transcription-mode-label"
              onClick={() => setModeOpen((open) => !open)}
            >
              <span>
                {uploadMode === 'accurate'
                  ? t('transcriptions.accurate', { defaultValue: 'Accurate' })
                  : t('transcriptions.fast', { defaultValue: 'Fast' })}
              </span>
              <ChevronDown
                size={13}
                className={`shrink-0 transition-transform ${modeOpen ? 'rotate-180' : ''}`}
                aria-hidden="true"
              />
            </button>
            {modeOpen && (
              <div
                className="absolute right-0 z-[var(--z-overlay)] mt-1 min-w-[136px] max-w-[calc(100vw-16px)] overflow-hidden rounded-lg border-0 bg-[var(--color-bg)] p-0 text-[var(--color-fg)] shadow-xl"
                role="listbox"
                aria-labelledby="transcription-mode-label"
              >
                {[
                  ['accurate', t('transcriptions.accurate', { defaultValue: 'Accurate' })],
                  ['fast', t('transcriptions.fast', { defaultValue: 'Fast' })],
                ].map(([value, label]) => (
                  <button
                    key={value}
                    type="button"
                    role="option"
                    aria-selected={uploadMode === value}
                    className="flex w-full items-center justify-between gap-3 rounded-none border-0 bg-transparent px-2.5 py-1.5 text-left text-xs font-medium text-[var(--chrome-fg)] transition-colors hover:bg-[var(--chrome-hover-bg)] focus-visible:bg-[var(--chrome-hover-bg)] focus-visible:outline-none"
                    onClick={() => {
                      setUploadMode(value);
                      setModeOpen(false);
                    }}
                  >
                    {label}
                    {uploadMode === value && <Check size={12} className="text-brand" />}
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>
        <label className="flex items-center gap-2 text-xs text-fg-muted">
          {t('transcriptions.language_hint', { defaultValue: 'Language' })}
          <input
            value={uploadLanguage}
            onChange={(event) => setUploadLanguage(event.target.value)}
            placeholder="auto"
            className="h-8 w-[68px] rounded-lg border border-border bg-bg px-2 py-1 text-xs text-fg"
            aria-label={t('transcriptions.language_hint', { defaultValue: 'Language' })}
          />
        </label>
      </section>

      {readiness.phase !== 'ready' && (
        <div
          className="rounded-lg border border-border bg-bg-elev-1 p-4 flex flex-col gap-3"
          role="status"
          aria-live="polite"
        >
          <p className="text-sm text-fg m-0">
            {readiness.phase === 'checking'
              ? t('setup.checking')
              : readiness.phase === 'error'
                ? t('common.error')
                : readiness.phase === 'installing'
                  ? t('dub.install_progress', { engine: installTarget?.label })
                  : t('asr_missing.message')}
          </p>
          {readiness.phase === 'installing' ? (
            <progress
              className="w-full"
              max={100}
              value={readiness.percent ?? undefined}
              aria-label={t('dub.install_progress', { engine: installTarget?.label })}
            />
          ) : (
            readiness.phase !== 'checking' && (
              <>
                {readiness.phase === 'missing' && (
                  <AsrModelChooser
                    fallback={readiness.missing?.recommended}
                    onInstall={readiness.install}
                    onSelect={readiness.select}
                  />
                )}
                <div className="flex items-center gap-3">
                  <Button
                    size="sm"
                    variant="ghost"
                    leading={<RotateCw size={12} />}
                    onClick={readiness.check}
                  >
                    {t('common.refresh')}
                  </Button>
                  {readiness.error && <span role="alert">{t('common.error')}</span>}
                </div>
              </>
            )
          )}
        </div>
      )}

      {liveActive && (
        <section
          className="flex flex-col gap-2 rounded-lg border border-border bg-bg-elev-1 p-4"
          aria-label={t('transcriptions.live_title')}
        >
          <div className="flex items-center justify-between gap-3">
            <span className="flex items-center gap-2 text-sm font-semibold text-fg">
              <span
                aria-hidden="true"
                className={`h-2 w-2 rounded-full bg-brand ${
                  live.state === 'recording' && !live.paused
                    ? 'motion-safe:animate-pulse'
                    : 'opacity-40'
                }`}
              />
              {t('transcriptions.live_title')}
            </span>
            <div className="flex items-center gap-3">
              <span className="font-mono text-xs text-fg-muted">
                {formatLiveClock(live.seconds)}
              </span>
              {live.state === 'recording' && (
                <Button
                  size="sm"
                  variant="ghost"
                  leading={<Square size={12} />}
                  onClick={stopCapture}
                >
                  {t('common.stop')}
                </Button>
              )}
            </div>
          </div>
          <p
            className="m-0 whitespace-pre-wrap text-sm leading-[1.7] text-fg [word-break:break-word]"
            aria-live="polite"
          >
            {live.text ||
              (live.state === 'transcribing'
                ? t('capture.transcribing_label')
                : live.noInput
                  ? t('capture.no_input')
                  : t('capture.listening_label'))}
          </p>
        </section>
      )}

      <div className="flex flex-wrap items-center gap-2 text-xs text-fg-muted">
        <kbd className="rounded border border-border bg-bg-elev-1 px-2 py-1 font-mono">
          {shortcut.display}
        </kbd>
        <span>
          {t('transcriptions.capture')} / {t('common.stop')}
        </span>
        <span aria-hidden="true" className="mx-2">
          ·
        </span>
        <kbd className="rounded border border-border bg-bg-elev-1 px-2 py-1 font-mono">
          {detectPlatform() === 'mac' ? '⌘+V' : 'Ctrl+V'}
        </kbd>
        <span>{t('clone.paste')}</span>
      </div>

      {/* Content */}
      <div className="txn-content grid min-h-0 flex-1 grid-cols-[minmax(250px,0.34fr)_minmax(0,1fr)] gap-4 max-[900px]:grid-cols-1">
        {/* List */}
        <section className="txn-library flex min-h-0 flex-col overflow-hidden rounded-xl border border-border bg-bg-elev-1 shadow-sm">
          <div className="flex items-center justify-between border-b border-border px-4 py-3">
            <div>
              <h2 className="m-0 text-sm font-semibold text-fg">
                {t('transcriptions.library', { defaultValue: 'Transcript library' })}
              </h2>
              <p className="m-0 mt-1 text-xs text-fg-muted">
                {t('transcriptions.entries', { count: transcriptions.length })}
              </p>
            </div>
            <FileText size={16} className="text-fg-subtle" aria-hidden="true" />
          </div>
          <div
            className="txn-list flex min-h-0 flex-1 flex-col gap-2 overflow-y-auto p-3"
            role="list"
          >
            {filtered.length === 0 ? (
              <div className="txn-empty flex h-full flex-col items-center justify-center gap-2 px-5 py-10 text-center text-fg-muted">
                <Mic size={32} className="txn-empty__icon opacity-30" />
                <p className="txn-empty__title m-0 text-[var(--text-sm)] font-medium text-fg">
                  {normalizedSearch
                    ? t('transcriptions.empty_search_title')
                    : t('transcriptions.empty_title')}
                </p>
                <p className="txn-empty__desc m-0 max-w-[280px] text-[var(--text-xs)] leading-[1.6] text-fg-muted">
                  {normalizedSearch
                    ? t('transcriptions.empty_search_desc')
                    : readiness.phase === 'ready'
                      ? emptyDescription
                      : ''}
                </p>
                {!normalizedSearch && (
                  <Button
                    size="sm"
                    variant="primary"
                    leading={<Mic size={13} />}
                    disabled={captureDisabled}
                    onClick={startCapture}
                  >
                    {t('transcriptions.capture')}
                  </Button>
                )}
              </div>
            ) : (
              filtered.map((t) => (
                <div
                  key={t.id}
                  role="listitem"
                  className={`cursor-pointer rounded-lg border p-3 transition-colors hover:border-brand/50 ${
                    selectedId === t.id
                      ? 'border-brand bg-brand/10 shadow-[0_0_0_1px_var(--color-brand-glow)]'
                      : ''
                  }`}
                  onClick={() => setSelectedId(t.id)}
                >
                  <div className="txn-item__text mb-[6px] text-[var(--text-sm)] leading-[1.5] text-fg [word-break:break-word]">
                    {t.text.length > 120 ? t.text.slice(0, 120) + '…' : t.text}
                  </div>
                  <div className="txn-item__meta flex items-center gap-[10px] text-[10px] text-fg-subtle">
                    <span className="txn-item__time flex items-center gap-[3px]">
                      <Clock size={10} /> {formatTime(t.timestamp)}
                    </span>
                    {t.language && t.language !== 'unknown' && (
                      <span className="txn-item__lang flex items-center gap-[3px]">
                        <Languages size={10} /> {t.language}
                      </span>
                    )}
                    {t.duration_s > 0 && (
                      <span className="txn-item__dur flex items-center gap-[3px]">
                        {t.duration_s.toFixed(1)}s
                      </span>
                    )}
                  </div>
                </div>
              ))
            )}
          </div>
        </section>

        {/* Detail panel */}
        <section className="txn-detail flex min-h-0 flex-col overflow-hidden rounded-xl border border-border bg-bg-elev-1 shadow-sm">
          {selected ? (
            <>
              <div className="txn-detail__header flex flex-wrap items-center justify-between gap-3 px-[14px] py-[10px] [border-bottom:1px_solid_var(--color-border)]">
                <div className="min-w-0">
                  <div className="flex items-center gap-2 text-sm font-semibold text-fg">
                    <FileAudio size={15} className="text-brand" />
                    {selected.source === 'upload'
                      ? t('transcriptions.imported_audio', { defaultValue: 'Imported audio' })
                      : t('transcriptions.dictation', { defaultValue: 'Dictation' })}
                  </div>
                  <span className="text-xs text-fg-muted">
                    {new Date(selected.timestamp).toLocaleString()}
                  </span>
                </div>
                <div className="txn-detail__actions flex flex-wrap items-center justify-end gap-1">
                  <Button
                    size="sm"
                    variant="ghost"
                    leading={<Clipboard size={12} />}
                    onClick={() => copyText(showRefined ? selected.refined_text : selected.text)}
                  >
                    {t('transcriptions.copy')}
                  </Button>
                  <Button
                    size="sm"
                    variant="ghost"
                    leading={<Pencil size={12} />}
                    onClick={() => setEditing((value) => !value)}
                  >
                    {editing
                      ? t('common.cancel', { defaultValue: 'Cancel' })
                      : t('transcriptions.edit', { defaultValue: 'Edit' })}
                  </Button>
                  {selected.segments?.length > 0 && (
                    <>
                      <Button
                        size="sm"
                        variant="ghost"
                        leading={<FileDown size={12} />}
                        onClick={() => exportSegments('srt')}
                      >
                        SRT
                      </Button>
                      <Button
                        size="sm"
                        variant="ghost"
                        leading={<FileDown size={12} />}
                        onClick={() => exportSegments('vtt')}
                      >
                        VTT
                      </Button>
                    </>
                  )}
                  <Button
                    size="sm"
                    variant="ghost"
                    leading={<Trash2 size={12} />}
                    onClick={() => deleteEntry(selected.id)}
                  >
                    {t('transcriptions.delete')}
                  </Button>
                </div>
              </div>
              <div className="txn-detail__body min-h-0 flex-1 overflow-y-auto p-5">
                {selected.refined_text && (
                  <div className="mb-3 flex items-center gap-2 text-xs text-fg-muted">
                    <Button
                      size="sm"
                      variant={showRefined ? 'primary' : 'ghost'}
                      onClick={() => setShowRefined(true)}
                    >
                      {t('transcriptions.refined', { defaultValue: 'Refined' })}
                    </Button>
                    <Button
                      size="sm"
                      variant={!showRefined ? 'primary' : 'ghost'}
                      onClick={() => setShowRefined(false)}
                    >
                      {t('transcriptions.raw', { defaultValue: 'Raw' })}
                    </Button>
                  </div>
                )}
                {editing ? (
                  <textarea
                    className="min-h-[220px] w-full resize-y rounded border border-border bg-bg p-3 text-sm leading-[1.7] text-fg"
                    value={draftText}
                    onChange={(event) => setDraftText(event.target.value)}
                    aria-label={t('transcriptions.edit', { defaultValue: 'Edit transcript' })}
                  />
                ) : (
                  <p className="txn-detail__text m-0 max-w-[850px] text-base leading-[1.85] whitespace-pre-wrap text-fg">
                    {showRefined ? selected.refined_text : selected.text}
                  </p>
                )}
                {editing && (
                  <Button
                    size="sm"
                    variant="primary"
                    leading={<Save size={12} />}
                    onClick={saveEditedTranscript}
                    className="mt-3"
                  >
                    {t('common.save', { defaultValue: 'Save' })}
                  </Button>
                )}
              </div>
              {selected.segments && selected.segments.length > 0 && (
                <div className="txn-detail__segments [border-top:1px_solid_var(--color-border)] px-[14px] py-[10px] max-h-[200px] overflow-y-auto">
                  <div className="[font-size:var(--text-xs)] [font-weight:var(--weight-semibold)] text-fg-muted m-0 mb-[6px] uppercase tracking-[0.5px]">
                    {t('transcriptions.segments_title')}
                  </div>
                  {selected.segments.map((seg, i) => (
                    <div
                      key={i}
                      className="txn-detail__seg flex gap-[8px] py-[3px] text-[var(--text-xs)]"
                    >
                      <span className="txn-detail__seg-time shrink-0 font-mono text-fg-subtle min-w-[80px]">
                        {segTimeRange(seg)}
                      </span>
                      <span className="txn-detail__seg-text text-fg">{seg.text}</span>
                    </div>
                  ))}
                </div>
              )}
            </>
          ) : (
            <div className="flex h-full flex-col items-center justify-center gap-3 p-8 text-center text-fg-muted">
              <FileText size={36} className="opacity-30" />
              <h2 className="m-0 text-base font-semibold text-fg">
                {t('transcriptions.select_title', { defaultValue: 'Select a transcript' })}
              </h2>
              <p className="m-0 max-w-[320px] text-sm leading-6">
                {t('transcriptions.select_desc', {
                  defaultValue:
                    'Choose an item from your library or import an audio file to begin.',
                })}
              </p>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
