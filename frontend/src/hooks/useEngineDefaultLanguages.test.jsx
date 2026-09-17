import { act, renderHook } from '@testing-library/react';
import { beforeEach, expect, it } from 'vitest';
import { useAppStore } from '../store';
import { useEngineDefaultLanguages } from './useEngineDefaultLanguages';

const engines = (tts, asr, ids = ['xtts-nepali', 'indic-conformer']) => ({
  tts: { active: ids[0], backends: [{ id: ids[0], default_language: tts }] },
  asr: { active: ids[1], backends: [{ id: ids[1], default_language: asr }] },
});

beforeEach(() => {
  useAppStore.setState({ language: 'Auto', dubSourceLangCode: 'auto' });
});

it("replaces Auto with the active engines' default language", () => {
  renderHook(() => useEngineDefaultLanguages(engines('ne', 'ne')));
  expect(useAppStore.getState().language).toBe('Nepali');
  expect(useAppStore.getState().dubSourceLangCode).toBe('ne');
});

it('keeps an explicit choice and ignores engines without a default', () => {
  useAppStore.setState({ language: 'English', dubSourceLangCode: 'hi' });
  renderHook(() => useEngineDefaultLanguages(engines('ne', 'ne')));
  expect(useAppStore.getState().language).toBe('English');
  expect(useAppStore.getState().dubSourceLangCode).toBe('hi');

  useAppStore.setState({ language: 'Auto', dubSourceLangCode: 'auto' });
  renderHook(() => useEngineDefaultLanguages(engines(null, null)));
  expect(useAppStore.getState().language).toBe('Auto');
  expect(useAppStore.getState().dubSourceLangCode).toBe('auto');
});

it('returns an auto-applied Nepali to Auto when switching to an engine without a default', () => {
  const { rerender } = renderHook(({ e }) => useEngineDefaultLanguages(e), {
    initialProps: { e: engines('ne', 'ne') },
  });
  expect(useAppStore.getState().language).toBe('Nepali');

  rerender({ e: engines(null, null, ['kokoro', 'faster-whisper']) });
  expect(useAppStore.getState().language).toBe('Auto');
  expect(useAppStore.getState().dubSourceLangCode).toBe('auto');
});

it('moves an auto-applied language to the next engine default', () => {
  const { rerender } = renderHook(({ e }) => useEngineDefaultLanguages(e), {
    initialProps: { e: engines('ne', 'ne') },
  });
  rerender({ e: engines('hi', 'hi', ['hindi-tts', 'hindi-asr']) });
  expect(useAppStore.getState().language).toBe('Hindi');
  expect(useAppStore.getState().dubSourceLangCode).toBe('hi');
});

it('never overrides a language the user picked after the default was applied', () => {
  const { rerender } = renderHook(({ e }) => useEngineDefaultLanguages(e), {
    initialProps: { e: engines('ne', 'ne') },
  });
  act(() => useAppStore.setState({ language: 'Maithili', dubSourceLangCode: 'mai' }));
  rerender({ e: engines(null, null, ['kokoro', 'faster-whisper']) });
  expect(useAppStore.getState().language).toBe('Maithili');
  expect(useAppStore.getState().dubSourceLangCode).toBe('mai');
});
