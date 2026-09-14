import { renderHook } from '@testing-library/react';
import { beforeEach, expect, it } from 'vitest';
import { useAppStore } from '../store';
import { useEngineDefaultLanguages } from './useEngineDefaultLanguages';

const engines = (tts, asr) => ({
  tts: { active: 'xtts-nepali', backends: [{ id: 'xtts-nepali', default_language: tts }] },
  asr: { active: 'indic-conformer', backends: [{ id: 'indic-conformer', default_language: asr }] },
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
