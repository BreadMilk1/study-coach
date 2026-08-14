import { describe, expect, it } from 'vitest'

import { memoryStorage } from '../test/memoryStorage'
import {
  ACTIVE_LEARNING_RUN_KEY,
  CHAT_SESSION_KEY,
  STARTUP_CHOICE_KEY,
  clearFactoryBrowserState,
  clearFactorySessionState,
  clearLearningBrowserState,
  clearStartupChoice,
  markStartupChoice,
  resolveStartupDecision,
} from './dataLifecycle'

describe('memoryStorage', () => {
  it('implements Storage length and key ordering', () => {
    const storage = memoryStorage({ first: '1', second: '2' })

    expect(storage.length).toBe(2)
    expect(storage.key(0)).toBe('first')
    expect(storage.key(1)).toBe('second')
    expect(storage.key(2)).toBeNull()

    storage.removeItem('first')
    storage.setItem('third', '3')
    expect(storage.key(0)).toBe('second')
    expect(storage.key(1)).toBe('third')

    storage.clear()
    expect(storage.length).toBe(0)
  })
})

describe('resolveStartupDecision', () => {
  it('skips the gate when reset is disabled even if learning data exists', () => {
    const storage = memoryStorage()

    expect(resolveStartupDecision({ resetEnabled: false, hasLearningData: true }, storage)).toBe('ready')
  })

  it('skips the gate when no learning data exists', () => {
    const storage = memoryStorage()

    expect(resolveStartupDecision({ resetEnabled: true, hasLearningData: false }, storage)).toBe('ready')
  })

  it('requires a choice once per tab when reset is enabled and data exists', () => {
    const storage = memoryStorage()

    expect(resolveStartupDecision({ resetEnabled: true, hasLearningData: true }, storage)).toBe('choice_required')
    markStartupChoice(storage)
    expect(storage.getItem(STARTUP_CHOICE_KEY)).toBe('continue')
    expect(resolveStartupDecision({ resetEnabled: true, hasLearningData: true }, storage)).toBe('ready')
    clearStartupChoice(storage)
    expect(resolveStartupDecision({ resetEnabled: true, hasLearningData: true }, storage)).toBe('choice_required')
  })
})

describe('browser state clearing', () => {
  it('learning clear removes only the current chat session key', () => {
    const local = memoryStorage({
      [CHAT_SESSION_KEY]: 'session-id',
      [ACTIVE_LEARNING_RUN_KEY]: 'run-1',
      'study-coach:settings': '{"model":"gemma"}',
      unrelated: 'keep-me',
    })

    clearLearningBrowserState(local)

    expect(local.getItem(CHAT_SESSION_KEY)).toBeNull()
    expect(local.getItem(ACTIVE_LEARNING_RUN_KEY)).toBe('run-1')
    expect(local.getItem('study-coach:settings')).toBe('{"model":"gemma"}')
    expect(local.getItem('unrelated')).toBe('keep-me')
  })

  it('factory clear preserves only the staged recovery fingerprint among app keys', () => {
    const local = memoryStorage({
      'study-coach:settings': '{}',
      [CHAT_SESSION_KEY]: 'session-id',
      [ACTIVE_LEARNING_RUN_KEY]: 'run-1',
      'study-coach:factory-recovery-fingerprint': 'next-fingerprint',
      unrelated: 'local-value',
    })
    const session = memoryStorage({
      [STARTUP_CHOICE_KEY]: 'continue',
      'study-coach:tab-state': 'value',
      unrelated: 'session-value',
    })

    clearFactoryBrowserState(local, session)

    expect(local.length).toBe(2)
    expect(local.getItem('study-coach:factory-recovery-fingerprint')).toBe('next-fingerprint')
    expect(local.getItem(ACTIVE_LEARNING_RUN_KEY)).toBeNull()
    expect(local.getItem('unrelated')).toBe('local-value')
    expect(session.length).toBe(1)
    expect(session.getItem('unrelated')).toBe('session-value')
  })

  it('external factory clear removes only this tab session state', () => {
    const local = memoryStorage({
      'study-coach:settings': '{"accessToken":"new-token"}',
      'study-coach:fingerprint': 'new-fingerprint',
    })
    const session = memoryStorage({
      [STARTUP_CHOICE_KEY]: 'continue',
      'study-coach:tab-state': 'value',
      unrelated: 'session-value',
    })

    clearFactorySessionState(session)

    expect(local.getItem('study-coach:settings')).toContain('new-token')
    expect(local.getItem('study-coach:fingerprint')).toBe('new-fingerprint')
    expect(session.getItem(STARTUP_CHOICE_KEY)).toBeNull()
    expect(session.getItem('study-coach:tab-state')).toBeNull()
    expect(session.getItem('unrelated')).toBe('session-value')
  })
})
