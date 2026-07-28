export const STARTUP_CHOICE_KEY = 'study-coach:startup-choice-made'
export const CHAT_SESSION_KEY = 'study-coach:current-chat-session-id'

const APP_PREFIX = 'study-coach:'
let learningStateEpoch = 0

export type StartupDecision = 'ready' | 'choice_required'

export function captureLearningStateEpoch(): number {
  return learningStateEpoch
}

export function isLearningStateEpochCurrent(epoch: number): boolean {
  return epoch === learningStateEpoch
}

export function invalidateLearningState(): void {
  learningStateEpoch += 1
}

export function resolveStartupDecision(
  summary: { resetEnabled: boolean; hasLearningData: boolean },
  session: Storage,
): StartupDecision {
  if (!summary.resetEnabled || !summary.hasLearningData) return 'ready'
  return session.getItem(STARTUP_CHOICE_KEY) ? 'ready' : 'choice_required'
}

export function clearLearningBrowserState(local: Storage): void {
  local.removeItem(CHAT_SESSION_KEY)
}

export function clearStoredChatSessionId(): void {
  try { localStorage.removeItem(CHAT_SESSION_KEY) }
  catch { /* storage unavailable */ }
}

export function markStartupChoice(session: Storage): void {
  session.setItem(STARTUP_CHOICE_KEY, 'continue')
}

export function clearStartupChoice(session: Storage): void {
  session.removeItem(STARTUP_CHOICE_KEY)
}

function clearAppState(storage: Storage): void {
  const keys = Array.from({ length: storage.length }, (_, index) => storage.key(index))
  for (const key of keys) {
    if (key?.startsWith(APP_PREFIX)) storage.removeItem(key)
  }
}

export function clearFactorySessionState(session: Storage): void {
  clearAppState(session)
}

export function clearFactoryBrowserState(local: Storage, session: Storage): void {
  clearAppState(local)
  clearFactorySessionState(session)
}
