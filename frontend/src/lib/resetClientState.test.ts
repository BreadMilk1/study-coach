import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('../stores/settings', () => ({
  authHeaders: vi.fn(() => ({})),
  getAccessToken: vi.fn(() => Promise.resolve('test-token')),
  llmHeaders: vi.fn(() => ({})),
}))

import { useActivity } from '../stores/activity'
import { useChat } from '../stores/chat'
import { useDocuments } from '../stores/documents'
import { useMastery } from '../stores/mastery'
import { useMistakes } from '../stores/mistakes'
import { usePlan } from '../stores/plan'
import { useQuiz } from '../stores/quiz'
import { memoryStorage } from '../test/memoryStorage'
import { streamChat } from './api'
import { CHAT_SESSION_KEY, clearStoredChatSessionId } from './dataLifecycle'
import { resetClientLearningState, type ClientLearningStores } from './resetClientState'

beforeEach(() => {
  setActivePinia(createPinia())
  vi.stubGlobal('localStorage', memoryStorage())
})

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('stored chat session reset', () => {
  it('clears only the owned chat-session key from browser localStorage', () => {
    const local = memoryStorage({
      [CHAT_SESSION_KEY]: 'session-id',
      'study-coach:settings': '{"model":"gemma"}',
      unrelated: 'keep-me',
    })
    vi.stubGlobal('localStorage', local)

    clearStoredChatSessionId()

    expect(local.getItem(CHAT_SESSION_KEY)).toBeNull()
    expect(local.getItem('study-coach:settings')).toBe('{"model":"gemma"}')
    expect(local.getItem('unrelated')).toBe('keep-me')
  })

  it('does not throw when browser localStorage is unavailable', () => {
    vi.stubGlobal('localStorage', {
      removeItem: () => { throw new Error('storage blocked') },
    })

    expect(() => clearStoredChatSessionId()).not.toThrow()
  })
})

describe('chat data reset', () => {
  it('clears all chat learning state', () => {
    const chat = useChat()
    chat.messages = [{ id: 'message-1', role: 'user', content: 'hello' }]
    chat.streaming = true
    chat.trace = [{ node: 'quiz' }]
    chat.sessionId = 'session-1'
    chat.restoring = true

    chat.resetAfterDataClear()

    expect(chat.messages).toEqual([])
    expect(chat.streaming).toBe(false)
    expect(chat.trace).toEqual([])
    expect(chat.sessionId).toBe('')
    expect(chat.restoring).toBe(false)
  })

  it('does not restore a stale response after data is cleared', async () => {
    let resolveResponse!: (response: Response) => void
    const fetchMock = vi.fn(() => new Promise<Response>((resolve) => {
      resolveResponse = resolve
    }))
    vi.stubGlobal('fetch', fetchMock)
    const chat = useChat()
    chat.sessionId = 'old-session'
    chat.trace = [{ node: 'old-trace' }]
    localStorage.setItem(CHAT_SESSION_KEY, 'old-session')

    const restore = chat.restoreCurrentSession()
    expect(fetchMock).toHaveBeenCalledTimes(1)
    clearStoredChatSessionId()
    chat.resetAfterDataClear()
    resolveResponse({
      ok: true,
      status: 200,
      json: vi.fn().mockResolvedValue({
        session_id: 'old-session',
        messages: [{
          id: 'old-message',
          role: 'assistant',
          content: 'stale',
          citations: [],
          agent_run: null,
        }],
      }),
    } as unknown as Response)

    await restore

    expect(chat.sessionId).toBe('')
    expect(chat.messages).toEqual([])
    expect(chat.trace).toEqual([])
    expect(chat.restoring).toBe(false)
    expect(localStorage.getItem(CHAT_SESSION_KEY)).toBeNull()
  })

  it('drops buffered stream events after data is cleared', async () => {
    let resolveRead!: (result: { value: Uint8Array; done: false }) => void
    const read = vi.fn()
      .mockImplementationOnce(() => new Promise((resolve) => { resolveRead = resolve }))
      .mockResolvedValueOnce({ value: undefined, done: true })
    const cancel = vi.fn().mockResolvedValue(undefined)
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      body: { getReader: () => ({ read, cancel }) },
    }))
    const chat = useChat()
    const onSession = vi.fn((sessionId: string) => chat.setSessionId(sessionId))
    const onToken = vi.fn((content: string) => {
      chat.messages.push({ id: 'streamed', role: 'assistant', content })
    })
    const onTrace = vi.fn((step: unknown) => chat.trace.push(step))
    const onError = vi.fn()

    const streaming = streamChat('question', {}, { onSession, onToken, onTrace, onError })
    await vi.waitFor(() => expect(read).toHaveBeenCalledTimes(1))
    clearStoredChatSessionId()
    chat.resetAfterDataClear()
    resolveRead({
      value: new TextEncoder().encode([
        'data: {"type":"session","session_id":"stale-session"}',
        'data: {"type":"token","text":"stale-token"}',
        'data: {"type":"trace","node":"stale-trace"}',
        'data: {"type":"done"}',
        '',
      ].join('\n\n')),
      done: false,
    })

    await streaming

    expect(onSession).not.toHaveBeenCalled()
    expect(onToken).not.toHaveBeenCalled()
    expect(onTrace).not.toHaveBeenCalled()
    expect(onError).not.toHaveBeenCalled()
    expect(cancel).toHaveBeenCalledTimes(1)
    expect(chat.sessionId).toBe('')
    expect(chat.messages).toEqual([])
    expect(chat.trace).toEqual([])
    expect(localStorage.getItem(CHAT_SESSION_KEY)).toBeNull()
  })
})

describe('plan data reset', () => {
  it('does not continue to events after a stale current-plan response', async () => {
    let resolveCurrent!: (response: Response) => void
    const fetchMock = vi.fn()
      .mockImplementationOnce(() => new Promise<Response>((resolve) => {
        resolveCurrent = resolve
      }))
      .mockResolvedValue({
        ok: true,
        status: 200,
        json: vi.fn().mockResolvedValue([]),
      })
    vi.stubGlobal('fetch', fetchMock)
    const plan = usePlan()

    const staleFetch = plan.fetch()
    expect(plan.loading).toBe(true)
    useChat().resetAfterDataClear()
    plan.resetAfterDataClear()
    plan.plan = {
      plan_id: 'new-plan',
      goal_id: 'new-goal',
      goal_title: 'New goal',
      milestones: [],
      updated_at: '2026-07-20T01:00:00Z',
    }
    plan.events = [{
      id: 'new-event',
      plan_id: 'new-plan',
      milestone_id: null,
      actor: 'user',
      action: 'created',
      before_json: null,
      after_json: null,
      reason: null,
      created_at: '2026-07-20T01:00:00Z',
    }]
    plan.error = 'new refresh pending'
    plan.loading = true
    resolveCurrent({
      ok: true,
      status: 200,
      json: vi.fn().mockResolvedValue({
        plan_id: 'old-plan',
        goal_id: 'old-goal',
        goal_title: 'Old goal',
        milestones: [],
        updated_at: '2026-07-20T00:00:00Z',
      }),
    } as unknown as Response)

    await expect(staleFetch).resolves.toBe(true)

    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(plan.plan?.plan_id).toBe('new-plan')
    expect(plan.events.map(event => event.id)).toEqual(['new-event'])
    expect(plan.error).toBe('new refresh pending')
    expect(plan.loading).toBe(true)
  })

  it('does not apply stale events from an in-flight plan refresh', async () => {
    let resolveEvents!: (response: Response) => void
    const fetchMock = vi.fn()
      .mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: vi.fn().mockResolvedValue({
          plan_id: 'old-plan',
          goal_id: 'old-goal',
          goal_title: 'Old goal',
          milestones: [],
          updated_at: '2026-07-20T00:00:00Z',
        }),
      })
      .mockImplementationOnce(() => new Promise<Response>((resolve) => {
        resolveEvents = resolve
      }))
    vi.stubGlobal('fetch', fetchMock)
    const plan = usePlan()

    const staleFetch = plan.fetch()
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2))
    useChat().resetAfterDataClear()
    plan.resetAfterDataClear()
    plan.plan = {
      plan_id: 'new-plan',
      goal_id: 'new-goal',
      goal_title: 'New goal',
      milestones: [],
      updated_at: '2026-07-20T01:00:00Z',
    }
    plan.events = [{
      id: 'new-event',
      plan_id: 'new-plan',
      milestone_id: null,
      actor: 'user',
      action: 'created',
      before_json: null,
      after_json: null,
      reason: null,
      created_at: '2026-07-20T01:00:00Z',
    }]
    plan.error = 'new refresh pending'
    plan.loading = true
    resolveEvents({
      ok: true,
      status: 200,
      json: vi.fn().mockResolvedValue([{
        id: 'old-event',
        plan_id: 'old-plan',
        milestone_id: null,
        actor: 'user',
        action: 'created',
        before_json: null,
        after_json: null,
        reason: null,
        created_at: '2026-07-20T00:00:00Z',
      }]),
    } as unknown as Response)

    await expect(staleFetch).resolves.toBe(true)

    expect(plan.plan?.plan_id).toBe('new-plan')
    expect(plan.events.map(event => event.id)).toEqual(['new-event'])
    expect(plan.error).toBe('new refresh pending')
    expect(plan.loading).toBe(true)
  })

  it('does not apply stale events from an independent events refresh', async () => {
    let resolveEvents!: (response: Response) => void
    vi.stubGlobal('fetch', vi.fn(() => new Promise<Response>((resolve) => {
      resolveEvents = resolve
    })))
    const plan = usePlan()
    plan.plan = {
      plan_id: 'old-plan',
      goal_id: 'old-goal',
      goal_title: 'Old goal',
      milestones: [],
      updated_at: '2026-07-20T00:00:00Z',
    }

    const staleFetch = plan.fetchEvents()
    useChat().resetAfterDataClear()
    plan.resetAfterDataClear()
    plan.plan = {
      plan_id: 'new-plan',
      goal_id: 'new-goal',
      goal_title: 'New goal',
      milestones: [],
      updated_at: '2026-07-20T01:00:00Z',
    }
    plan.events = [{
      id: 'new-event',
      plan_id: 'new-plan',
      milestone_id: null,
      actor: 'user',
      action: 'created',
      before_json: null,
      after_json: null,
      reason: null,
      created_at: '2026-07-20T01:00:00Z',
    }]
    resolveEvents({
      ok: true,
      status: 200,
      json: vi.fn().mockResolvedValue([{
        id: 'old-event',
        plan_id: 'old-plan',
        milestone_id: null,
        actor: 'user',
        action: 'created',
        before_json: null,
        after_json: null,
        reason: null,
        created_at: '2026-07-20T00:00:00Z',
      }]),
    } as unknown as Response)

    await staleFetch

    expect(plan.plan?.plan_id).toBe('new-plan')
    expect(plan.events.map(event => event.id)).toEqual(['new-event'])
  })

  it('clears every owned learning-state field', () => {
    const plan = usePlan()
    plan.plan = {
      plan_id: 'plan-1',
      goal_id: 'goal-1',
      goal_title: 'Learn',
      milestones: [],
      updated_at: '2026-07-20T00:00:00Z',
    }
    plan.events = [{
      id: 'event-1',
      plan_id: 'plan-1',
      milestone_id: null,
      actor: 'user',
      action: 'updated',
      before_json: null,
      after_json: null,
      reason: null,
      created_at: '2026-07-20T00:00:00Z',
    }]
    plan.lastValidationHint = { show_quick_quiz: true, topic: 'LLMs', reason: 'validate' }
    plan.loading = true
    plan.updatingMilestoneId = 'milestone-1'
    plan.error = 'stale error'
    plan.noActive = true
    plan.mindmapMermaid = 'graph TD; A-->B'

    plan.resetAfterDataClear()

    expect(plan.plan).toBeNull()
    expect(plan.events).toEqual([])
    expect(plan.lastValidationHint).toBeNull()
    expect(plan.loading).toBe(false)
    expect(plan.updatingMilestoneId).toBeNull()
    expect(plan.error).toBeNull()
    expect(plan.noActive).toBe(false)
    expect(plan.mindmapMermaid).toBeNull()
  })

  it('treats a missing current plan as refreshed empty state', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: false, status: 404 }))
    const plan = usePlan()
    plan.mindmapMermaid = 'graph TD; stale-->map'

    await expect(plan.fetch()).resolves.toBe(true)

    expect(plan.noActive).toBe(true)
    expect(plan.plan).toBeNull()
    expect(plan.events).toEqual([])
    expect(plan.lastValidationHint).toBeNull()
    expect(plan.mindmapMermaid).toBeNull()
  })
})

describe('server-backed store data reset', () => {
  it('does not let a stale documents request overwrite refreshed state', async () => {
    let resolveResponse!: (response: Response) => void
    vi.stubGlobal('fetch', vi.fn(() => new Promise<Response>((resolve) => {
      resolveResponse = resolve
    })))
    const documents = useDocuments()

    const staleFetch = documents.fetch()
    expect(documents.loading).toBe(true)
    useChat().resetAfterDataClear()
    documents.resetAfterDataClear()
    documents.docs = [{ id: 'new-doc', filename: 'new.pdf', chunks_count: 1 }]
    documents.error = 'new refresh pending'
    documents.loading = true
    resolveResponse({
      ok: true,
      status: 200,
      json: vi.fn().mockResolvedValue([
        { id: 'old-doc', filename: 'old.pdf', chunks_count: 9 },
      ]),
    } as unknown as Response)

    await expect(staleFetch).resolves.toBe(true)

    expect(documents.docs).toEqual([
      { id: 'new-doc', filename: 'new.pdf', chunks_count: 1 },
    ])
    expect(documents.error).toBe('new refresh pending')
    expect(documents.loading).toBe(true)
  })

  it('clears documents before they are refreshed', () => {
    const documents = useDocuments()
    documents.docs = [{ id: 'doc-1', filename: 'stale.pdf', chunks_count: 3 }]
    documents.loading = true
    documents.error = 'stale error'

    documents.resetAfterDataClear()

    expect(documents.docs).toEqual([])
    expect(documents.loading).toBe(false)
    expect(documents.error).toBeNull()
  })

  it('clears mistakes before they are refreshed', () => {
    const mistakes = useMistakes()
    mistakes.items = [{
      mistake_id: 'mistake-1',
      question: {
        id: 'question-1',
        prompt: 'stale question',
        options: ['A', 'B'],
        answer: 'A',
        explanation: 'stale explanation',
      },
      due_at: '2026-07-20T00:00:00Z',
      srs_interval_days: 1,
      srs_ease: 2.5,
      topic_name: 'stale topic',
    }]
    mistakes.loading = true
    mistakes.error = 'stale error'

    mistakes.resetAfterDataClear()

    expect(mistakes.items).toEqual([])
    expect(mistakes.loading).toBe(false)
    expect(mistakes.error).toBeNull()
  })

  it('does not let a stale mistakes failure overwrite refreshed state', async () => {
    let rejectResponse!: (error: Error) => void
    vi.stubGlobal('fetch', vi.fn(() => new Promise<Response>((_resolve, reject) => {
      rejectResponse = reject
    })))
    const mistakes = useMistakes()

    const staleFetch = mistakes.fetch()
    expect(mistakes.loading).toBe(true)
    useChat().resetAfterDataClear()
    mistakes.resetAfterDataClear()
    mistakes.items = [{
      mistake_id: 'new-mistake',
      question: {
        id: 'new-question',
        prompt: 'new question',
        options: ['A', 'B'],
        answer: 'A',
        explanation: 'new explanation',
      },
      due_at: '2026-07-20T01:00:00Z',
      srs_interval_days: 2,
      srs_ease: 2.6,
      topic_name: 'new topic',
    }]
    mistakes.error = 'new refresh pending'
    mistakes.loading = true
    rejectResponse(new Error('old request failed'))

    await expect(staleFetch).resolves.toBe(false)

    expect(mistakes.items.map(item => item.mistake_id)).toEqual(['new-mistake'])
    expect(mistakes.error).toBe('new refresh pending')
    expect(mistakes.loading).toBe(true)
  })

  it('clears mastery before it is refreshed', () => {
    const mastery = useMastery()
    mastery.data = {
      scores: [{
        topic_id: 'topic-1',
        topic_name: 'stale topic',
        score: 0.7,
        last_reviewed: '2026-07-20T00:00:00Z',
      }],
      weak_topics: ['stale topic'],
      overdue_milestones_count: 2,
      streak_days: 4,
      coverage: 0.5,
    }
    mastery.loading = true
    mastery.error = 'stale error'

    mastery.resetAfterDataClear()

    expect(mastery.data).toEqual({
      scores: [],
      weak_topics: [],
      overdue_milestones_count: 0,
      streak_days: 0,
      coverage: 0,
    })
    expect(mastery.loading).toBe(false)
    expect(mastery.error).toBeNull()
  })

  it('does not let a stale mastery request overwrite refreshed state', async () => {
    let resolveResponse!: (response: Response) => void
    vi.stubGlobal('fetch', vi.fn(() => new Promise<Response>((resolve) => {
      resolveResponse = resolve
    })))
    const mastery = useMastery()

    const staleFetch = mastery.fetch()
    expect(mastery.loading).toBe(true)
    useChat().resetAfterDataClear()
    mastery.resetAfterDataClear()
    mastery.data = {
      scores: [],
      weak_topics: ['new topic'],
      overdue_milestones_count: 1,
      streak_days: 2,
      coverage: 0.3,
    }
    mastery.error = 'new refresh pending'
    mastery.loading = true
    resolveResponse({
      ok: true,
      status: 200,
      json: vi.fn().mockResolvedValue({
        scores: [],
        weak_topics: ['old topic'],
        overdue_milestones_count: 9,
        streak_days: 9,
        coverage: 0.9,
      }),
    } as unknown as Response)

    await expect(staleFetch).resolves.toBe(true)

    expect(mastery.data).toEqual({
      scores: [],
      weak_topics: ['new topic'],
      overdue_milestones_count: 1,
      streak_days: 2,
      coverage: 0.3,
    })
    expect(mastery.error).toBe('new refresh pending')
    expect(mastery.loading).toBe(true)
  })

  it('reports non-expected refresh failures from every server-backed store', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: false, status: 500 }))

    await expect(Promise.all([
      useDocuments().fetch(),
      usePlan().fetch(),
      useMistakes().fetch(),
      useMastery().fetch(),
      useActivity().fetch(),
    ])).resolves.toEqual([false, false, false, false, false])
  })

  it('reports successful refreshes from every server-backed store', async () => {
    vi.stubGlobal('fetch', vi.fn(async (input: string | URL | Request) => {
      const path = String(input)
      const body = path === '/api/users/me/stats'
        ? {
            streak_days: 0,
            coverage: 0,
            total_sessions: 0,
            last_active_date: null,
            activity_daily: [],
          }
        : path === '/api/documents' || path.includes('/events') || path.includes('/mistakes/')
          ? []
          : path === '/api/mastery'
          ? { scores: [], weak_topics: [], overdue_milestones_count: 0, streak_days: 0, coverage: 0 }
          : {
              plan_id: 'plan-1',
              goal_id: 'goal-1',
              goal_title: 'Learn',
              milestones: [],
              updated_at: '2026-07-20T00:00:00Z',
            }
      return {
        ok: true,
        status: 200,
        json: vi.fn().mockResolvedValue(body),
      }
    }))

    await expect(Promise.all([
      useDocuments().fetch(),
      usePlan().fetch(),
      useMistakes().fetch(),
      useMastery().fetch(),
      useActivity().fetch(),
    ])).resolves.toEqual([true, true, true, true, true])
  })

  it('rejects safely without retaining stale data when real-store refreshes fail', async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: false, status: 500 })
    vi.stubGlobal('fetch', fetchMock)
    const chat = useChat()
    const quiz = useQuiz()
    const documents = useDocuments()
    const plan = usePlan()
    const mistakes = useMistakes()
    const mastery = useMastery()
    const activity = useActivity()
    chat.messages = [{ id: 'message-1', role: 'user', content: 'stale chat' }]
    quiz.raw = 'stale quiz'
    documents.docs = [{ id: 'doc-1', filename: 'stale.pdf', chunks_count: 3 }]
    plan.mindmapMermaid = 'graph TD; stale-->map'
    mistakes.items = [{
      mistake_id: 'mistake-1',
      question: {
        id: 'question-1',
        prompt: 'stale question',
        options: [],
        answer: 'A',
        explanation: 'stale explanation',
      },
      due_at: '2026-07-20T00:00:00Z',
      srs_interval_days: 1,
      srs_ease: 2.5,
      topic_name: 'stale topic',
    }]
    mastery.data.weak_topics = ['stale topic']
    activity.days = [{ date: '2026-07-27', count: 4 }]

    await expect(resetClientLearningState({
      clearChatSession: clearStoredChatSessionId,
      chat,
      quiz,
      documents,
      plan,
      mistakes,
      mastery,
      activity,
    })).rejects.toThrow('Client learning data refresh failed.')

    expect(fetchMock).toHaveBeenCalledTimes(5)
    expect(chat.messages).toEqual([])
    expect(quiz.raw).toBe('')
    expect(documents.docs).toEqual([])
    expect(plan.mindmapMermaid).toBeNull()
    expect(mistakes.items).toEqual([])
    expect(mastery.data.weak_topics).toEqual([])
    expect(activity.days).toEqual([])
  })
})

describe('resetClientLearningState', () => {
  it('clears chat and quiz before refreshing every server-backed store', async () => {
    const calls: string[] = []
    const dependencies = {
      clearChatSession: () => calls.push('chat-key'),
      chat: { resetAfterDataClear: () => calls.push('chat') },
      quiz: { reset: () => calls.push('quiz') },
      documents: {
        resetAfterDataClear: () => calls.push('documents-reset'),
        fetch: async () => { calls.push('documents'); return true },
      },
      plan: {
        resetAfterDataClear: () => calls.push('plan-reset'),
        fetch: async () => { calls.push('plan'); return true },
      },
      mistakes: {
        resetAfterDataClear: () => calls.push('mistakes-reset'),
        fetch: async () => { calls.push('mistakes'); return true },
      },
      mastery: {
        resetAfterDataClear: () => calls.push('mastery-reset'),
        fetch: async () => { calls.push('mastery'); return true },
      },
      activity: {
        resetAfterDataClear: () => calls.push('activity-reset'),
        fetch: async () => { calls.push('activity'); return true },
      },
    }

    await resetClientLearningState(dependencies)

    expect(calls).toEqual([
      'chat-key',
      'chat',
      'quiz',
      'documents-reset',
      'plan-reset',
      'mistakes-reset',
      'mastery-reset',
      'activity-reset',
      'documents',
      'plan',
      'mistakes',
      'mastery',
      'activity',
    ])
  })

  it('does not finish until every server-backed refresh finishes', async () => {
    const calls: string[] = []
    let resolveDocuments!: (result: boolean) => void
    let resolvePlan!: (result: boolean) => void
    let resolveMistakes!: (result: boolean) => void
    let resolveMastery!: (result: boolean) => void
    let resolveActivity!: (result: boolean) => void
    const pending = (
      name: string,
      capture: (resolve: (result: boolean) => void) => void,
    ) => new Promise<boolean>((resolve) => {
      calls.push(name)
      capture(resolve)
    })
    const dependencies = {
      clearChatSession: () => calls.push('chat-key'),
      chat: { resetAfterDataClear: () => calls.push('chat') },
      quiz: { reset: () => calls.push('quiz') },
      documents: {
        resetAfterDataClear: () => calls.push('documents-reset'),
        fetch: () => pending('documents', resolve => { resolveDocuments = resolve }),
      },
      plan: {
        resetAfterDataClear: () => calls.push('plan-reset'),
        fetch: () => pending('plan', resolve => { resolvePlan = resolve }),
      },
      mistakes: {
        resetAfterDataClear: () => calls.push('mistakes-reset'),
        fetch: () => pending('mistakes', resolve => { resolveMistakes = resolve }),
      },
      mastery: {
        resetAfterDataClear: () => calls.push('mastery-reset'),
        fetch: () => pending('mastery', resolve => { resolveMastery = resolve }),
      },
      activity: {
        resetAfterDataClear: () => calls.push('activity-reset'),
        fetch: () => pending('activity', resolve => { resolveActivity = resolve }),
      },
    }

    let settled = false
    const reset = resetClientLearningState(dependencies).then(() => { settled = true })

    expect(calls).toEqual([
      'chat-key',
      'chat',
      'quiz',
      'documents-reset',
      'plan-reset',
      'mistakes-reset',
      'mastery-reset',
      'activity-reset',
      'documents',
      'plan',
      'mistakes',
      'mastery',
      'activity',
    ])
    resolveDocuments(true)
    resolvePlan(true)
    resolveMistakes(true)
    resolveMastery(true)
    await Promise.resolve()
    expect(settled).toBe(false)

    resolveActivity(true)
    await reset
    expect(settled).toBe(true)
  })

  it('starts every refresh and propagates a refresh failure', async () => {
    const calls: string[] = []
    const failure = new Error('documents refresh failed')
    const dependencies = {
      clearChatSession: () => calls.push('chat-key'),
      chat: { resetAfterDataClear: () => calls.push('chat') },
      quiz: { reset: () => calls.push('quiz') },
      documents: {
        resetAfterDataClear: () => calls.push('documents-reset'),
        fetch: () => { calls.push('documents'); return Promise.reject(failure) },
      },
      plan: {
        resetAfterDataClear: () => calls.push('plan-reset'),
        fetch: async () => { calls.push('plan'); return true },
      },
      mistakes: {
        resetAfterDataClear: () => calls.push('mistakes-reset'),
        fetch: async () => { calls.push('mistakes'); return true },
      },
      mastery: {
        resetAfterDataClear: () => calls.push('mastery-reset'),
        fetch: async () => { calls.push('mastery'); return true },
      },
      activity: {
        resetAfterDataClear: () => calls.push('activity-reset'),
        fetch: async () => { calls.push('activity'); return true },
      },
    }

    await expect(resetClientLearningState(dependencies)).rejects.toBe(failure)
    expect(calls).toEqual([
      'chat-key',
      'chat',
      'quiz',
      'documents-reset',
      'plan-reset',
      'mistakes-reset',
      'mastery-reset',
      'activity-reset',
      'documents',
      'plan',
      'mistakes',
      'mastery',
      'activity',
    ])
  })

  it('does not clear eval artifacts because learning reset preserves them', async () => {
    const evalReset = vi.fn()
    const dependencies = {
      clearChatSession: () => undefined,
      chat: { resetAfterDataClear: () => undefined },
      quiz: { reset: () => undefined },
      documents: { resetAfterDataClear: () => undefined, fetch: async () => true },
      plan: { resetAfterDataClear: () => undefined, fetch: async () => true },
      mistakes: { resetAfterDataClear: () => undefined, fetch: async () => true },
      mastery: { resetAfterDataClear: () => undefined, fetch: async () => true },
      activity: { resetAfterDataClear: () => undefined, fetch: async () => true },
      eval: { reset: evalReset },
    }

    await resetClientLearningState(dependencies as ClientLearningStores)

    expect(evalReset).not.toHaveBeenCalled()
  })
})
