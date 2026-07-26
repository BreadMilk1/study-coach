import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  DATA_LIFECYCLE_CHANNEL,
  createDataLifecycleChannel,
  type ResetBroadcast,
} from './dataLifecycleChannel'

class FakeChannel {
  onmessage: ((event: MessageEvent) => void) | null = null
  readonly posted: unknown[] = []
  readonly name: string
  closed = false

  constructor(name: string) {
    this.name = name
  }

  postMessage(value: unknown) {
    this.posted.push(value)
  }

  close() {
    this.closed = true
  }

  receive(value: unknown) {
    this.onmessage?.({ data: value } as MessageEvent)
  }
}

function harness(
  onReset: (message: ResetBroadcast) => void | Promise<void> = vi.fn(),
  onError: (error: unknown) => void | Promise<void> = vi.fn(),
) {
  let fake!: FakeChannel
  const channel = createDataLifecycleChannel(onReset, name => {
    fake = new FakeChannel(name)
    return fake as unknown as BroadcastChannel
  }, onError)
  return { channel, fake, onReset, onError }
}

afterEach(() => {
  vi.restoreAllMocks()
})

describe('data lifecycle channel', () => {
  it('publishes only a typed reset-completed envelope on the fixed channel', () => {
    vi.spyOn(Date, 'now').mockReturnValue(1234)
    const { channel, fake } = harness()

    channel.publish('learning')

    expect(fake.name).toBe(DATA_LIFECYCLE_CHANNEL)
    expect(fake.posted).toEqual([{
      type: 'reset-completed',
      scope: 'learning',
      epoch: 1234,
    }])
  })

  it('delivers a valid reset-completed message', () => {
    const { fake, onReset } = harness()
    const message: ResetBroadcast = {
      type: 'reset-completed',
      scope: 'factory',
      epoch: 42,
    }

    fake.receive(message)

    expect(onReset).toHaveBeenCalledOnce()
    expect(onReset).toHaveBeenCalledWith(message)
  })

  it.each([
    null,
    undefined,
    'reset-completed',
    7,
    [],
    {},
    { type: 'other', scope: 'learning', epoch: 1 },
    { type: 'reset-completed', scope: 'other', epoch: 1 },
    { type: 'reset-completed', scope: 'learning', epoch: '1' },
    { type: 'reset-completed', scope: 'learning', epoch: Number.NaN },
    { type: 'reset-completed', scope: 'learning', epoch: Number.POSITIVE_INFINITY },
    { type: 'reset-completed', scope: 'learning', epoch: -1 },
    { type: 'reset-completed', scope: 'learning', epoch: 1.5 },
    { type: 'reset-completed', scope: 'learning', epoch: 1, token: 'secret' },
    { type: 'reset-completed', scope: 'learning', epoch: 1, deleted: { documents: 1 } },
    { type: 'reset-completed', scope: 'learning', epoch: 1, extra: true },
  ])('rejects invalid channel payload %#', value => {
    const { fake, onReset } = harness()

    fake.receive(value)

    expect(onReset).not.toHaveBeenCalled()
  })

  it('closes the underlying channel', () => {
    const { channel, fake } = harness()

    channel.close()

    expect(fake.closed).toBe(true)
  })

  it('isolates synchronous handler failures through onError', () => {
    const failure = new Error('sync handler failed')
    const onError = vi.fn()
    const { fake } = harness(() => { throw failure }, onError)

    expect(() => fake.receive({
      type: 'reset-completed',
      scope: 'learning',
      epoch: 1,
    })).not.toThrow()
    expect(onError).toHaveBeenCalledWith(failure)
  })

  it('isolates asynchronous handler rejection through onError', async () => {
    const failure = new Error('async handler failed')
    const onError = vi.fn()
    const { fake } = harness(async () => { throw failure }, onError)

    fake.receive({ type: 'reset-completed', scope: 'factory', epoch: 2 })

    await vi.waitFor(() => expect(onError).toHaveBeenCalledWith(failure))
  })

  it('swallows onError failures instead of leaking them through the channel event', async () => {
    const { fake } = harness(
      async () => { throw new Error('handler failed') },
      () => { throw new Error('error handler failed') },
    )

    expect(() => fake.receive({
      type: 'reset-completed',
      scope: 'learning',
      epoch: 3,
    })).not.toThrow()
    await Promise.resolve()
  })
})
