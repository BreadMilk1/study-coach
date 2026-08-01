export const DATA_LIFECYCLE_CHANNEL = 'study-coach:data-lifecycle'

export interface ResetBroadcast {
  type: 'reset-completed'
  scope: 'learning' | 'factory'
  epoch: number
}

function isResetBroadcast(value: unknown): value is ResetBroadcast {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) return false
  const candidate = value as Record<string, unknown>
  const keys = Object.keys(candidate)
  return keys.length === 3
    && keys.includes('type')
    && keys.includes('scope')
    && keys.includes('epoch')
    && candidate.type === 'reset-completed'
    && (candidate.scope === 'learning' || candidate.scope === 'factory')
    && typeof candidate.epoch === 'number'
    && Number.isSafeInteger(candidate.epoch)
    && candidate.epoch >= 0
}

export function createDataLifecycleChannel(
  onReset: (message: ResetBroadcast) => void | Promise<void>,
  factory: (name: string) => BroadcastChannel = name => new BroadcastChannel(name),
  onError: (error: unknown) => void | Promise<void> = () => undefined,
) {
  const channel = factory(DATA_LIFECYCLE_CHANNEL)
  const reportError = (error: unknown) => {
    try {
      void Promise.resolve(onError(error)).catch(() => undefined)
    } catch {
      // Error reporting must not escape the BroadcastChannel event boundary.
    }
  }
  channel.onmessage = event => {
    if (!isResetBroadcast(event.data)) return
    try {
      void Promise.resolve(onReset(event.data)).catch(reportError)
    } catch (error) {
      reportError(error)
    }
  }
  return {
    publish(scope: ResetBroadcast['scope']) {
      channel.postMessage({
        type: 'reset-completed',
        scope,
        epoch: Date.now(),
      } satisfies ResetBroadcast)
    },
    close() {
      channel.close()
    },
  }
}
