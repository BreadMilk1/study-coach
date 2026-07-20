export interface ClientLearningStores {
  clearChatSession: () => void
  chat: { resetAfterDataClear: () => void }
  quiz: { reset: () => void }
  documents: { resetAfterDataClear: () => void; fetch: () => Promise<boolean> }
  plan: { resetAfterDataClear: () => void; fetch: () => Promise<boolean> }
  mistakes: { resetAfterDataClear: () => void; fetch: () => Promise<boolean> }
  mastery: { resetAfterDataClear: () => void; fetch: () => Promise<boolean> }
}

export async function resetClientLearningState(stores: ClientLearningStores): Promise<void> {
  stores.clearChatSession()
  stores.chat.resetAfterDataClear()
  stores.quiz.reset()
  stores.documents.resetAfterDataClear()
  stores.plan.resetAfterDataClear()
  stores.mistakes.resetAfterDataClear()
  stores.mastery.resetAfterDataClear()
  const refreshed = await Promise.all([
    stores.documents.fetch(),
    stores.plan.fetch(),
    stores.mistakes.fetch(),
    stores.mastery.fetch(),
  ])
  if (refreshed.some(result => result === false)) {
    throw new Error('Client learning data refresh failed.')
  }
}
