import { defineStore } from 'pinia'
import { ref } from 'vue'

export type NotificationKind = 'success' | 'info'

export interface NotificationItem {
  id: number
  kind: NotificationKind
  message: string
}

export const useNotifications = defineStore('notifications', () => {
  const items = ref<NotificationItem[]>([])
  let nextId = 1

  function dismiss(id: number): void {
    items.value = items.value.filter(item => item.id !== id)
  }

  function push(input: Omit<NotificationItem, 'id'>): number {
    const id = nextId++
    items.value.push({ id, ...input })
    globalThis.setTimeout(() => dismiss(id), 5000)
    return id
  }

  return { items, push, dismiss }
})
