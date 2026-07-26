<script setup lang="ts">
import { CheckCircle2, Info, X } from 'lucide-vue-next'

import { useNotifications, type NotificationKind } from '../stores/notifications'

const notifications = useNotifications()
const icons: Record<NotificationKind, typeof CheckCircle2> = {
  success: CheckCircle2,
  info: Info,
}
</script>

<template>
  <div
    class="pointer-events-none fixed inset-x-4 top-4 z-[100] flex justify-end"
    role="status"
    aria-live="polite"
    aria-atomic="false"
  >
    <TransitionGroup name="toast" tag="div" class="flex w-full max-w-sm flex-col gap-2">
      <div
        v-for="item in notifications.items"
        :key="item.id"
        class="pointer-events-auto flex items-start gap-3 rounded-lg border border-border-strong bg-surface-2 px-4 py-3 text-fg shadow-xl"
      >
        <component
          :is="icons[item.kind]"
          class="mt-0.5 h-5 w-5 shrink-0"
          :class="item.kind === 'success' ? 'text-success' : 'text-primary-2'"
          aria-hidden="true"
        />
        <p class="min-w-0 flex-1 text-sm leading-5">{{ item.message }}</p>
        <button
          type="button"
          class="-m-1 inline-flex h-7 w-7 shrink-0 items-center justify-center rounded-md text-fg-muted transition-colors hover:bg-white/5 hover:text-fg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-ring"
          aria-label="Dismiss notification"
          @click="notifications.dismiss(item.id)"
        >
          <X class="h-4 w-4" aria-hidden="true" />
        </button>
      </div>
    </TransitionGroup>
  </div>
</template>

<style scoped>
.toast-enter-active,
.toast-leave-active,
.toast-move {
  transition: opacity 160ms ease, transform 160ms ease;
}

.toast-enter-from,
.toast-leave-to {
  opacity: 0;
  transform: translateY(-0.5rem);
}

@media (prefers-reduced-motion: reduce) {
  .toast-enter-active,
  .toast-leave-active,
  .toast-move {
    transition: none;
  }
}
</style>
