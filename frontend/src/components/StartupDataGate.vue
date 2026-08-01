<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, ref, watch } from 'vue'
import { AlertTriangle, Database, LoaderCircle, RefreshCw } from 'lucide-vue-next'

import type { DataSummaryDto } from '../lib/api'
import type { LifecyclePhase } from '../stores/dataLifecycle'

const props = withDefaults(defineProps<{
  phase: LifecyclePhase
  summary: DataSummaryDto | null
  error: Error | null
  pending?: boolean
}>(), {
  pending: false,
})

const emit = defineEmits<{
  continue: []
  continueWithoutClearing: []
  startFresh: []
  retry: []
  acknowledgeExternal: []
}>()

const dialog = ref<HTMLDialogElement | null>(null)
const primaryAction = ref<HTMLButtonElement | null>(null)
const shouldOpen = computed(() => {
  if (props.phase === 'choice_required' && props.summary?.reset_enabled === false) return false
  return ['checking', 'choice_required', 'inspection_error', 'external_reset'].includes(props.phase)
})

async function syncDialog(): Promise<void> {
  await nextTick()
  const element = dialog.value
  if (!element) return
  if (shouldOpen.value && !element.open) element.showModal()
  if (!shouldOpen.value && element.open) element.close()
  if (element.open && props.phase !== 'checking' && !primaryAction.value?.disabled) {
    primaryAction.value?.focus()
  }
}

function preventCancel(event: Event): void {
  event.preventDefault()
}

watch(
  () => [props.phase, props.pending, props.summary?.reset_enabled] as const,
  syncDialog,
  { immediate: true, flush: 'post' },
)

onBeforeUnmount(() => {
  if (dialog.value?.open) dialog.value.close()
})
</script>

<template>
  <Teleport to="body">
    <dialog
      ref="dialog"
      class="fixed inset-0 m-auto w-[min(32rem,calc(100%-2rem))] rounded-xl border border-border-strong bg-surface-2 p-0 text-fg shadow-2xl"
      aria-labelledby="startup-data-title"
      @cancel="preventCancel"
      @keydown.esc.prevent.stop="preventCancel"
      @click.self.prevent
    >
      <section class="p-6 sm:p-7">
        <template v-if="phase === 'checking'">
          <LoaderCircle class="mb-5 h-7 w-7 animate-spin text-primary" aria-hidden="true" />
          <h2 id="startup-data-title" class="text-xl font-semibold">
            {{ $t('dataLifecycle.startup.checkingTitle') }}
          </h2>
          <p class="mt-2 text-sm leading-6 text-fg-muted" aria-live="polite">
            {{ $t('dataLifecycle.startup.checkingBody') }}
          </p>
        </template>

        <template v-else-if="phase === 'choice_required'">
          <Database class="mb-5 h-7 w-7 text-primary-2" aria-hidden="true" />
          <h2 id="startup-data-title" class="text-xl font-semibold">
            {{ $t('dataLifecycle.startup.choiceTitle') }}
          </h2>
          <p class="mt-2 text-sm leading-6 text-fg-muted">
            {{ $t('dataLifecycle.startup.choiceBody') }}
          </p>
          <p class="mt-4 font-mono text-xs text-fg-dim">
            {{ $t('dataLifecycle.startup.counts', {
              documents: summary?.documents ?? 0,
              chunks: summary?.source_chunks ?? 0,
              sessions: summary?.chat_sessions ?? 0,
            }) }}
          </p>
          <div class="mt-7 flex flex-col-reverse gap-3 sm:flex-row sm:justify-end">
            <button
              v-if="summary?.reset_enabled"
              type="button"
              class="rounded-md border border-danger/40 px-4 py-2 text-sm font-medium text-danger transition-colors hover:bg-danger-bg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-danger/50"
              @click="emit('startFresh')"
            >
              {{ $t('dataLifecycle.startup.startFresh') }}
            </button>
            <button
              ref="primaryAction"
              type="button"
              class="rounded-md bg-primary px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-primary-2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-ring"
              autofocus
              @click="emit('continue')"
            >
              {{ $t('dataLifecycle.startup.continue') }}
            </button>
          </div>
        </template>

        <template v-else-if="phase === 'inspection_error'">
          <AlertTriangle class="mb-5 h-7 w-7 text-warning" aria-hidden="true" />
          <h2 id="startup-data-title" class="text-xl font-semibold">
            {{ $t('dataLifecycle.startup.inspectionErrorTitle') }}
          </h2>
          <p class="mt-2 text-sm leading-6 text-fg-muted">
            {{ $t('dataLifecycle.startup.inspectionErrorBody') }}
          </p>
          <p v-if="error" class="mt-4 rounded-md bg-warning-bg px-3 py-2 text-sm text-warning" role="alert">
            {{ error.message }}
          </p>
          <div class="mt-7 flex flex-col-reverse gap-3 sm:flex-row sm:justify-end">
            <button
              type="button"
              class="rounded-md border border-border-strong px-4 py-2 text-sm font-medium text-fg-muted transition-colors hover:bg-white/5 hover:text-fg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-ring"
              @click="emit('continueWithoutClearing')"
            >
              {{ $t('dataLifecycle.startup.continueWithoutClearing') }}
            </button>
            <button
              ref="primaryAction"
              type="button"
              class="inline-flex items-center justify-center gap-2 rounded-md bg-primary px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-primary-2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-ring"
              autofocus
              @click="emit('retry')"
            >
              <RefreshCw class="h-4 w-4" aria-hidden="true" />
              {{ $t('dataLifecycle.actions.retry') }}
            </button>
          </div>
        </template>

        <template v-else-if="phase === 'external_reset'">
          <RefreshCw class="mb-5 h-7 w-7 text-primary-2" :class="{ 'animate-spin': pending }" aria-hidden="true" />
          <h2 id="startup-data-title" class="text-xl font-semibold">
            {{ $t('dataLifecycle.startup.externalTitle') }}
          </h2>
          <p class="mt-2 text-sm leading-6 text-fg-muted">
            {{ $t('dataLifecycle.startup.externalBody') }}
          </p>
          <p v-if="error" class="mt-4 rounded-md bg-warning-bg px-3 py-2 text-sm text-warning" role="alert">
            {{ error.message }}
          </p>
          <div class="mt-7 flex justify-end">
            <button
              ref="primaryAction"
              type="button"
              class="rounded-md bg-primary px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-primary-2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-ring disabled:cursor-wait disabled:opacity-50"
              :disabled="pending"
              autofocus
              @click="emit('acknowledgeExternal')"
            >
              {{ pending
                ? $t('dataLifecycle.startup.refreshing')
                : $t('dataLifecycle.startup.continue') }}
            </button>
          </div>
        </template>
      </section>
    </dialog>
  </Teleport>
</template>

<style scoped>
dialog::backdrop {
  background: rgb(3 6 18 / 78%);
  backdrop-filter: blur(3px);
}
</style>
