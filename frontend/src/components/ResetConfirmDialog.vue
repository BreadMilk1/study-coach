<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, ref, watch } from 'vue'
import { AlertTriangle, LoaderCircle, RotateCcw, Trash2 } from 'lucide-vue-next'

import type { DataSummaryDto, ResetScope } from '../lib/api'
import type { LifecyclePhase } from '../stores/dataLifecycle'

const props = defineProps<{
  phase: LifecyclePhase
  scope: ResetScope | null
  summary: DataSummaryDto | null
  error: Error | null
}>()

const emit = defineEmits<{
  cancel: []
  confirmLearning: []
  confirmFactory: []
  retry: []
}>()

const dialog = ref<HTMLDialogElement | null>(null)
const primaryAction = ref<HTMLElement | null>(null)
const factoryText = ref('')
const factoryConfirmed = computed(() => factoryText.value === 'RESET')
const cancelable = computed(() => (
  props.phase === 'confirming_learning' || props.phase === 'confirming_factory'
))
const shouldOpen = computed(() => [
  'confirming_learning',
  'confirming_factory',
  'resetting',
  'reset_error',
  'factory_restarting',
].includes(props.phase))
const failedStage = computed(() => {
  const value = props.error as (Error & { failedStage?: unknown }) | null
  return typeof value?.failedStage === 'string' ? value.failedStage : null
})

async function syncDialog(): Promise<void> {
  await nextTick()
  const element = dialog.value
  if (!element) return
  if (shouldOpen.value && !element.open) element.showModal()
  if (!shouldOpen.value && element.open) element.close()
  if (element.open && [
    'confirming_learning',
    'confirming_factory',
    'reset_error',
  ].includes(props.phase)) primaryAction.value?.focus()
}

function handleCancel(event: Event): void {
  event.preventDefault()
  if (cancelable.value) emit('cancel')
}

watch(() => props.phase, (phase, previous) => {
  if (previous === 'confirming_factory' && phase !== 'confirming_factory') factoryText.value = ''
  void syncDialog()
}, { immediate: true, flush: 'post' })

onBeforeUnmount(() => {
  if (dialog.value?.open) dialog.value.close()
})
</script>

<template>
  <Teleport to="body">
    <dialog
      ref="dialog"
      class="fixed inset-0 m-auto w-[min(36rem,calc(100%-2rem))] rounded-xl border border-border-strong bg-surface-2 p-0 text-fg shadow-2xl"
      aria-labelledby="reset-dialog-title"
      :aria-busy="phase === 'resetting' || phase === 'factory_restarting'"
      @cancel="handleCancel"
      @click.self.prevent
    >
      <section class="p-6 sm:p-7">
        <template v-if="phase === 'confirming_learning'">
          <Trash2 class="mb-5 h-7 w-7 text-danger" aria-hidden="true" />
          <h2 id="reset-dialog-title" class="text-xl font-semibold">
            {{ $t('dataLifecycle.reset.learningTitle') }}
          </h2>
          <p class="mt-3 text-sm leading-6 text-fg-muted">
            {{ $t('dataLifecycle.reset.learningBody') }}
          </p>
          <p class="mt-4 font-mono text-xs text-fg-dim">
            {{ $t('dataLifecycle.reset.counts', {
              documents: summary?.documents ?? 0,
              chunks: summary?.source_chunks ?? 0,
              vectors: summary?.vectors ?? 0,
            }) }}
          </p>
          <div class="mt-7 flex justify-end gap-3">
            <button
              ref="primaryAction"
              type="button"
              class="rounded-md border border-border-strong px-4 py-2 text-sm font-medium text-fg-muted transition-colors hover:bg-white/5 hover:text-fg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-ring"
              autofocus
              @click="emit('cancel')"
            >
              {{ $t('dataLifecycle.actions.cancel') }}
            </button>
            <button
              type="button"
              class="rounded-md bg-danger px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-danger/85 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-danger/50"
              @click="emit('confirmLearning')"
            >
              {{ $t('dataLifecycle.actions.clearLearning') }}
            </button>
          </div>
        </template>

        <template v-else-if="phase === 'confirming_factory'">
          <AlertTriangle class="mb-5 h-7 w-7 text-danger" aria-hidden="true" />
          <h2 id="reset-dialog-title" class="text-xl font-semibold">
            {{ $t('dataLifecycle.reset.factoryTitle') }}
          </h2>
          <p class="mt-3 text-sm leading-6 text-fg-muted">
            {{ $t('dataLifecycle.reset.factoryBody') }}
          </p>
          <label for="factory-reset-confirmation" class="mt-6 block text-sm font-medium text-fg">
            {{ $t('dataLifecycle.reset.factoryLabel') }}
          </label>
          <input
            ref="primaryAction"
            id="factory-reset-confirmation"
            v-model="factoryText"
            type="text"
            autocomplete="off"
            spellcheck="false"
            class="mt-2 w-full rounded-md border border-border-strong bg-bg px-3 py-2 font-mono text-sm text-fg outline-none transition-colors placeholder:text-fg-dim focus:border-danger/60 focus:ring-2 focus:ring-danger/20"
            placeholder="RESET"
            autofocus
          />
          <p class="mt-2 text-xs text-fg-dim">{{ $t('dataLifecycle.reset.factoryHint') }}</p>
          <div class="mt-7 flex justify-end gap-3">
            <button
              type="button"
              class="rounded-md border border-border-strong px-4 py-2 text-sm font-medium text-fg-muted transition-colors hover:bg-white/5 hover:text-fg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-ring"
              @click="emit('cancel')"
            >
              {{ $t('dataLifecycle.actions.cancel') }}
            </button>
            <button
              type="button"
              class="rounded-md bg-danger px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-danger/85 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-danger/50 disabled:cursor-not-allowed disabled:opacity-40"
              :disabled="!factoryConfirmed"
              @click="emit('confirmFactory')"
            >
              {{ $t('dataLifecycle.actions.factoryReset') }}
            </button>
          </div>
        </template>

        <template v-else-if="phase === 'resetting'">
          <LoaderCircle class="mb-5 h-7 w-7 animate-spin text-primary" aria-hidden="true" />
          <h2 id="reset-dialog-title" class="text-xl font-semibold">
            {{ $t('dataLifecycle.reset.resettingTitle') }}
          </h2>
          <p class="mt-2 text-sm leading-6 text-fg-muted" aria-live="polite">
            {{ $t('dataLifecycle.reset.resettingBody') }}
          </p>
          <div class="mt-7 flex justify-end gap-3">
            <button type="button" class="rounded-md border border-border-strong px-4 py-2 text-sm text-fg-dim" disabled>
              {{ $t('dataLifecycle.actions.cancel') }}
            </button>
            <button type="button" class="rounded-md bg-primary px-4 py-2 text-sm font-medium text-white opacity-50" disabled>
              {{ $t('dataLifecycle.reset.resettingAction') }}
            </button>
          </div>
        </template>

        <template v-else-if="phase === 'reset_error'">
          <AlertTriangle class="mb-5 h-7 w-7 text-danger" aria-hidden="true" />
          <h2 id="reset-dialog-title" class="text-xl font-semibold">
            {{ $t('dataLifecycle.reset.errorTitle') }}
          </h2>
          <p v-if="error" class="mt-3 rounded-md bg-danger-bg px-3 py-2 text-sm text-danger" role="alert">
            {{ error.message }}
          </p>
          <p v-if="failedStage" class="mt-2 font-mono text-xs text-fg-dim">
            {{ $t('dataLifecycle.reset.failedStage', { stage: failedStage }) }}
          </p>
          <div class="mt-7 flex justify-end gap-3">
            <button
              ref="primaryAction"
              type="button"
              class="rounded-md bg-primary px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-primary-2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-ring"
              autofocus
              @click="emit('retry')"
            >
              {{ $t('dataLifecycle.actions.retry') }}
            </button>
          </div>
        </template>

        <template v-else-if="phase === 'factory_restarting'">
          <RotateCcw class="mb-5 h-7 w-7 animate-spin text-primary-2" aria-hidden="true" />
          <h2 id="reset-dialog-title" class="text-xl font-semibold" aria-live="polite">
            {{ $t('dataLifecycle.reset.restarting') }}
          </h2>
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
