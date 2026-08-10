<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { getChunk, type ChunkDto } from '../lib/api'
import type { Citation } from '../stores/chat'

const props = defineProps<{
  citation: Citation | null
}>()
const emit = defineEmits<{ (e: 'close'): void }>()

const loading = ref(false)
const error = ref<string | null>(null)
const chunk = ref<ChunkDto | null>(null)

watch(
  () => props.citation?.chunk_id,
  async (chunkId, _prev, onCleanup) => {
    let cancelled = false
    onCleanup(() => {
      cancelled = true
    })
    chunk.value = null
    error.value = null
    if (!chunkId) return
    loading.value = true
    try {
      const result = await getChunk(chunkId)
      if (cancelled) return
      chunk.value = result
    } catch (e: any) {
      if (cancelled) return
      error.value = e?.message ?? 'Failed to load chunk'
    } finally {
      if (!cancelled) loading.value = false
    }
  },
  { immediate: true },
)

const preview = computed(() => {
  const text = chunk.value?.content ?? ''
  const start = props.citation?.span_start ?? 0
  const end = props.citation?.span_end ?? 0
  if (!text || end <= start || start < 0 || end > text.length) {
    return { before: text, highlight: '', after: '' }
  }
  return {
    before: text.slice(0, start),
    highlight: text.slice(start, end),
    after: text.slice(end),
  }
})

let keyTarget: EventTarget | null = null

function onKey(e: KeyboardEvent) {
  if (props.citation != null && e.key === 'Escape') emit('close')
}

onMounted(() => {
  if (typeof window === 'undefined') return
  keyTarget = window
  keyTarget.addEventListener('keydown', onKey as EventListener)
})

onBeforeUnmount(() => {
  keyTarget?.removeEventListener('keydown', onKey as EventListener)
  keyTarget = null
})
</script>

<template>
  <div
    v-if="citation"
    class="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4"
    role="dialog"
    aria-modal="true"
    aria-label="Citation preview"
    @click.self="emit('close')"
  >
    <div class="w-full max-w-lg rounded-lg border border-border bg-surface p-5 shadow-xl">
      <div class="flex items-start justify-between gap-3 mb-3">
        <div>
          <h3 class="text-sm font-semibold text-fg">
            {{ chunk?.source || citation.source || citation.chunk_id }}
          </h3>
          <p class="text-xs text-fg-muted font-mono mt-1">
            p.{{ citation.page }} · {{ citation.chunk_id }}
          </p>
        </div>
        <button
          type="button"
          class="text-fg-muted hover:text-fg text-lg leading-none"
          aria-label="Close"
          @click="emit('close')"
        >&times;</button>
      </div>
      <div v-if="loading" class="text-sm text-fg-muted">Loading chunk…</div>
      <div v-else-if="error" class="text-sm text-rose-300">{{ error }}</div>
      <p v-else class="text-sm text-fg leading-relaxed whitespace-pre-wrap max-h-80 overflow-y-auto">
        <span>{{ preview.before }}</span>
        <mark v-if="preview.highlight" class="bg-indigo-500/30 text-fg rounded-sm px-0.5">{{ preview.highlight }}</mark>
        <span>{{ preview.after }}</span>
      </p>
    </div>
  </div>
</template>
