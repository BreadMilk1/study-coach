<script setup lang="ts">
import { computed } from 'vue'
import { ArrowLeftRight } from 'lucide-vue-next'
import { useSettings, type Mode } from '../stores/settings'

const props = defineProps<{
  mode: Mode
  defaultMode: Mode
  disabled?: boolean
}>()
const emit = defineEmits<{ (e: 'toggle'): void }>()

const settings = useSettings()
const overridden = computed(() => props.mode !== props.defaultMode)
const tooltip = computed(() => {
  if (settings.toolCapable === false) return 'Agent loop unavailable — model does not support tool calling'
  if (overridden.value) return `Overridden — default is ${props.defaultMode}`
  return `Default mode for this view`
})
</script>

<template>
  <button
    type="button"
    :aria-pressed="overridden"
    :disabled="settings.toolCapable === false"
    @click="emit('toggle')"
    class="inline-flex items-center gap-2 rounded-full border border-primary-ring bg-primary-bg px-3 py-1 text-xs font-mono text-primary hover:bg-primary/20 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
    :title="tooltip"
  >
    {{ mode }}
    <ArrowLeftRight class="w-3 h-3 opacity-60" />
  </button>
</template>
