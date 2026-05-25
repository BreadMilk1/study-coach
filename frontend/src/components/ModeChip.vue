<script setup lang="ts">
import { computed } from 'vue'
import { ArrowLeftRight } from 'lucide-vue-next'
import type { Mode } from '../stores/settings'

const props = defineProps<{
  mode: Mode
  defaultMode: Mode
}>()
const emit = defineEmits<{ (e: 'toggle'): void }>()

const overridden = computed(() => props.mode !== props.defaultMode)
</script>

<template>
  <button
    type="button"
    :aria-pressed="overridden"
    @click="emit('toggle')"
    class="inline-flex items-center gap-2 rounded-full border border-primary-ring bg-primary-bg px-3 py-1 text-xs font-mono text-primary hover:bg-primary/20 transition-colors"
    :title="overridden ? `Overridden — default is ${defaultMode}` : `Default mode for this view`"
  >
    {{ mode }}
    <ArrowLeftRight class="w-3 h-3 opacity-60" />
  </button>
</template>
