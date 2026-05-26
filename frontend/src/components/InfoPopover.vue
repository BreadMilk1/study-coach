<script setup lang="ts">
import { ref } from 'vue'
import { HelpCircle } from 'lucide-vue-next'

defineProps<{ title: string }>()

const open = ref(false)

function toggle() { open.value = !open.value }
function close() { open.value = false }

// Click outside closes
function onBlur(e: FocusEvent) {
  if (!(e.currentTarget as HTMLElement)?.contains(e.relatedTarget as HTMLElement)) {
    close()
  }
}
</script>

<template>
  <div class="relative inline-flex" tabindex="-1" @focusout="onBlur">
    <button
      type="button"
      @click="toggle"
      class="inline-flex items-center justify-center w-6 h-6 rounded-full text-fg-muted hover:text-fg hover:bg-white/10 transition-colors"
      :title="title"
    >
      <HelpCircle class="w-4 h-4" />
    </button>
    <div
      v-if="open"
      class="absolute top-8 right-0 z-50 w-72 rounded-lg border border-white/15 bg-surface p-4 shadow-xl text-sm leading-relaxed"
    >
      <div class="flex items-start justify-between mb-2">
        <span class="font-semibold text-fg">{{ title }}</span>
        <button @click="close" class="text-fg-muted hover:text-fg ml-2 shrink-0">&times;</button>
      </div>
      <div class="text-fg-muted space-y-2">
        <slot />
      </div>
    </div>
  </div>
</template>
