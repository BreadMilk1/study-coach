<script setup lang="ts">
import { ref, onMounted, watch, nextTick } from 'vue'
import { ChevronDown, ChevronRight } from 'lucide-vue-next'

const props = defineProps<{ mermaid: string }>()
const open = ref(true)
const container = ref<HTMLDivElement | null>(null)
const renderError = ref<string | null>(null)

async function render() {
  if (!open.value || !container.value) return
  try {
    const mermaid = (await import('mermaid')).default
    mermaid.initialize({ startOnLoad: false, theme: 'dark', securityLevel: 'loose' })
    container.value.innerHTML = ''
    const { svg } = await mermaid.render(`mm-${Date.now()}`, props.mermaid)
    container.value.innerHTML = svg
    renderError.value = null
  } catch (e: any) {
    renderError.value = e?.message ?? 'render failed'
  }
}

onMounted(render)
watch(() => props.mermaid, () => nextTick(render))
watch(open, render)
</script>

<template>
  <section class="rounded-lg border border-border bg-surface mt-6">
    <button @click="open = !open"
            class="w-full flex items-center gap-2 p-4 text-sm font-medium hover:bg-white/5 transition-colors">
      <component :is="open ? ChevronDown : ChevronRight" class="w-4 h-4" />
      Mindmap
    </button>
    <div v-show="open" class="px-4 pb-4">
      <div v-if="renderError" class="text-xs text-danger font-mono">{{ renderError }}</div>
      <div ref="container" class="overflow-x-auto"></div>
    </div>
  </section>
</template>
