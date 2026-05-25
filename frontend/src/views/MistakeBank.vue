<script setup lang="ts">
import { onMounted, computed } from 'vue'
import { useMistakes } from '../stores/mistakes'
import MistakeRow from '../components/MistakeRow.vue'

const store = useMistakes()
onMounted(() => store.fetch())
const dueCount = computed(() => store.due.length)
const trackedCount = computed(() => store.items.length)
</script>

<template>
  <div class="h-full overflow-y-auto p-8">
    <div class="max-w-4xl mx-auto">
      <header class="mb-6">
        <h1 class="text-2xl font-semibold">Mistake Bank</h1>
        <p class="text-sm text-fg-muted mt-1 font-mono">
          {{ dueCount }} due today · {{ trackedCount }} tracked
        </p>
      </header>

      <div v-if="store.loading" class="text-fg-muted text-sm">Loading…</div>
      <div v-else-if="store.error" class="text-sm text-danger">{{ store.error }}</div>
      <div v-else-if="trackedCount === 0" class="rounded-lg border border-border bg-surface p-6 text-center">
        <p class="text-fg-muted">No mistakes tracked yet. Take a quiz to start tracking.</p>
      </div>
      <div v-else class="flex flex-col gap-3">
        <MistakeRow v-for="row in store.items" :key="row.mistake_id" :row="row" />
      </div>
    </div>
  </div>
</template>
