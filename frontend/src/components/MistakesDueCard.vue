<script setup lang="ts">
import { useMistakes } from '../stores/mistakes'
const store = useMistakes()
</script>
<template>
  <section class="rounded-lg border border-border bg-surface p-6">
    <h2 class="text-sm font-semibold mb-4 text-fg-muted uppercase tracking-wider">Mistakes due</h2>
    <div v-if="store.due.length === 0" class="text-fg-dim text-sm">Nothing due.</div>
    <template v-else>
      <div class="text-2xl font-mono mb-3">{{ store.due.length }}</div>
      <ul class="flex flex-col gap-2 text-xs text-fg-muted">
        <li v-for="m in store.due.slice(0, 3)" :key="m.mistake_id" class="truncate">
          {{ m.question.prompt }}
        </li>
      </ul>
      <RouterLink to="/mistakes" class="mt-3 inline-block text-xs text-primary hover:underline">
        Review all →
      </RouterLink>
    </template>
  </section>
</template>
