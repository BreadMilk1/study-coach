<script setup lang="ts">
defineProps<{ topics: string[] }>()
</script>
<template>
  <section class="rounded-lg border border-border bg-surface p-6">
    <h2 class="text-sm font-semibold mb-4 text-fg-muted uppercase tracking-wider">{{ $t('overview.weakTopics') }}</h2>
    <div v-if="topics.length === 0" class="text-fg-dim text-sm">No weak topics yet.</div>
    <div v-else class="flex flex-wrap gap-2">
      <RouterLink v-for="t in topics" :key="t"
                  :to="{ path: '/quiz', query: { topic: t } }"
                  class="font-mono text-xs px-2 py-1 rounded-md bg-danger-bg text-danger border border-danger/30 hover:bg-danger/20 transition-colors">
        {{ t }}
      </RouterLink>
    </div>
  </section>
</template>
