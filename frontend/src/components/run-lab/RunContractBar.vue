<script setup lang="ts">
defineProps<{
  task?: string
  corpus?: string
  prompt?: string
  model?: string
  runProfile?: string
  budget?: string
  compatibility?: string
}>()
</script>

<template>
  <section class="border-b border-border bg-surface px-4 py-3 md:px-6">
    <h2 class="text-xs uppercase tracking-wider text-fg-dim mb-2">{{ $t('runLab.contract') }}</h2>
    <dl class="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm min-w-0">
      <div class="min-w-0">
        <dt class="text-fg-dim">Task</dt>
        <dd class="font-mono truncate">{{ task || '—' }}</dd>
      </div>
      <div class="min-w-0">
        <dt class="text-fg-dim">Corpus</dt>
        <dd class="font-mono truncate">{{ corpus || '—' }}</dd>
      </div>
      <div class="min-w-0">
        <dt class="text-fg-dim">Prompt</dt>
        <dd class="font-mono truncate">{{ prompt || '—' }}</dd>
      </div>
      <div class="min-w-0">
        <dt class="text-fg-dim">Model</dt>
        <dd class="font-mono truncate">{{ model || '—' }}</dd>
      </div>
      <div class="min-w-0">
        <dt class="text-fg-dim">RunProfile</dt>
        <dd class="font-mono truncate">{{ runProfile || '—' }}</dd>
      </div>
      <div class="min-w-0">
        <dt class="text-fg-dim">Budget</dt>
        <dd class="font-mono truncate">{{ budget || '—' }}</dd>
      </div>
      <div class="min-w-0 col-span-2">
        <dt class="text-fg-dim">Compatibility</dt>
        <dd>{{ compatibility || '—' }}</dd>
      </div>
    </dl>
  </section>
</template>
