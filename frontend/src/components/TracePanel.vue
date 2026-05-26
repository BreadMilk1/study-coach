<script setup lang="ts">
import { useChat } from '../stores/chat'

const chat = useChat()
</script>

<template>
  <div v-if="chat.trace.length" class="rounded-lg border border-border bg-surface p-3 text-xs font-mono mt-2">
    <div class="text-fg-muted mb-2 text-[10px] uppercase tracking-wider">Agent Trace</div>
    <div v-for="(step, i) in chat.trace" :key="i"
         class="flex gap-3 py-1 border-b border-border/30 last:border-0">
      <span class="text-primary w-16 shrink-0">{{ step.step }}</span>
      <span class="text-fg">
        <template v-if="step.step === 'router'">
          intent={{ step.intent }}
          <span v-if="step.active_quiz" class="text-fg-dim"> quiz={{ step.active_quiz }}</span>
        </template>
        <template v-else-if="step.step === 'judge'">
          score={{ step.score?.toFixed(2) }}
          <span v-if="step.weak_dims?.length" class="text-warning"> weak: {{ step.weak_dims.join(', ') }}</span>
        </template>
        <template v-else-if="step.step === 'tutor'">
          {{ step.citations_count }} citations
        </template>
      </span>
    </div>
  </div>
</template>
