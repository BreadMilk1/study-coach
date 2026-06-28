<script setup lang="ts">
import { computed } from 'vue'
import { useChat } from '../stores/chat'

const chat = useChat()
const latestAgentRun = computed(() => (
  chat.messages.slice().reverse().find((m) => m.role === 'assistant' && m.agentRun)?.agentRun || null
))
const toolBreakdown = computed(() => {
  const run = latestAgentRun.value
  if (!run) return ''
  return Object.entries(run.tool_call_breakdown)
    .map(([name, count]) => `${name}:${count}`)
    .join(', ')
})
</script>

<template>
  <div v-if="chat.trace.length || latestAgentRun" class="rounded-lg border border-border bg-surface p-3 text-xs font-mono mt-2">
    <div v-if="chat.trace.length">
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
    <div v-if="latestAgentRun" :class="chat.trace.length ? 'mt-3 border-t border-border/40 pt-3' : ''">
      <div class="text-fg-muted mb-2 text-[10px] uppercase tracking-wider">Agent Run</div>
      <div class="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
        <div>node=<span class="text-primary">{{ latestAgentRun.node }}</span></div>
        <div>mode={{ latestAgentRun.mode }}</div>
        <div>exit={{ latestAgentRun.exit_reason }}</div>
        <div>time={{ latestAgentRun.wall_time_s.toFixed(2) }}s</div>
        <div>iterations={{ latestAgentRun.total_iterations }}</div>
        <div>tools={{ latestAgentRun.total_tool_calls }}</div>
        <div>errors={{ latestAgentRun.tool_errors }}</div>
        <div>tokens={{ latestAgentRun.input_tokens }}/{{ latestAgentRun.output_tokens }}</div>
      </div>
      <div v-if="toolBreakdown" class="mt-2 text-fg-dim break-all">
        breakdown={{ toolBreakdown }}
      </div>
      <div v-if="latestAgentRun.tool_calls.length" class="mt-2 border-t border-border/40 pt-2 space-y-1">
        <div v-for="(tc, i) in latestAgentRun.tool_calls" :key="i" class="text-xs break-all">
          <span :class="tc.error ? 'text-warning' : 'text-primary'">{{ tc.name }}</span>
          <span class="text-fg-dim"> args={{ tc.args_preview }}</span>
          <span class="text-fg-dim"> out={{ tc.output_preview }}</span>
        </div>
      </div>
      <div v-if="latestAgentRun.llm_error" class="mt-2 text-warning text-xs break-all">
        {{ latestAgentRun.llm_error }}
      </div>
    </div>
  </div>
</template>
