<script setup lang="ts">
import { computed } from 'vue'
import { usePlan } from '../stores/plan'
import type { MilestoneDto } from '../lib/api'

const plan = usePlan()
const props = defineProps<{ nextMilestone: MilestoneDto | null }>()
const done = computed(() => plan.plan?.milestones.filter(m => m.done).length ?? 0)
const total = computed(() => plan.plan?.milestones.length ?? 0)
const pct = computed(() => (total.value === 0 ? 0 : Math.round(100 * done.value / total.value)))
const lowMasteryCompleted = computed(() =>
  plan.plan?.milestones.filter(m => m.done && m.validation_recommended).length ?? 0
)
</script>
<template>
  <section class="rounded-lg border border-border bg-surface p-6">
    <h2 class="text-sm font-semibold mb-4 text-fg-muted uppercase tracking-wider">{{ $t('overview.planProgress') }}</h2>
    <div v-if="!plan.plan" class="text-fg-dim text-sm">No active plan.</div>
    <template v-else>
      <div class="flex items-baseline gap-2">
        <span class="text-2xl font-mono">{{ done }}</span>
        <span class="text-fg-muted">/ {{ total }} done</span>
        <span class="ml-auto text-xs font-mono text-fg-muted">{{ pct }}%</span>
      </div>
      <div class="h-2 bg-bg rounded-full overflow-hidden mt-2">
        <div class="h-full bg-primary transition-all" :style="{ width: pct + '%' }"></div>
      </div>
      <div v-if="props.nextMilestone" class="mt-4 text-xs text-fg-muted">
        Next: <span class="text-fg">{{ props.nextMilestone.title }}</span>
      </div>
      <div v-if="lowMasteryCompleted" class="mt-2 text-xs text-warning">
        {{ lowMasteryCompleted }} completed milestone{{ lowMasteryCompleted === 1 ? '' : 's' }} still need mastery validation.
      </div>
    </template>
  </section>
</template>
