<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { usePlan } from '../stores/plan'
import { useSettings, type Mode } from '../stores/settings'
import { streamChat, type MilestoneDto } from '../lib/api'
import MilestoneList from '../components/MilestoneList.vue'
import ModeChip from '../components/ModeChip.vue'
import MindmapPanel from '../components/MindmapPanel.vue'

const planStore = usePlan()
const settings = useSettings()
const router = useRouter()
const mode = ref<Mode>(settings.defaultPlannerMode)
const checkInLoading = ref(false)

onMounted(() => planStore.fetch())

function toggleMode() {
  mode.value = mode.value === 'agent_loop' ? 'deterministic' : 'agent_loop'
}

async function toggleMilestone(milestone: MilestoneDto) {
  if (!milestone.id) return
  await planStore.toggleMilestone(milestone.id, !milestone.done)
}

function validateMilestone(milestone: MilestoneDto) {
  if (!milestone.topic) return
  router.push({ path: '/quiz', query: { topic: milestone.topic } })
}

async function checkIn() {
  checkInLoading.value = true
  const nextMode = mode.value
  await streamChat(
    '进度怎么样了',
    settings.$state,
    {
      onDone: () => { planStore.fetch(); mode.value = settings.defaultPlannerMode },
      onError: () => { mode.value = settings.defaultPlannerMode },
    },
    { plannerMode: nextMode },
  )
  checkInLoading.value = false
}
</script>

<template>
  <div class="h-full overflow-y-auto p-8">
    <div class="max-w-4xl mx-auto">
      <header class="flex items-center justify-between mb-6">
        <div>
          <h1 class="text-2xl font-semibold">Plan</h1>
          <p v-if="planStore.plan" class="text-sm text-fg-muted mt-1">
            {{ planStore.plan.goal_title }} ·
            <span class="font-mono">{{ planStore.plan.milestones.length }} milestones</span>
          </p>
        </div>
        <ModeChip :mode="mode" :default-mode="settings.defaultPlannerMode" @toggle="toggleMode" />
      </header>

      <div v-if="planStore.loading" class="text-fg-muted text-sm">Loading…</div>

      <div v-else-if="planStore.noActive"
           class="rounded-lg border border-border bg-surface p-6 text-center">
        <p class="text-fg-muted">No active plan yet.</p>
        <p class="text-xs text-fg-dim mt-2">
          Go to <RouterLink to="/chat" class="underline">Chat</RouterLink> and ask
          <span class="font-mono">帮我做学习计划 on &lt;topic&gt;</span>.
        </p>
      </div>

      <template v-else-if="planStore.plan">
        <MilestoneList
          :milestones="planStore.plan.milestones"
          :updating-milestone-id="planStore.updatingMilestoneId"
          @toggle="toggleMilestone"
          @validate="validateMilestone"
        />
        <section v-if="planStore.events.length" class="mt-6 rounded-lg border border-border bg-surface p-4">
          <h2 class="text-sm font-semibold text-fg-muted uppercase tracking-wider">Recent changes</h2>
          <ul class="mt-3 flex flex-col gap-2 text-xs text-fg-muted">
            <li v-for="event in planStore.events.slice(0, 5)" :key="event.id">
              <span class="font-mono text-fg">{{ event.action }}</span>
              <span v-if="event.reason"> — {{ event.reason }}</span>
            </li>
          </ul>
        </section>
        <MindmapPanel v-if="planStore.mindmapMermaid" :mermaid="planStore.mindmapMermaid" />
        <div class="mt-6 flex justify-end">
          <button @click="checkIn" :disabled="checkInLoading"
                  class="rounded-md bg-primary px-4 py-2 text-sm font-medium text-white hover:bg-primary-2 disabled:opacity-40 transition-colors">
            {{ checkInLoading ? 'Checking in…' : 'Check-in progress' }}
          </button>
        </div>
      </template>
    </div>
  </div>
</template>
