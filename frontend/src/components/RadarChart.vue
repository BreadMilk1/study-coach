<script setup lang="ts">
import { computed } from 'vue'
import { Radar } from 'vue-chartjs'
import {
  Chart as ChartJS,
  RadialLinearScale, PointElement, LineElement, Filler, Tooltip, Legend,
} from 'chart.js'
import { useMastery } from '../stores/mastery'
import { useMistakes } from '../stores/mistakes'
import { usePlan } from '../stores/plan'
import { computeRadarAxes } from '../lib/radar'

ChartJS.register(RadialLinearScale, PointElement, LineElement, Filler, Tooltip, Legend)

const mastery = useMastery()
const mistakes = useMistakes()
const plan = usePlan()

// Quiz accuracy: proxy via mastered-mistake ratio.
// No quiz activity → 0 (no data). Mistakes with interval >= 7 (well-learned)
// count positively; small intervals drag score down. Floor at 5 for stability.
const data = computed(() => {
  const hasActivity = mastery.data.scores.length > 0 || mistakes.items.length > 0
  const totalTracked = mistakes.items.length
  const quizAccuracy = totalTracked === 0
    ? (hasActivity ? 1 : 0)
    : Math.min(1, mistakes.items.filter(m => m.srs_interval_days >= 7).length / Math.max(totalTracked, 5))
  const milestones = plan.plan?.milestones ?? []
  const axes = computeRadarAxes({
    scores: mastery.data.scores,
    planMilestoneDone: milestones.filter(m => m.done).length,
    planMilestoneTotal: milestones.length,
    quizAccuracy,
    streakDays: mastery.data.streak_days ?? 0,
    coverage: mastery.data.coverage ?? 0,
  })

  return {
    labels: ['Mastery', 'Plan progress', 'Quiz accuracy', 'Streak', 'Coverage'],
    datasets: [{
      label: 'You',
      data: [axes.mastery, axes.planProgress, axes.quizAccuracy, axes.streak, axes.coverage],
      backgroundColor: 'rgba(99,102,241,0.25)',
      borderColor: '#6366f1',
      borderWidth: 2,
      pointBackgroundColor: '#6366f1',
    }],
  }
})

const options = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: { legend: { display: false } },
  scales: {
    r: {
      min: 0, max: 1,
      ticks: { display: false, stepSize: 0.25 },
      grid: { color: 'rgba(255,255,255,0.08)' },
      angleLines: { color: 'rgba(255,255,255,0.08)' },
      pointLabels: { color: '#b0b6c5', font: { size: 12, family: 'JetBrains Mono' } },
    },
  },
  animation: { duration: 250 },
}
</script>

<template>
  <section class="rounded-lg border border-border bg-surface p-6 mt-6">
    <h2 class="text-sm font-semibold mb-4 text-fg-muted uppercase tracking-wider">5-axis profile</h2>
    <div class="h-80">
      <Radar :data="data" :options="options" />
    </div>
  </section>
</template>
