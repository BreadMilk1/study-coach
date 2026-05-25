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
import { useDocuments } from '../stores/documents'

ChartJS.register(RadialLinearScale, PointElement, LineElement, Filler, Tooltip, Legend)

const mastery = useMastery()
const mistakes = useMistakes()
const plan = usePlan()
const docs = useDocuments()

// Simple normalized 0..1 vector across 5 dims.
const data = computed(() => {
  const avgMastery = mastery.data.scores.length === 0
    ? 0
    : mastery.data.scores.reduce((a, s) => a + s.score, 0) / mastery.data.scores.length
  const planProgress = !plan.plan || plan.plan.milestones.length === 0
    ? 0
    : plan.plan.milestones.filter(m => m.done).length / plan.plan.milestones.length
  // Quiz "accuracy" = 1 - normalizedMistakes; cap due count at 20 for normalization.
  const quizAccuracy = Math.max(0, 1 - Math.min(mistakes.due.length, 20) / 20)
  // Streak — P4. Placeholder = 0.5 if any activity, else 0.
  const streak = (mastery.data.scores.length || mistakes.due.length) ? 0.5 : 0
  // Coverage = doc count / 5 capped at 1.
  const coverage = Math.min(docs.docs.length / 5, 1)

  return {
    labels: ['Mastery', 'Plan progress', 'Quiz accuracy', 'Streak', 'Coverage'],
    datasets: [{
      label: 'You',
      data: [avgMastery, planProgress, quizAccuracy, streak, coverage],
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
