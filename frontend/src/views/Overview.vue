<script setup lang="ts">
import { onMounted } from 'vue'
import { useOverview } from '../stores/overview'
import { useDocuments } from '../stores/documents'
import { useMastery } from '../stores/mastery'
import UploadGate from '../components/UploadGate.vue'
import MasteryCard from '../components/MasteryCard.vue'
import PlanProgressCard from '../components/PlanProgressCard.vue'
import MistakesDueCard from '../components/MistakesDueCard.vue'
import WeakTopicsChips from '../components/WeakTopicsChips.vue'
import RadarChart from '../components/RadarChart.vue'
import HeatmapCard from '../components/HeatmapCard.vue'

const overview = useOverview()
const docs = useDocuments()
const mastery = useMastery()

onMounted(() => overview.fetchAll())
</script>

<template>
  <div class="h-full overflow-y-auto p-8">
    <div class="max-w-6xl mx-auto">
      <header class="mb-8">
        <h1 class="text-3xl font-bold tracking-tight">{{ $t('nav.overview') }}</h1>
        <p class="text-sm text-fg-muted mt-1">
          {{ new Date().toDateString() }} · overdue
          <span class="font-mono text-warning">{{ mastery.data.overdue_milestones_count }}</span>
        </p>
      </header>

      <UploadGate v-if="docs.isEmpty" />

      <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <MasteryCard :scores="overview.topMastery" />
        <PlanProgressCard :next-milestone="overview.nextMilestone" />
        <MistakesDueCard />
        <WeakTopicsChips :topics="mastery.data.weak_topics" />
      </div>

      <RadarChart />
      <HeatmapCard />
    </div>
  </div>
</template>
