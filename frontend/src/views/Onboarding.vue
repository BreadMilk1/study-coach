<script setup lang="ts">
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import StepName from '../components/onboarding/StepName.vue'
import StepDate from '../components/onboarding/StepDate.vue'
import StepUpload from '../components/onboarding/StepUpload.vue'
import { authHeaders } from '../stores/settings'

const router = useRouter()
const step = ref(1)
const goalTitle = ref('')
const examDate = ref('')

function onNameDone(title: string) { goalTitle.value = title; step.value = 2 }
function onDateDone(date: string) { examDate.value = date; step.value = 3 }
async function onUploadDone() {
  const resp = await fetch('/api/goals', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...authHeaders(),
    },
    body: JSON.stringify({ title: goalTitle.value, exam_date: examDate.value || null }),
  })
  const { goal_id } = await resp.json()
  router.push({ path: '/chat', query: { goal_id, auto: `Help me make a study plan for ${goalTitle.value}` } })
}
</script>

<template>
  <div class="h-full flex items-center justify-center bg-bg">
    <div class="w-full max-w-lg px-4">
      <div class="flex gap-2 mb-8 justify-center">
        <div v-for="s in 3" :key="s"
             class="w-8 h-8 rounded-full flex items-center justify-center text-xs font-mono transition-colors"
             :class="s <= step ? 'bg-primary text-white' : 'bg-surface-2 text-fg-muted'">
          {{ s }}
        </div>
      </div>
      <StepName v-if="step === 1" @done="onNameDone" />
      <StepDate v-if="step === 2" @done="onDateDone" @skip="step = 3" />
      <StepUpload v-if="step === 3" @done="onUploadDone" @skip="onUploadDone" />
    </div>
  </div>
</template>
