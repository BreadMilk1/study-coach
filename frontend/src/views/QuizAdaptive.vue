<script setup lang="ts">
import { ref, onMounted, watch, computed } from 'vue'
import { useRoute } from 'vue-router'
import { useQuiz } from '../stores/quiz'
import { useMistakes } from '../stores/mistakes'
import { useDocuments } from '../stores/documents'
import { useSettings, type Mode } from '../stores/settings'
import { streamChat } from '../lib/api'
import DifficultySelector from '../components/DifficultySelector.vue'
import MCQCard from '../components/MCQCard.vue'
import GradeResult from '../components/GradeResult.vue'
import ModeChip from '../components/ModeChip.vue'
import EmptyCorpusBanner from '../components/EmptyCorpusBanner.vue'

const route = useRoute()
const quiz = useQuiz()
const mistakes = useMistakes()
const docs = useDocuments()
const settings = useSettings()
const mode = ref<Mode>(settings.defaultQuizMode)
const topicHint = ref('')

const needsUpload = computed(() => docs.isEmpty || quiz.needsUpload)

onMounted(async () => {
  await Promise.all([mistakes.fetch(), docs.fetch()])
  if (route.query.topic) {
    topicHint.value = String(route.query.topic)
    return
  }
  if (route.query.mistake_id) {
    const m = mistakes.items.find(d => d.mistake_id === route.query.mistake_id)
    topicHint.value = m?.topic_name ?? ''
  }
})

watch(() => [route.query.mistake_id, route.query.topic], ([mistakeId, topic]) => {
  if (topic) {
    topicHint.value = String(topic)
    quiz.reset()
    return
  }
  if (!mistakeId) { topicHint.value = ''; quiz.reset(); return }
  const m = mistakes.items.find(d => d.mistake_id === mistakeId)
  topicHint.value = m?.topic_name ?? ''
  quiz.reset()
})

function toggleMode() {
  mode.value = mode.value === 'agent_loop' ? 'deterministic' : 'agent_loop'
}

async function send(message: string) {
  quiz.startStream()
  await streamChat(
    message,
    settings.$state,
    {
      onToken: (t) => quiz.appendRaw(t),
      onDone: () => { quiz.finishStream(); mode.value = settings.defaultQuizMode },
      onError: (e) => { quiz.finishStream(); console.error(e); mode.value = settings.defaultQuizMode },
    },
    { quizMode: mode.value },
  )
}

function generate() {
  const topic = topicHint.value || 'HyDE'  // default — student usually has just-uploaded material
  send(`[difficulty:${quiz.difficulty}] quiz me on ${topic}`)
}

function submit(choice: string) {
  send(choice)
}

function nextQuestion() {
  quiz.reset()
  generate()
}
</script>

<template>
  <div class="h-full overflow-y-auto p-8">
    <div class="max-w-3xl mx-auto">
      <header class="flex items-center justify-between mb-6">
        <h1 class="text-2xl font-semibold">Quiz</h1>
        <div class="flex items-center gap-3">
          <DifficultySelector :value="quiz.difficulty" @update:value="quiz.setDifficulty" />
          <ModeChip :mode="mode" :default-mode="settings.defaultQuizMode" @toggle="toggleMode" />
        </div>
      </header>

      <EmptyCorpusBanner v-if="needsUpload" />

      <template v-else>
        <div v-if="quiz.streaming" class="text-fg-muted text-sm">Generating…</div>

        <div v-else-if="!quiz.currentMCQ && !quiz.lastGrade">
          <p v-if="topicHint" class="mb-3 text-sm text-fg-muted">
            Topic: <span class="font-mono text-fg">{{ topicHint }}</span>
          </p>
          <button @click="generate"
                  class="rounded-md bg-primary px-4 py-2 text-sm font-medium text-white hover:bg-primary-2 transition-colors">
            Generate a question
          </button>
        </div>

        <MCQCard v-else-if="quiz.currentMCQ && !quiz.lastGrade"
                 :prompt="quiz.currentMCQ.prompt"
                 :options="quiz.currentMCQ.options"
                 @submit="submit" />

        <GradeResult v-if="quiz.lastGrade"
                     :correct="quiz.lastGrade.correct"
                     :correct-answer="quiz.lastGrade.correctAnswer"
                     :explanation="quiz.lastGrade.explanation"
                     @next="nextQuestion" />
      </template>
    </div>
  </div>
</template>
