<script setup lang="ts">
import { ref, onBeforeUnmount, onMounted, watch, computed } from 'vue'
import { useRoute } from 'vue-router'
import { useQuiz } from '../stores/quiz'
import { useMistakes } from '../stores/mistakes'
import { useMastery } from '../stores/mastery'
import { useDocuments } from '../stores/documents'
import { useSettings, type Mode } from '../stores/settings'
import { streamChat } from '../lib/api'
import DifficultySelector from '../components/DifficultySelector.vue'
import MCQCard from '../components/MCQCard.vue'
import GradeResult from '../components/GradeResult.vue'
import ModeChip from '../components/ModeChip.vue'
import EmptyCorpusBanner from '../components/EmptyCorpusBanner.vue'
import InfoPopover from '../components/InfoPopover.vue'

const route = useRoute()
const quiz = useQuiz()
const mistakes = useMistakes()
const mastery = useMastery()
const docs = useDocuments()
const settings = useSettings()
const mode = ref<Mode>(
  settings.toolCapable === false ? 'deterministic' : settings.defaultQuizMode
)
const topicHint = ref('')
let active = true

const needsUpload = computed(() => docs.isEmpty || quiz.needsUpload)

onBeforeUnmount(() => {
  active = false
})

onMounted(async () => {
  quiz.reset()
  await Promise.all([mistakes.fetch(), docs.fetch()])
  if (!active) return
  if (route.query.topic) {
    topicHint.value = String(route.query.topic)
    generate()
    return
  }
  if (route.query.mistake_id) {
    const m = mistakes.items.find(d => d.mistake_id === route.query.mistake_id)
    if (m) {
      topicHint.value = m.topic_name
      quiz.currentMCQ = { prompt: m.question.prompt, options: m.question.options }
      quiz.currentMistakeId = m.mistake_id
    }
  }
})

watch(() => [route.query.mistake_id, route.query.topic], ([mistakeId, topic]) => {
  quiz.reset()
  if (topic) {
    topicHint.value = String(topic)
    generate()
    return
  }
  if (!mistakeId) { topicHint.value = ''; return }
  const m = mistakes.items.find(d => d.mistake_id === mistakeId)
  if (m) {
    topicHint.value = m.topic_name
    quiz.currentMCQ = { prompt: m.question.prompt, options: m.question.options }
    quiz.currentMistakeId = m.mistake_id
  }
})

function toggleMode() {
  if (settings.toolCapable === false) return
  mode.value = mode.value === 'agent_loop' ? 'deterministic' : 'agent_loop'
}

async function send(message: string) {
  quiz.startStream()
  await streamChat(
    message,
    settings.$state,
    {
      onToken: (t) => quiz.appendRaw(t),
      onDone: () => {
        quiz.finishStream()
        mastery.fetch()
        mistakes.fetch()
        if (settings.toolCapable !== false) mode.value = settings.defaultQuizMode
      },
      onError: (e) => {
        quiz.finishStream()
        console.error(e)
        mastery.fetch()
        mistakes.fetch()
        if (settings.toolCapable !== false) mode.value = settings.defaultQuizMode
      },
    },
    { quizMode: mode.value },
  )
}

function generate() {
  const topic = topicHint.value || 'HyDE'  // default — student usually has just-uploaded material
  send(`[difficulty:${quiz.difficulty}] quiz me on ${topic}`)
}

async function submit(choice: string) {
  if (quiz.currentMistakeId) {
    try {
      const reviewed = await quiz.reviewCurrentMistake(choice)
      if (!reviewed) return
      await Promise.all([mastery.fetch(), mistakes.fetch()])
    } catch (e) {
      console.error('review failed', e)
    }
    return
  }
  send(choice)
}

function nextQuestion() {
  // After mistake redo: load next due mistake, or show "all caught up"
  const due = mistakes.due
  if (due.length > 0) {
    const next = due[0]
    quiz.reset()
    topicHint.value = next.topic_name
    quiz.currentMCQ = { prompt: next.question.prompt, options: next.question.options }
    quiz.currentMistakeId = next.mistake_id
    return
  }
  // No more due mistakes — back to blank quiz
  quiz.reset()
  topicHint.value = ''
}
</script>

<template>
  <div class="h-full overflow-y-auto p-8 max-md:p-4">
    <div class="max-w-3xl mx-auto">
      <header class="flex items-center justify-between mb-6">
        <div class="flex items-center gap-2">
          <h1 class="text-2xl font-semibold">Quiz</h1>
          <InfoPopover title="Quiz 功能说明">
            <p>Quiz 从你的 PDF 资料中生成选择题，测试知识掌握程度。</p>
            <p><strong>关联 Plan：</strong>Plan 中的 milestone 主题可直接跳转到 Quiz 验证掌握情况。Quiz 成绩会回写到 Mastery Score，驱动 milestone 的通过判定。</p>
            <p><strong>关联 Mistakes：</strong>答错的题目进入 Mistake Bank，按 SM-2 间隔重复算法安排重做。24 小时内再次出现，连续答对后间隔逐渐延长至淡出。</p>
            <p><strong>模式切换：</strong>Deterministic 适合本地小模型；Agent Loop 适合支持 Tool Call 的模型，可自动检索资料并过滤无根据的题目。</p>
          </InfoPopover>
        </div>
        <div class="flex items-center gap-3">
          <DifficultySelector :value="quiz.difficulty" @update:value="quiz.setDifficulty" />
          <ModeChip :mode="mode" :default-mode="settings.defaultQuizMode" @toggle="toggleMode" />
        </div>
      </header>

      <div v-if="quiz.errorMsg"
           class="rounded-lg border border-red-500/30 bg-red-500/10 p-4 flex items-start gap-3 mb-6">
        <span class="text-sm text-fg flex-1">{{ $t('quiz.errorBanner') }}</span>
        <button @click="quiz.errorMsg = ''"
                class="text-fg-muted hover:text-fg shrink-0 text-lg leading-none">&times;</button>
      </div>

      <EmptyCorpusBanner v-if="needsUpload" />

      <template v-else>
        <div v-if="quiz.streaming" class="text-fg-muted text-sm">Generating…</div>

        <div v-else-if="!quiz.currentMCQ && !quiz.lastGrade">
          <p v-if="topicHint" class="mb-3 text-sm text-fg-muted">
            Topic: <span class="font-mono text-fg">{{ topicHint }}</span>
          </p>
          <button @click="generate"
                  class="rounded-md bg-primary px-4 py-2 text-sm font-medium text-white hover:bg-primary-2 transition-colors max-md:py-3">
            {{ $t('quiz.generate') }}
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
