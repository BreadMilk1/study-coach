<script setup lang="ts">
import { onBeforeUnmount, onMounted, watch } from 'vue'
import { RouterLink, RouterView } from 'vue-router'
import { useI18n } from 'vue-i18n'
import {
  LayoutDashboard, MessageSquare, ListTodo, BookOpen,
  AlertTriangle, FolderOpen, Settings as SettingsIcon,
} from 'lucide-vue-next'
import { getDataSummary, resetData } from './lib/api'
import { createDataLifecycleChannel } from './lib/dataLifecycleChannel'
import {
  clearFactoryBrowserState,
  clearFactorySessionState,
  clearStartupChoice,
  clearStoredChatSessionId,
  markStartupChoice,
} from './lib/dataLifecycle'
import { resetClientLearningState } from './lib/resetClientState'
import { useActivity } from './stores/activity'
import { useChat } from './stores/chat'
import { useDataLifecycle } from './stores/dataLifecycle'
import { useDocuments } from './stores/documents'
import { useMastery } from './stores/mastery'
import { useMistakes } from './stores/mistakes'
import { useNotifications } from './stores/notifications'
import { usePlan } from './stores/plan'
import { useQuiz } from './stores/quiz'
import { useMediaQuery } from './composables/useMediaQuery'
import MobileNav from './components/MobileNav.vue'
import ResetConfirmDialog from './components/ResetConfirmDialog.vue'
import StartupDataGate from './components/StartupDataGate.vue'
import ToastHost from './components/ToastHost.vue'

const isMobile = useMediaQuery('(max-width: 767px)')
const { t } = useI18n()
const activity = useActivity()
const chat = useChat()
const lifecycle = useDataLifecycle()
const documents = useDocuments()
const mastery = useMastery()
const mistakes = useMistakes()
const notifications = useNotifications()
const plan = usePlan()
const quiz = useQuiz()
let notifiedLearningGeneration = -1

const lifecycleChannel = createDataLifecycleChannel(message => (
  lifecycle.handleExternalReset(message.scope)
))

lifecycle.initialize({
  summary: getDataSummary,
  reset: resetData,
  resetClient: () => resetClientLearningState({
    clearChatSession: clearStoredChatSessionId,
    chat,
    quiz,
    documents,
    plan,
    mistakes,
    mastery,
    activity,
  }),
  markChoice: () => markStartupChoice(sessionStorage),
  clearChoice: () => clearStartupChoice(sessionStorage),
  clearFactory: () => clearFactoryBrowserState(localStorage, sessionStorage),
  clearFactorySession: () => clearFactorySessionState(sessionStorage),
  broadcast: scope => lifecycleChannel.publish(scope),
  reload: () => window.location.reload(),
  pause: milliseconds => new Promise(resolve => globalThis.setTimeout(resolve, milliseconds)),
})

onMounted(() => {
  void lifecycle.inspect()
})

onBeforeUnmount(() => {
  lifecycleChannel.close()
})

watch(() => lifecycle.phase, (phase, previous) => {
  const result = lifecycle.lastResult
  if (previous !== 'resetting' || phase !== 'ready' || result?.scope !== 'learning') return
  if (notifiedLearningGeneration === lifecycle.operationGeneration) return
  notifiedLearningGeneration = lifecycle.operationGeneration
  const count = Object.values(result.deleted).reduce((total, value) => total + value, 0)
  notifications.push({
    kind: 'success',
    message: t('dataLifecycle.toast.learningCleared', { count }),
  })
})

const navSections = [
  {
    label: 'nav.study',
    items: [
      { to: '/',          icon: LayoutDashboard, text: 'nav.overview' },
      { to: '/chat',      icon: MessageSquare,   text: 'nav.chat' },
      { to: '/plan',      icon: ListTodo,        text: 'nav.plan' },
      { to: '/quiz',      icon: BookOpen,        text: 'nav.quiz' },
    ],
  },
  {
    label: 'nav.review',
    items: [
      { to: '/mistakes',  icon: AlertTriangle,   text: 'nav.mistakes' },
    ],
  },
  {
    label: 'nav.system',
    items: [
      { to: '/library',   icon: FolderOpen,      text: 'nav.library' },
      { to: '/settings',  icon: SettingsIcon,    text: 'nav.settings' },
    ],
  },
]
</script>

<template>
  <div class="h-full flex">
    <nav v-if="!isMobile" class="w-56 bg-surface p-4 flex flex-col gap-1 border-r border-border">
      <h1 class="text-lg font-semibold mb-4 px-2">Study Coach</h1>
      <template v-for="section in navSections" :key="section.label">
        <div class="px-2 text-[10px] uppercase tracking-wider text-fg-dim mt-3 mb-1">
          {{ $t(section.label) }}
        </div>
        <RouterLink
          v-for="item in section.items"
          :key="item.to"
          :to="item.to"
          class="flex items-center gap-3 px-3 py-2 rounded-md text-sm text-fg-muted hover:bg-white/5 transition-colors"
          active-class="!bg-primary-bg !text-fg"
        >
          <component :is="item.icon" class="w-4 h-4" />
          {{ $t(item.text) }}
        </RouterLink>
      </template>
      <div class="mt-auto text-xs text-fg-dim px-2">P5 · local-first</div>
    </nav>
    <main class="flex-1 overflow-hidden" :class="{ 'pb-14': isMobile }">
      <RouterView v-if="lifecycle.workspaceUnlocked" />
    </main>
    <MobileNav v-if="isMobile" />
    <StartupDataGate
      :phase="lifecycle.phase"
      :summary="lifecycle.summary"
      :error="lifecycle.error"
      :pending="lifecycle.externalClientPending"
      @continue="lifecycle.continueExisting()"
      @continue-without-clearing="lifecycle.continueWithoutClearing()"
      @start-fresh="lifecycle.requestLearningReset()"
      @retry="lifecycle.inspect()"
      @acknowledge-external="lifecycle.acknowledgeExternalReset()"
    />
    <ResetConfirmDialog
      :phase="lifecycle.phase"
      :scope="lifecycle.pendingScope"
      :summary="lifecycle.summary"
      :error="lifecycle.error"
      @cancel="lifecycle.cancelReset()"
      @confirm-learning="lifecycle.confirmLearningReset()"
      @confirm-factory="lifecycle.confirmFactoryReset()"
      @retry="lifecycle.retryReset()"
    />
    <ToastHost />
  </div>
</template>
