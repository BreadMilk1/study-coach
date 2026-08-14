import { createRouter, createWebHistory } from 'vue-router'
import Overview from './views/Overview.vue'
import Chat from './views/Chat.vue'
import PlanTimeline from './views/PlanTimeline.vue'
import QuizAdaptive from './views/QuizAdaptive.vue'
import MistakeBank from './views/MistakeBank.vue'
import Library from './views/Library.vue'
import Settings from './views/Settings.vue'
import RunLab from './views/RunLab.vue'
import RunDetail from './views/RunDetail.vue'
import RunCompare from './views/RunCompare.vue'

export const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', name: 'overview', component: Overview },
    { path: '/chat', name: 'chat', component: Chat },
    { path: '/plan', name: 'plan', component: PlanTimeline },
    { path: '/quiz', name: 'quiz', component: QuizAdaptive },
    { path: '/mistakes', name: 'mistakes', component: MistakeBank },
    { path: '/library', name: 'library', component: Library },
    { path: '/settings', name: 'settings', component: Settings },
    { path: '/run-lab', name: 'run-lab', component: RunLab },
    { path: '/run-lab/runs/:runId', name: 'run-detail', component: RunDetail },
    { path: '/run-lab/compare', name: 'run-compare', component: RunCompare },
    { path: '/onboarding', name: 'onboarding', component: () => import('./views/Onboarding.vue') },
  ],
})
