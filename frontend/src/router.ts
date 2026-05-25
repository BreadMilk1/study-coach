import { createRouter, createWebHistory } from 'vue-router'
import Overview from './views/Overview.vue'
import Chat from './views/Chat.vue'
import PlanTimeline from './views/PlanTimeline.vue'
import QuizAdaptive from './views/QuizAdaptive.vue'
import MistakeBank from './views/MistakeBank.vue'
import Library from './views/Library.vue'
import Settings from './views/Settings.vue'

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
  ],
})
