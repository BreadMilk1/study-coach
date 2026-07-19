<script setup lang="ts">
import { RouterLink, RouterView } from 'vue-router'
import {
  LayoutDashboard, MessageSquare, ListTodo, BookOpen,
  AlertTriangle, FolderOpen, Settings as SettingsIcon,
} from 'lucide-vue-next'
import { useMediaQuery } from './composables/useMediaQuery'
import MobileNav from './components/MobileNav.vue'

const isMobile = useMediaQuery('(max-width: 767px)')

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
      <RouterView />
    </main>
    <MobileNav v-if="isMobile" />
  </div>
</template>
