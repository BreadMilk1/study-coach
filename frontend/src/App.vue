<script setup lang="ts">
import { RouterLink, RouterView } from 'vue-router'
import {
  LayoutDashboard, MessageSquare, ListTodo, BookOpen,
  AlertTriangle, FolderOpen, Settings as SettingsIcon,
} from 'lucide-vue-next'

const navSections = [
  {
    label: 'Study',
    items: [
      { to: '/',          icon: LayoutDashboard, text: 'Overview' },
      { to: '/chat',      icon: MessageSquare,   text: 'Chat' },
      { to: '/plan',      icon: ListTodo,        text: 'Plan' },
      { to: '/quiz',      icon: BookOpen,        text: 'Quiz' },
    ],
  },
  {
    label: 'Review',
    items: [
      { to: '/mistakes',  icon: AlertTriangle,   text: 'Mistakes' },
    ],
  },
  {
    label: 'System',
    items: [
      { to: '/library',   icon: FolderOpen,      text: 'Library' },
      { to: '/settings',  icon: SettingsIcon,    text: 'Settings' },
    ],
  },
]
</script>

<template>
  <div class="h-full flex">
    <nav class="w-56 bg-surface p-4 flex flex-col gap-1 border-r border-border">
      <h1 class="text-lg font-semibold mb-4 px-2">Study Coach</h1>
      <template v-for="section in navSections" :key="section.label">
        <div class="px-2 text-[10px] uppercase tracking-wider text-fg-dim mt-3 mb-1">
          {{ section.label }}
        </div>
        <RouterLink
          v-for="item in section.items"
          :key="item.to"
          :to="item.to"
          class="flex items-center gap-3 px-3 py-2 rounded-md text-sm text-fg-muted hover:bg-white/5 transition-colors"
          active-class="!bg-primary-bg !text-fg"
        >
          <component :is="item.icon" class="w-4 h-4" />
          {{ item.text }}
        </RouterLink>
      </template>
      <div class="mt-auto text-xs text-fg-dim px-2">P3 · productized shell</div>
    </nav>
    <main class="flex-1 overflow-hidden">
      <RouterView />
    </main>
  </div>
</template>
