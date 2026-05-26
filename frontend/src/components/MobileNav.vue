<script setup lang="ts">
import { useRoute } from 'vue-router'
import { MessageSquare, ListTodo, BookOpen } from 'lucide-vue-next'

const route = useRoute()

const tabs = [
  { to: '/chat', icon: MessageSquare, text: 'nav.chat' },
  { to: '/plan', icon: ListTodo, text: 'nav.plan' },
  { to: '/quiz', icon: BookOpen, text: 'nav.quiz' },
]

function isActive(to: string) {
  return route.path === to
}
</script>

<template>
  <nav class="fixed bottom-0 left-0 right-0 z-50 h-14 bg-surface border-t border-border flex items-center justify-around">
    <RouterLink
      v-for="tab in tabs"
      :key="tab.to"
      :to="tab.to"
      class="flex flex-col items-center justify-center gap-0.5 flex-1 h-full text-[10px] transition-colors"
      :class="isActive(tab.to) ? 'text-indigo-400' : 'text-fg-muted'"
    >
      <component :is="tab.icon" class="w-5 h-5" />
      <span>{{ $t(tab.text) }}</span>
    </RouterLink>
  </nav>
</template>
