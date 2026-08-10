<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import {
  MessageSquare, ListTodo, BookOpen, MoreHorizontal,
  LayoutDashboard, AlertTriangle, FolderOpen, Settings as SettingsIcon,
} from 'lucide-vue-next'

const route = useRoute()
const router = useRouter()
const moreOpen = ref(false)

watch(
  () => route.path,
  () => {
    moreOpen.value = false
  },
)

const primaryTabs = [
  { to: '/chat', icon: MessageSquare, text: 'nav.chat' },
  { to: '/plan', icon: ListTodo, text: 'nav.plan' },
  { to: '/quiz', icon: BookOpen, text: 'nav.quiz' },
]

const moreItems = [
  { to: '/', icon: LayoutDashboard, text: 'nav.overview' },
  { to: '/mistakes', icon: AlertTriangle, text: 'nav.mistakes' },
  { to: '/library', icon: FolderOpen, text: 'nav.library' },
  { to: '/settings', icon: SettingsIcon, text: 'nav.settings' },
]

const moreActive = computed(() => moreItems.some(item => item.to === route.path))

function isActive(to: string) {
  return route.path === to
}

function goMore(to: string) {
  moreOpen.value = false
  router.push(to)
}
</script>

<template>
  <nav class="fixed bottom-0 left-0 right-0 z-50 h-14 bg-surface border-t border-border flex items-center justify-around">
    <RouterLink
      v-for="tab in primaryTabs"
      :key="tab.to"
      :to="tab.to"
      class="flex flex-col items-center justify-center gap-0.5 flex-1 h-full text-[10px] transition-colors"
      :class="isActive(tab.to) ? 'text-indigo-400' : 'text-fg-muted'"
    >
      <component :is="tab.icon" class="w-5 h-5" />
      <span>{{ $t(tab.text) }}</span>
    </RouterLink>
    <button
      type="button"
      class="flex flex-col items-center justify-center gap-0.5 flex-1 h-full text-[10px] transition-colors"
      :class="moreActive || moreOpen ? 'text-indigo-400' : 'text-fg-muted'"
      :aria-expanded="moreOpen"
      aria-haspopup="true"
      @click="moreOpen = !moreOpen"
    >
      <MoreHorizontal class="w-5 h-5" />
      <span>{{ $t('nav.more') }}</span>
    </button>
  </nav>

  <div
    v-if="moreOpen"
    class="fixed inset-0 z-40 bg-black/50"
    @click="moreOpen = false"
  />
  <div
    v-if="moreOpen"
    class="fixed bottom-14 left-0 right-0 z-50 rounded-t-xl border border-border bg-surface p-3 shadow-xl"
    role="menu"
  >
    <button
      v-for="item in moreItems"
      :key="item.to"
      type="button"
      role="menuitem"
      class="flex w-full items-center gap-3 rounded-md px-3 py-3 text-sm transition-colors max-md:min-h-12"
      :class="isActive(item.to) ? 'bg-primary-bg text-fg' : 'text-fg-muted hover:bg-white/5'"
      @click="goMore(item.to)"
    >
      <component :is="item.icon" class="w-4 h-4" />
      {{ $t(item.text) }}
    </button>
  </div>
</template>
