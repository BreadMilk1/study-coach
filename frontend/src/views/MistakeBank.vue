<script setup lang="ts">
import { onMounted, ref, computed } from 'vue'
import { useMistakes } from '../stores/mistakes'
import MistakeRow from '../components/MistakeRow.vue'
import InfoPopover from '../components/InfoPopover.vue'

const store = useMistakes()
const showAll = ref(false)
onMounted(() => store.fetch())
const dueCount = computed(() => store.due.length)
const trackedCount = computed(() => store.items.length)

async function toggleShowAll() {
  showAll.value = !showAll.value
  await store.fetch(showAll.value)
}
</script>

<template>
  <div class="h-full overflow-y-auto p-8">
    <div class="max-w-4xl mx-auto">
      <header class="mb-6">
        <div class="flex items-center gap-2">
          <h1 class="text-2xl font-semibold">{{ $t('nav.mistakes') }}</h1>
          <InfoPopover title="Mistake Bank 功能说明">
            <p>Mistake Bank 收集你在 Quiz 中答错的题目，使用 SM-2 间隔重复算法管理复习节奏。</p>
            <p><strong>关联 Quiz：</strong>Quiz 中答错的题目自动进入 Mistake Bank。点击 Redo 可重做原题，答对后 SRS 间隔延长（1 → 6 → 15 → 38+ 天），答错重置为 1 天。复习结果同步更新 Topic Mastery Score。</p>
            <p><strong>SRS 逻辑：</strong>SM-2 算法根据答题表现动态调整复习间隔。连续答对的题目逐渐淡出（interval &gt; 30 天视为已掌握），顽固错题持续高频出现。</p>
            <p><strong>Redo 流程：</strong>点击 Redo → 显示原题 → 选择答案 → 后端判分并更新 SRS / Mastery → Next 加载下一个待审错题。</p>
          </InfoPopover>
        </div>
        <div class="flex items-center gap-3 mt-1">
          <p class="text-sm text-fg-muted font-mono">
            {{ dueCount }} due today · {{ trackedCount }} tracked
          </p>
          <button @click="toggleShowAll"
                  class="text-xs font-mono px-2 py-0.5 rounded border border-white/15 text-fg-muted hover:text-fg hover:border-white/25 transition-colors">
            {{ showAll ? $t('mistakes.dueOnly') : $t('mistakes.showAll') }}
          </button>
        </div>
      </header>

      <div v-if="store.loading" class="text-fg-muted text-sm">Loading…</div>
      <div v-else-if="store.error" class="text-sm text-danger">{{ store.error }}</div>
      <div v-else-if="trackedCount === 0" class="rounded-lg border border-border bg-surface p-6 text-center">
        <p class="text-fg-muted">No mistakes tracked yet. Take a quiz to start tracking.</p>
      </div>
      <div v-else class="flex flex-col gap-3">
        <MistakeRow v-for="row in store.items" :key="row.mistake_id" :row="row" />
      </div>
    </div>
  </div>
</template>
