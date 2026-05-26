<script setup lang="ts">
import type { Difficulty } from '../stores/quiz'

const props = defineProps<{ value: Difficulty }>()
const emit = defineEmits<{ (e: 'update:value', d: Difficulty): void }>()

const levels: Difficulty[] = ['easy', 'med', 'hard']
</script>

<template>
  <div role="radiogroup" :aria-label="$t('quiz.difficulty')"
       class="inline-flex rounded-md border border-border bg-surface p-0.5">
    <button v-for="level in levels" :key="level"
            role="radio" :aria-checked="props.value === level"
            @click="emit('update:value', level)"
            :class="[
              'px-3 py-1 text-xs font-mono rounded transition-colors',
              props.value === level
                ? 'bg-primary text-white'
                : 'text-fg-muted hover:text-fg hover:bg-white/5'
            ]">
      {{ level }}
    </button>
  </div>
</template>
