<script setup lang="ts">
import { CheckCircle2, XCircle } from 'lucide-vue-next'

defineProps<{ correct: boolean; correctAnswer: string; explanation: string }>()
defineEmits<{ (e: 'next'): void }>()
</script>

<template>
  <div :class="[
        'mt-4 rounded-lg border p-4',
        correct ? 'border-success/40 bg-success-bg' : 'border-danger/40 bg-danger-bg'
       ]">
    <div class="flex items-center gap-2 mb-2">
      <component :is="correct ? CheckCircle2 : XCircle"
                 :class="['w-5 h-5', correct ? 'text-success' : 'text-danger']" />
      <span class="text-sm font-semibold">{{ correct ? 'Correct' : 'Incorrect' }}</span>
      <span v-if="!correct && correctAnswer" class="font-mono text-xs text-fg-muted">
        correct answer: {{ correctAnswer }}
      </span>
    </div>
    <p class="text-sm whitespace-pre-wrap">{{ explanation }}</p>
    <button @click="$emit('next')"
            class="mt-3 rounded-md bg-primary px-3 py-1.5 text-sm font-medium text-white hover:bg-primary-2 transition-colors">
      Next question
    </button>
  </div>
</template>
