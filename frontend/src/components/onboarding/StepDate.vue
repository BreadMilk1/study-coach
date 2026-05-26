<script setup lang="ts">
import { ref } from 'vue'
const emit = defineEmits<{ done: [date: string]; skip: [] }>()
const date = ref('')
</script>
<template>
  <div class="space-y-4">
    <h2 class="text-xl font-semibold text-fg">When is your exam? (optional)</h2>
    <input type="date" v-model="date"
           class="w-full rounded-md bg-surface-2 border border-border px-4 py-3 text-fg focus:border-primary-ring focus:outline-none" />
    <div class="flex gap-2">
      <button @click="emit('done', date)" :disabled="!date"
              class="rounded-md bg-primary px-6 py-2 text-sm font-medium text-white hover:bg-primary-2 disabled:opacity-40 disabled:cursor-not-allowed transition-colors">
        Next
      </button>
      <button @click="emit('skip')"
              class="rounded-md border border-border px-4 py-2 text-sm text-fg-muted hover:bg-surface-2 transition-colors">
        Skip
      </button>
    </div>
  </div>
</template>
