<script setup lang="ts">
import { ref } from 'vue'
const emit = defineEmits<{ done: [title: string] }>()
const title = ref('')
</script>
<template>
  <div class="space-y-4">
    <h2 class="text-xl font-semibold text-fg">What's your exam or goal?</h2>
    <p class="text-sm text-fg-muted">e.g. HKBU CS Final, Learn Machine Learning</p>
    <input v-model="title"
           placeholder="Enter your goal..."
           class="w-full rounded-md bg-surface-2 border border-border px-4 py-3 text-fg placeholder-fg-dim focus:border-primary-ring focus:outline-none" />
    <button @click="emit('done', title)" :disabled="!title.trim()"
            class="rounded-md bg-primary px-6 py-2 text-sm font-medium text-white hover:bg-primary-2 disabled:opacity-40 disabled:cursor-not-allowed transition-colors">
      Next
    </button>
  </div>
</template>
