<script setup lang="ts">
import { ref } from 'vue'
import { uploadDocument } from '../../lib/api'

const emit = defineEmits<{ done: []; skip: [] }>()
const uploading = ref(false)
const uploaded = ref(false)

async function handleFile(e: Event) {
  const file = (e.target as HTMLInputElement).files?.[0]
  if (!file) return
  uploading.value = true
  try {
    await uploadDocument(file)
    uploaded.value = true
  } catch { /* user can still proceed */ }
  uploading.value = false
}
</script>
<template>
  <div class="space-y-4">
    <h2 class="text-xl font-semibold text-fg">Upload study materials</h2>
    <p class="text-sm text-fg-muted">PDF files with your course content</p>
    <div class="rounded-lg border-2 border-dashed border-border p-8 text-center">
      <input type="file" accept=".pdf" @change="handleFile"
             class="text-sm text-fg-muted file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-medium file:bg-primary file:text-white hover:file:bg-primary-2 file:cursor-pointer file:transition-colors" />
      <p v-if="uploading" class="text-sm text-fg-muted mt-2">Uploading...</p>
      <p v-if="uploaded" class="text-sm text-success mt-2">Uploaded!</p>
    </div>
    <div class="flex gap-2">
      <button @click="emit('done')"
              class="rounded-md bg-primary px-6 py-2 text-sm font-medium text-white hover:bg-primary-2 transition-colors">
        Start Learning
      </button>
      <button @click="emit('skip')"
              class="rounded-md border border-border px-4 py-2 text-sm text-fg-muted hover:bg-surface-2 transition-colors">
        Skip for now
      </button>
    </div>
  </div>
</template>
