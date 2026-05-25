<script setup lang="ts">
import { useRouter } from 'vue-router'
import { BookOpen, Upload } from 'lucide-vue-next'

const router = useRouter()
function go() { router.push({ path: '/library', query: { return: '/quiz' } }) }
</script>

<template>
  <div class="rounded-lg border border-warning/30 bg-warning-bg p-6 flex items-start gap-4">
    <BookOpen class="w-6 h-6 text-warning mt-1 shrink-0" />
    <div class="flex-1">
      <h3 class="text-base font-semibold text-fg">Quiz needs your study materials</h3>
      <p class="text-sm text-fg-muted mt-1">
        Upload a PDF in Library to start generating questions grounded in your sources.
      </p>
      <button @click="go"
              class="mt-3 rounded-md bg-warning px-4 py-2 text-sm font-medium text-bg hover:opacity-90 inline-flex items-center gap-2 transition-opacity">
        <Upload class="w-4 h-4" /> Upload PDF
      </button>
    </div>
  </div>
</template>
