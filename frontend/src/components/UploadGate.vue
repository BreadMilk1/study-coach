<script setup lang="ts">
import { useRouter } from 'vue-router'
import { Upload } from 'lucide-vue-next'
const router = useRouter()
</script>
<template>
  <div class="rounded-lg border border-warning/30 bg-warning-bg p-4 mb-6 flex items-center gap-3">
    <Upload class="w-5 h-5 text-warning" />
    <p class="text-sm flex-1">No sources uploaded yet. Upload a PDF to start tracking.</p>
    <button @click="router.push('/library')"
            class="rounded-md bg-warning px-3 py-1.5 text-xs font-medium text-bg hover:opacity-90">
      Upload PDF
    </button>
  </div>
</template>
