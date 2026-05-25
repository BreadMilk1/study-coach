<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { uploadDocument } from '../lib/api'
import { useDocuments } from '../stores/documents'

const docs = useDocuments()
const status = ref<string>('')
const uploading = ref(false)
const lastDoc = ref<{ filename: string; chunks_count: number } | null>(null)

onMounted(() => {
  docs.fetch()
})

async function onFile(e: Event) {
  const f = (e.target as HTMLInputElement).files?.[0]
  if (!f) return
  uploading.value = true
  status.value = `Uploading ${f.name}…`
  try {
    const res = await uploadDocument(f)
    lastDoc.value = { filename: res.filename, chunks_count: res.chunks_count }
    status.value = `Indexed ${res.chunks_count} chunks from ${res.filename}.`
    await docs.fetch()
  } catch (e) {
    status.value = `Upload failed: ${e}`
  } finally {
    uploading.value = false
  }
}
</script>

<template>
  <div class="p-8 max-w-2xl">
    <h2 class="text-xl font-semibold mb-4">Library</h2>
    <p class="text-white/60 text-sm mb-6">Upload a PDF to index into your local corpus.</p>
    <label class="block">
      <input type="file" accept="application/pdf" @change="onFile" :disabled="uploading"
             class="block w-full text-sm file:mr-3 file:py-2 file:px-4 file:rounded-lg file:border-0
                    file:bg-indigo-500 file:text-white hover:file:bg-indigo-400" />
    </label>
    <div v-if="status" class="mt-6 text-sm text-white/70">{{ status }}</div>
    <div v-if="lastDoc" class="mt-2 text-xs text-white/40">{{ lastDoc.filename }} → {{ lastDoc.chunks_count }} chunks</div>
    <div class="mt-8">
      <h3 class="text-sm font-medium text-white/80 mb-3">Indexed PDFs</h3>
      <div v-if="docs.loading" class="text-sm text-white/50">Loading…</div>
      <div v-else-if="docs.error" class="text-sm text-rose-300">{{ docs.error }}</div>
      <div v-else-if="docs.docs.length === 0" class="text-sm text-white/40">
        No PDFs indexed yet.
      </div>
      <ul v-else class="space-y-2">
        <li v-for="doc in docs.docs" :key="doc.id"
            class="rounded-lg border border-white/10 bg-white/5 px-4 py-3 flex items-center justify-between gap-4">
          <span class="text-sm text-white/80 break-all">{{ doc.filename }}</span>
          <span class="text-xs text-white/50 font-mono whitespace-nowrap">{{ doc.chunks_count }} chunks</span>
        </li>
      </ul>
    </div>
  </div>
</template>
