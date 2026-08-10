<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { uploadDocument } from '../lib/api'
import { useDocuments } from '../stores/documents'

const ALLOWED_RETURNS = new Set(['/quiz', '/chat', '/plan', '/'])

const docs = useDocuments()
const route = useRoute()
const router = useRouter()
const status = ref<string>('')
const uploading = ref(false)
const lastDoc = ref<{ filename: string; chunks_count: number } | null>(null)
let active = true

onBeforeUnmount(() => {
  active = false
})

const returnPath = computed(() => {
  const raw = route.query.return
  const value = Array.isArray(raw) ? raw[0] : raw
  if (typeof value !== 'string') return null
  return ALLOWED_RETURNS.has(value) ? value : null
})

const returnTopic = computed(() => {
  const raw = route.query.topic
  const value = Array.isArray(raw) ? raw[0] : raw
  return typeof value === 'string' && value ? value : null
})

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
    if (!active) return
    lastDoc.value = { filename: res.filename, chunks_count: res.chunks_count }
    status.value = `Indexed ${res.chunks_count} chunks from ${res.filename}.`
    await docs.fetch()
    if (!active) return
    if (returnPath.value) {
      if (returnPath.value === '/quiz' && returnTopic.value) {
        await router.push({ path: '/quiz', query: { topic: returnTopic.value } })
      } else {
        await router.push(returnPath.value)
      }
      return
    }
  } catch (err) {
    if (!active) return
    status.value = `Upload failed: ${err}`
  } finally {
    if (active) uploading.value = false
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
    <div v-if="lastDoc && !returnPath" class="mt-4 flex flex-wrap gap-2">
      <RouterLink
        to="/chat"
        class="rounded-md bg-indigo-500 px-3 py-2 text-sm font-medium text-white hover:bg-indigo-400"
      >Open Chat</RouterLink>
      <RouterLink
        to="/quiz"
        class="rounded-md border border-indigo-400/40 px-3 py-2 text-sm font-medium text-indigo-200 hover:bg-indigo-400/10"
      >Start Quiz</RouterLink>
    </div>
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
