<script setup lang="ts">
import { ref, onBeforeUnmount, onMounted, nextTick, useTemplateRef } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useChat } from '../stores/chat'
import { useSettings } from '../stores/settings'
import { streamChat } from '../lib/api'
import TracePanel from '../components/TracePanel.vue'

const chat = useChat()
const settings = useSettings()
const route = useRoute()
const router = useRouter()
const input = ref('')
const scrollEl = useTemplateRef<HTMLDivElement>('scrollEl')
let active = true

onBeforeUnmount(() => {
  active = false
})

onMounted(async () => {
  await chat.restoreCurrentSession()
  if (!active) return
  const autoText = route.query.auto as string | undefined
  if (autoText) {
    input.value = autoText
    send()
    router.replace({ query: { goal_id: route.query.goal_id } })
  }
})

async function send() {
  const text = input.value.trim()
  if (!text || chat.streaming) return
  chat.pushUser(text)
  input.value = ''
  const assistant = chat.startAssistant()
  await nextTick()
  scrollEl.value?.scrollTo({ top: scrollEl.value.scrollHeight })
  await streamChat(text, settings.$state, {
    onSession: (sessionId) => chat.setSessionId(sessionId),
    onCitations: (cs) => chat.setCitations(assistant, cs),
    onAgentRun: (run) => chat.setAgentRun(assistant, run),
    onToken: (t) => {
      chat.appendToken(assistant, t)
      scrollEl.value?.scrollTo({ top: scrollEl.value.scrollHeight })
    },
    onTrace: (step) => chat.trace.push(step),
    onDone: () => chat.finish(),
    onError: (e) => {
      chat.appendToken(assistant, `\n[error: ${e}]`)
      chat.finish()
    },
  })
}
</script>

<template>
  <div class="h-full flex flex-col">
    <div ref="scrollEl" class="flex-1 overflow-y-auto p-6 space-y-4 max-md:p-3">
      <div v-if="chat.messages.length === 0" class="text-white/40 text-center mt-20">
        Upload a PDF in <RouterLink to="/library" class="underline">Library</RouterLink>, then ask a question.
      </div>
      <div v-for="m in chat.messages" :key="m.id" class="max-w-3xl mx-auto max-md:max-w-full">
        <div :class="m.role === 'user' ? 'text-right' : ''">
          <div :class="[
            'inline-block px-4 py-3 rounded-2xl text-sm leading-relaxed whitespace-pre-wrap',
            m.role === 'user' ? 'bg-indigo-600/30' : 'bg-white/5'
          ]">{{ m.content || (chat.streaming && m === chat.messages[chat.messages.length - 1] ? '…' : '') }}</div>
        </div>
        <div v-if="m.role === 'assistant' && m.citations && m.citations.length"
             class="mt-2 flex flex-wrap gap-2 max-w-3xl">
          <span v-for="(c, i) in m.citations" :key="c.chunk_id"
                class="text-xs px-2 py-1 rounded-md bg-indigo-400/10 text-indigo-200 border border-indigo-400/20">
            [{{ i + 1 }}] {{ c.source }} · p.{{ c.page }}
          </span>
        </div>
      </div>
    </div>
    <form @submit.prevent="send" class="border-t border-white/5 p-4 flex gap-2 max-md:fixed max-md:bottom-14 max-md:left-0 max-md:right-0 max-md:px-3 max-md:py-2 max-md:bg-bg">
      <input v-model="input" :disabled="chat.streaming" :placeholder="$t('chat.placeholder')"
             class="flex-1 bg-white/5 px-4 py-2 rounded-lg outline-none border border-transparent focus:border-indigo-400/40" />
      <button type="submit" :disabled="chat.streaming || !input.trim()"
              class="px-4 py-2 rounded-lg bg-indigo-500 hover:bg-indigo-400 disabled:opacity-40 disabled:cursor-not-allowed max-md:min-h-12 max-md:min-w-12">
        {{ chat.streaming ? '…' : $t('chat.send') }}
      </button>
    </form>
    <TracePanel v-if="settings.debugMode" />
  </div>
</template>
