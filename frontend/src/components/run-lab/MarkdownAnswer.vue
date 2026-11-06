<script setup lang="ts">
import { computed } from 'vue'
import DOMPurify from 'dompurify'

import { parseAnswerMarkdown } from './answerMarkdown'

const props = defineProps<{ answer: string }>()
const sanitizedHtml = computed(() => DOMPurify.sanitize(parseAnswerMarkdown(props.answer), {
  ALLOWED_TAGS: [
    'p', 'br', 'strong', 'em', 'del', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
    'ul', 'ol', 'li', 'pre', 'code', 'blockquote', 'hr', 'table', 'thead',
    'tbody', 'tr', 'th', 'td', 'a',
  ],
  ALLOWED_ATTR: ['href', 'title', 'start'],
  ALLOW_DATA_ATTR: false,
  ALLOW_ARIA_ATTR: false,
  ALLOWED_URI_REGEXP: /^https?:\/\//i,
}))
</script>

<template>
  <div class="markdown-answer" v-html="sanitizedHtml" />
</template>

<style scoped>
.markdown-answer { min-width: 0; overflow-wrap: anywhere; line-height: 1.75; }
.markdown-answer :deep(> :first-child) { margin-top: 0; }
.markdown-answer :deep(> :last-child) { margin-bottom: 0; }
.markdown-answer :deep(p), .markdown-answer :deep(ul), .markdown-answer :deep(ol),
.markdown-answer :deep(blockquote), .markdown-answer :deep(pre), .markdown-answer :deep(table) { margin: 0 0 1rem; }
.markdown-answer :deep(h1), .markdown-answer :deep(h2), .markdown-answer :deep(h3),
.markdown-answer :deep(h4), .markdown-answer :deep(h5), .markdown-answer :deep(h6) {
  margin: 1.5rem 0 .65rem; font-weight: 600; line-height: 1.4; color: var(--color-fg);
}
.markdown-answer :deep(h1) { font-size: 1.3rem; }
.markdown-answer :deep(h2) { font-size: 1.15rem; }
.markdown-answer :deep(h3) { font-size: 1.05rem; }
.markdown-answer :deep(ul) { list-style: disc; padding-left: 1.5rem; }
.markdown-answer :deep(ol) { list-style: decimal; padding-left: 1.5rem; }
.markdown-answer :deep(li + li) { margin-top: .35rem; }
.markdown-answer :deep(strong) { color: var(--color-fg); font-weight: 600; }
.markdown-answer :deep(a) { color: var(--color-primary-2); text-decoration: underline; text-underline-offset: .2em; }
.markdown-answer :deep(blockquote) { border-left: 2px solid var(--color-border-strong); padding-left: 1rem; }
.markdown-answer :deep(code) { font-family: var(--font-mono); font-size: .875em; background: var(--color-surface-2); padding: .1rem .25rem; }
.markdown-answer :deep(pre) { max-width: 100%; overflow-x: auto; padding: 1rem; background: var(--color-surface-2); border: 1px solid var(--color-border-strong); }
.markdown-answer :deep(pre code) { padding: 0; white-space: pre; overflow-wrap: normal; }
.markdown-answer :deep(table) { width: 100%; table-layout: fixed; border-collapse: collapse; font-size: .875em; }
.markdown-answer :deep(th), .markdown-answer :deep(td) { border: 1px solid var(--color-border-strong); padding: .5rem .65rem; text-align: left; vertical-align: top; }
.markdown-answer :deep(th) { color: var(--color-fg); font-weight: 600; }
.markdown-answer :deep(hr) { border: 0; border-top: 1px solid var(--color-border-strong); margin: 1.5rem 0; }
</style>
