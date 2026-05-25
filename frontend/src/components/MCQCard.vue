<script setup lang="ts">
import { ref } from 'vue'

const props = defineProps<{ prompt: string; options: string[] }>()
const emit = defineEmits<{ (e: 'submit', choice: string): void }>()

const selected = ref<string | null>(null)
const submitted = ref(false)

function submit() {
  if (!selected.value || submitted.value) return
  submitted.value = true
  emit('submit', selected.value)
}
</script>

<template>
  <div class="rounded-lg border border-border bg-surface p-6">
    <p class="text-base font-medium mb-4">{{ prompt }}</p>
    <div role="radiogroup" aria-label="Options" class="flex flex-col gap-2">
      <label v-for="opt in options" :key="opt"
             :class="[
               'flex items-start gap-3 rounded-md border p-3 cursor-pointer transition-colors',
               selected === opt[0]
                 ? 'border-primary-ring bg-primary-bg'
                 : 'border-border hover:bg-white/5'
             ]">
        <input type="radio" name="mcq" :value="opt[0]" v-model="selected" :disabled="submitted"
               class="mt-1 accent-primary" />
        <span class="text-sm">{{ opt }}</span>
      </label>
    </div>
    <button @click="submit" :disabled="!selected || submitted"
            class="mt-4 rounded-md bg-primary px-4 py-2 text-sm font-medium text-white hover:bg-primary-2 disabled:opacity-40 transition-colors">
      Submit
    </button>
  </div>
</template>
