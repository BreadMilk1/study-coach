import { createI18n } from 'vue-i18n'
import en from './locales/en.json'
import zhCN from './locales/zh-CN.json'

function savedLocale(): string {
  try {
    const raw = localStorage.getItem('study-coach:settings')
    if (raw) return JSON.parse(raw).language || 'en'
  } catch { /* empty */ }
  return 'en'
}

export const i18n = createI18n({
  legacy: false,
  locale: savedLocale(),
  fallbackLocale: 'en',
  messages: { en, 'zh-CN': zhCN },
})
