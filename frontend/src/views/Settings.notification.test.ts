import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'

import { describe, expect, it } from 'vitest'

describe('Settings save feedback', () => {
  it('publishes a localized success notification after persisting', () => {
    const source = readFileSync(
      fileURLToPath(new URL('./Settings.vue', import.meta.url)),
      'utf8',
    )

    expect(source).toContain("import { useNotifications } from '../stores/notifications'")
    expect(source).toMatch(/function save\(\) \{\s*s\.persist\(\)\s*notifications\.push\(\{\s*kind: 'success',\s*message: t\('settings\.saved'\),\s*\}\)\s*\}/)
  })
})
