import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'

import { describe, expect, it } from 'vitest'

describe('application lifecycle boundary', () => {
  it('does not mount the active route before the startup decision is ready', () => {
    const appSource = readFileSync(
      fileURLToPath(new URL('./App.vue', import.meta.url)),
      'utf8',
    )

    expect(appSource).toMatch(
      /<RouterView\s+v-if="lifecycle\.workspaceUnlocked"\s*\/>/,
    )
  })
})
