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

  it('keeps lifecycle dialogs synchronized with store state when Escape is pressed', () => {
    const startupGateSource = readFileSync(
      fileURLToPath(new URL('./components/StartupDataGate.vue', import.meta.url)),
      'utf8',
    )
    const resetDialogSource = readFileSync(
      fileURLToPath(new URL('./components/ResetConfirmDialog.vue', import.meta.url)),
      'utf8',
    )

    expect(startupGateSource).toMatch(
      /@keydown\.esc\.prevent\.stop="preventCancel"/,
    )
    expect(resetDialogSource).toMatch(
      /@keydown\.esc\.prevent\.stop="handleCancel"/,
    )
  })

  it('keeps a desktop path into Run Lab', () => {
    const appSource = readFileSync(
      fileURLToPath(new URL('./App.vue', import.meta.url)),
      'utf8',
    )
    expect(appSource).toMatch(/to:\s*'\/run-lab'/)
  })
})
