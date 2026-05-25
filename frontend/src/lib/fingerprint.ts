// Fingerprint: minimal device identity for Phase 1.
// Stores a per-browser UUID in localStorage; real FingerprintJS comes in P3.

const KEY = 'study-coach:fingerprint'

export function getFingerprint(): string {
  let v = localStorage.getItem(KEY)
  if (!v) {
    v = crypto.randomUUID()
    localStorage.setItem(KEY, v)
  }
  return v
}
