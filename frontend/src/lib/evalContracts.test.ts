import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { describe, expect, it } from 'vitest'

import {
  parseCompareResponse,
  parseEvalErrorDetail,
  parseEvalEvent,
  parseRunDetail,
} from './evalContracts'

function readExample(name: string): string {
  return readFileSync(
    fileURLToPath(new URL(`../../../contracts/eval-api-v1/examples/${name}`, import.meta.url)),
    'utf8',
  )
}

describe('eval-api-v1 examples', () => {
  it('accepts every backend-approved stream event', () => {
    const lines = readExample('run-stream.jsonl').trim().split('\n')
    const types = lines.map(line => parseEvalEvent(JSON.parse(line)).type)
    expect(types).toEqual([
      'run_created',
      'stage_started',
      'stage_completed',
      'score_set_created',
      'scorer_completed',
      'score_set_finished',
      'run_finished',
    ])
  })

  it('accepts the approved run detail example', () => {
    const detail = parseRunDetail(JSON.parse(readExample('run-detail.json')))
    expect(detail.summary.run_id).toBe('run-example-001')
    expect(detail.summary.lifecycle).toBe('finished')
  })

  it('accepts the approved compare example and fills schema defaults', () => {
    const compare = parseCompareResponse(JSON.parse(readExample('compare-controlled.json')))
    expect(compare.compatibility).toBe('controlled')
    expect(compare.caption).toBe('case delta')
    expect(compare.rescore_required).toBe(false)
    expect(compare.scope).toBe('case')
  })

  it('accepts the approved busy error example without flattening it', () => {
    const error = parseEvalErrorDetail(JSON.parse(readExample('evaluation-busy.json')))
    expect(error).toEqual({
      code: 'evaluation_busy',
      message: 'another evaluation is already running',
      fields: [],
      active_entity_id: 'run-active-001',
      active_kind: 'run',
    })
  })
})

describe('eval contracts fail closed', () => {
  it('rejects an unknown schema version', () => {
    expect(() => parseEvalEvent({
      schema_version: 'eval-api-v0',
      type: 'run_created',
      run_id: 'run-1',
    })).toThrowError(/unknown eval schema version/)
  })

  it('rejects an unknown event type', () => {
    expect(() => parseEvalEvent({
      schema_version: 'eval-api-v1',
      type: 'token',
      run_id: 'run-1',
    })).toThrowError(/unknown eval event type/)
  })
})
