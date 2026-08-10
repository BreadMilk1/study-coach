import { expect, test } from 'vitest'

import { computeRadarAxes, STREAK_FULL_DAYS } from './radar.ts'

test('uses API streak_days and coverage instead of placeholders', () => {
  const axes = computeRadarAxes({
    scores: [{ score: 0.8 }, { score: 0.4 }],
    planMilestoneDone: 1,
    planMilestoneTotal: 4,
    quizAccuracy: 0.6,
    streakDays: 7,
    coverage: 0.42,
  })

  expect(axes.mastery).toBeCloseTo(0.6)
  expect(axes.planProgress).toBeCloseTo(0.25)
  expect(axes.quizAccuracy).toBeCloseTo(0.6)
  expect(axes.streak).toBe(1)
  expect(axes.coverage).toBeCloseTo(0.42)
})

test('caps streak at 1 and zeros empty plan', () => {
  const axes = computeRadarAxes({
    scores: [],
    planMilestoneDone: 0,
    planMilestoneTotal: 0,
    quizAccuracy: 0,
    streakDays: STREAK_FULL_DAYS * 3,
    coverage: 1.5,
  })

  expect(axes.mastery).toBe(0)
  expect(axes.planProgress).toBe(0)
  expect(axes.streak).toBe(1)
  expect(axes.coverage).toBe(1)
})
