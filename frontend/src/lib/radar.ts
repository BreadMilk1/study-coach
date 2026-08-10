/** Normalize mastery API fields into the 0..1 radar axes. */

export interface RadarInputs {
  scores: { score: number }[]
  planMilestoneDone: number
  planMilestoneTotal: number
  quizAccuracy: number
  streakDays: number
  coverage: number
}

export interface RadarAxes {
  mastery: number
  planProgress: number
  quizAccuracy: number
  streak: number
  coverage: number
}

/** Streak axis: 7 active days → full scale. */
export const STREAK_FULL_DAYS = 7

export function computeRadarAxes(input: RadarInputs): RadarAxes {
  const mastery =
    input.scores.length === 0
      ? 0
      : input.scores.reduce((a, s) => a + s.score, 0) / input.scores.length
  const planProgress =
    input.planMilestoneTotal === 0
      ? 0
      : input.planMilestoneDone / input.planMilestoneTotal
  return {
    mastery,
    planProgress,
    quizAccuracy: Math.min(1, Math.max(0, input.quizAccuracy)),
    streak: Math.min(1, Math.max(0, input.streakDays) / STREAK_FULL_DAYS),
    coverage: Math.min(1, Math.max(0, input.coverage)),
  }
}
