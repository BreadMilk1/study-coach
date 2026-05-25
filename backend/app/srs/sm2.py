"""SM-2 lite spaced-repetition scheduler.

Used by the Quiz tool chain (`record_mistake`) to decide when a mistaken
question should resurface. Pure function — no DB, no time-of-day quirks.

SM-2 lite simplifies the original algorithm by deriving "repetition count"
implicitly from `previous_interval_days` (0 = brand new, 1 = first review,
≥ 6 = matured). This avoids carrying an explicit `repetitions` column on
the `mistakes` table.

Reference: SuperMemo 2 (Wozniak 1990). Ease floor 1.3 per SM-2 convention.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta


_EASE_FLOOR = 1.3


@dataclass(frozen=True)
class SrsSchedule:
    interval_days: int
    ease: float
    due_at: datetime


def _ease_delta(quality: int) -> float:
    """SM-2 ease update: Δ = 0.1 - (5-q) * (0.08 + (5-q) * 0.02)."""
    miss = 5 - quality
    return 0.1 - miss * (0.08 + miss * 0.02)


def next_schedule(
    *,
    quality: int,
    previous_interval_days: int = 0,
    previous_ease: float = 2.5,
    now: datetime | None = None,
) -> SrsSchedule:
    """Compute next review schedule.

    Args:
        quality: SM-2 grade 0-5. < 3 = wrong, ≥ 3 = correct.
        previous_interval_days: days between last review and this one (0 if first).
        previous_ease: ease factor from the last review (2.5 default per SM-2).
        now: review timestamp; defaults to utcnow().
    """
    now = now or datetime.utcnow()

    if quality < 3:
        new_interval = 1
    elif previous_interval_days == 0:
        new_interval = 1
    elif previous_interval_days == 1:
        new_interval = 6
    else:
        new_interval = round(previous_interval_days * previous_ease)

    new_ease = max(_EASE_FLOOR, previous_ease + _ease_delta(quality))

    return SrsSchedule(
        interval_days=new_interval,
        ease=new_ease,
        due_at=now + timedelta(days=new_interval),
    )
