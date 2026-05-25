"""Cut ④a — SM-2 lite scheduler tests.

Domain: spaced-repetition scheduling for the Mistake Bank. A learner who
got a quiz question wrong sees it again on a schedule that lengthens with
each successful recall, shortens on a slip-up.

Tested function: next_schedule(quality, previous_interval_days, previous_ease, now?)
returning SrsSchedule(interval_days, ease, due_at).

Quality grades follow SM-2 0-5 convention:
  0-2 = wrong (reset interval to 1, lower ease)
  3-5 = correct (ramp interval: 0 → 1, 1 → 6, otherwise prev * ease)
"""
from datetime import datetime, timedelta

import pytest


# We use approx for floats since SM-2 ease updates produce non-terminating decimals.
APPROX = pytest.approx


def test_first_time_correct_sets_interval_to_one_and_bumps_ease():
    from app.srs.sm2 import next_schedule

    schedule = next_schedule(quality=4)

    assert schedule.interval_days == 1
    # SM-2 ease delta for quality=4: 0.1 - (5-4)*(0.08 + (5-4)*0.02) = 0.1 - 0.10 = 0.0
    assert schedule.ease == APPROX(2.5)


def test_first_time_wrong_resets_interval_and_lowers_ease():
    from app.srs.sm2 import next_schedule

    schedule = next_schedule(quality=0)

    assert schedule.interval_days == 1
    # ease delta for quality=0: 0.1 - 5*(0.08 + 5*0.02) = 0.1 - 0.9 = -0.8 → 2.5 - 0.8 = 1.7
    assert schedule.ease == APPROX(1.7)


def test_second_correct_after_one_day_advances_to_six_days():
    from app.srs.sm2 import next_schedule

    schedule = next_schedule(
        quality=5,
        previous_interval_days=1,
        previous_ease=2.5,
    )

    assert schedule.interval_days == 6
    # quality=5: 0.1 - 0 = 0.1 → 2.5 + 0.1 = 2.6
    assert schedule.ease == APPROX(2.6)


def test_third_correct_multiplies_by_ease():
    from app.srs.sm2 import next_schedule

    schedule = next_schedule(
        quality=4,
        previous_interval_days=6,
        previous_ease=2.5,
    )

    assert schedule.interval_days == 15  # round(6 * 2.5)
    assert schedule.ease == APPROX(2.5)


def test_repeated_wrong_floors_ease_at_one_point_three():
    from app.srs.sm2 import next_schedule

    # Already close to floor; another wrong should clamp.
    schedule = next_schedule(
        quality=0,
        previous_interval_days=6,
        previous_ease=1.4,
    )

    assert schedule.interval_days == 1
    assert schedule.ease == APPROX(1.3)


def test_due_at_equals_now_plus_interval_days():
    from app.srs.sm2 import next_schedule

    fixed_now = datetime(2026, 5, 21, 10, 0)
    schedule = next_schedule(
        quality=5,
        previous_interval_days=6,
        previous_ease=2.5,
        now=fixed_now,
    )

    assert schedule.interval_days == 15
    assert schedule.due_at == fixed_now + timedelta(days=15)
