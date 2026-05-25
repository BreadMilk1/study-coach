"""Cut ④g — disclaimer text bug fix.

Quiz path degrades immediately (no retry) per Cut ④e, so the disclaimer's
"after N retries" wording is misleading for quiz weak verdicts. The helper
now takes retry_count and omits the retries clause when it's 0.
"""
from app.agent.graph import _degrade_disclaimer


def test_degrade_disclaimer_omits_retries_when_count_is_zero():
    text = _degrade_disclaimer(0.42, ["question_quality"], retry_count=0)
    lower = text.lower()
    assert "retries" not in lower
    assert "0.42" in text
    assert "question_quality" in text


def test_degrade_disclaimer_includes_retries_when_count_is_nonzero():
    text = _degrade_disclaimer(0.40, ["accuracy"], retry_count=2)
    assert "after 2 retries" in text
    assert "0.40" in text
    assert "accuracy" in text
