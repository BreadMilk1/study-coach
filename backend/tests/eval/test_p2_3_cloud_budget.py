"""Offline budget-ledger v1 cases. Does not read or write the real output ledger."""

import json

import pytest

from app.eval.p2_3_cloud_capability.budget import (
    LOCK_NAME,
    BudgetExceeded,
    BudgetLedger,
    BudgetLedgerCorrupt,
    MissingUsage,
)


def test_estimate_uses_max_tokens_worst_case():
    ledger = BudgetLedger.for_stage("main")
    est = ledger.estimate_cost(
        model_key="minimax:MiniMax-M3",
        input_tokens=2000,
        max_output_tokens=4096,
    )
    assert est == pytest.approx(2000 / 1e6 * 0.60 + 4096 / 1e6 * 2.40)


def test_refuse_when_estimate_exceeds_remaining():
    ledger = BudgetLedger.for_stage("main")
    ledger.record(actual_usd=4.99, usage={"input_tokens": 1, "output_tokens": 1})
    with pytest.raises(BudgetExceeded):
        ledger.check_before_call(
            model_key="minimax:MiniMax-M3",
            input_tokens=2000,
            max_output_tokens=4096,
        )


def test_estimate_includes_cache_read_tokens():
    ledger = BudgetLedger.for_stage("main")
    est = ledger.estimate_cost(
        model_key="minimax:MiniMax-M3",
        input_tokens=1,
        max_output_tokens=17,
        cache_read_tokens=385,
    )
    assert est == pytest.approx(1 / 1e6 * 0.60 + 17 / 1e6 * 2.40 + 385 / 1e6 * 0.12)


def test_missing_usage_blocks_matrix():
    ledger = BudgetLedger.for_stage("main")
    with pytest.raises(MissingUsage):
        ledger.record(actual_usd=0.01, usage=None)
    assert ledger.matrix_allowed is False


def test_second_ledger_restores_first_spend(tmp_path):
    first = BudgetLedger.for_stage("main", output_dir=tmp_path)
    reservation = first.check_before_call(
        model_key="minimax:MiniMax-M3",
        input_tokens=1000,
        max_output_tokens=1000,
    )
    first.record(
        actual_usd=0.02,
        usage={"input_tokens": 1000, "output_tokens": 1000},
        reservation_id=reservation,
    )
    second = BudgetLedger.for_stage("appendix", output_dir=tmp_path)
    assert second.spent_usd == pytest.approx(0.02)
    assert first.spent_usd == pytest.approx(0.02)


def _commit(ledger: BudgetLedger, usd: float) -> None:
    reservation = ledger.check_before_call(
        model_key="minimax:MiniMax-M3",
        input_tokens=1,
        max_output_tokens=1,
    )
    ledger.record(
        actual_usd=usd,
        usage={"input_tokens": 1, "output_tokens": 1},
        reservation_id=reservation,
    )


def test_combined_hard_cap_rejects_main_plus_appendix(tmp_path):
    main = BudgetLedger.for_stage("main", output_dir=tmp_path)
    _commit(main, 5.0)
    appendix = BudgetLedger.for_stage("appendix", output_dir=tmp_path)
    _commit(appendix, 4.99)
    with pytest.raises(BudgetExceeded):
        appendix.check_before_call(
            model_key="minimax:MiniMax-M3",
            input_tokens=2000,
            max_output_tokens=4096,
        )


def test_main_stage_cap_rejects_over_five(tmp_path):
    ledger = BudgetLedger.for_stage("main", output_dir=tmp_path)
    _commit(ledger, 4.99)
    with pytest.raises(BudgetExceeded):
        ledger.check_before_call(
            model_key="minimax:MiniMax-M3",
            input_tokens=2000,
            max_output_tokens=4096,
        )


def test_appendix_stage_cap_is_independent_of_main(tmp_path):
    ledger = BudgetLedger.for_stage("appendix", output_dir=tmp_path)
    _commit(ledger, 4.99)
    with pytest.raises(BudgetExceeded):
        ledger.check_before_call(
            model_key="minimax:MiniMax-M3",
            input_tokens=2000,
            max_output_tokens=4096,
        )
    restored = BudgetLedger.for_stage("main", output_dir=tmp_path)
    assert restored.spent_by_stage["main"] == pytest.approx(0.0)
    assert restored.spent_by_stage["appendix"] == pytest.approx(4.99)
    assert restored.spent_usd == pytest.approx(4.99)
    assert restored.stage_cap == pytest.approx(5.0)
    assert restored.hard_cap == pytest.approx(10.0)
    # The combined cap still has ~$5.01 left, so only the appendix stage cap can explain the refusal.
    assert restored.hard_cap - restored.spent_usd == pytest.approx(5.01)


def test_missing_usage_keeps_reservation(tmp_path):
    ledger = BudgetLedger.for_stage("main", output_dir=tmp_path)
    reservation = ledger.check_before_call(
        model_key="minimax:MiniMax-M3",
        input_tokens=4000,
        max_output_tokens=4096,
    )
    with pytest.raises(MissingUsage):
        ledger.record(actual_usd=0.0, usage=None, reservation_id=reservation)
    restored = BudgetLedger.for_stage("main", output_dir=tmp_path)
    assert restored.matrix_allowed is False
    assert restored.reserved_usd() > 0
    with pytest.raises(MissingUsage):
        restored.check_before_call(
            model_key="minimax:MiniMax-M3",
            input_tokens=1,
            max_output_tokens=1,
        )


def test_corrupt_ledger_fails_closed(tmp_path):
    path = tmp_path / "budget_ledger_v1.json"
    path.write_text("{not-json", encoding="utf-8")
    with pytest.raises(BudgetLedgerCorrupt):
        BudgetLedger.for_stage("main", output_dir=tmp_path)


def test_legacy_spend_is_unverifiable_not_seeded(tmp_path):
    ledger = BudgetLedger.for_stage("main", output_dir=tmp_path)
    payload = json.loads((tmp_path / "budget_ledger_v1.json").read_text())
    assert payload["legacy_spend"]["status"] == "unverifiable"
    assert payload["legacy_spend"]["usd"] is None
    assert payload["legacy_spend"]["usd"] != 0.55
    assert ledger.legacy_spend["usd"] is None


def test_valid_v1_load_does_not_rewrite_or_upgrade_ledger(tmp_path):
    """A valid v1 ledger loads read-only, without a rewrite or an upgrade to v2.

    The constructor is not a filesystem read-only API: taking the transaction lock
    creates the sibling lock file, so only the ledger file itself is asserted here.
    """

    _write_ledger(tmp_path)
    path = tmp_path / "budget_ledger_v1.json"
    before = _fingerprint(path)
    ledger = BudgetLedger.for_stage("main", output_dir=tmp_path)
    assert _fingerprint(path) == before
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "cloud-capability-budget-ledger-v1"
    assert payload["version"] == 1
    assert ledger.ledger_schema == "cloud-capability-budget-ledger-v1"
    assert ledger.version == 1
    assert ledger.spent_by_stage == {"main": 0.0, "appendix": 0.0}
    assert ledger.reservations == {}
    assert ledger.matrix_allowed is True
    assert (tmp_path / LOCK_NAME).exists()


_CHILD_DEADLINE = 10.0
_REAP_DEADLINE = 15.0


def _fingerprint(path):
    raw = path.read_bytes()
    return raw, path.stat().st_mtime_ns


def _concurrent_ledger_worker(output_dir: str, stage: str, usd: float, started, gate, errors) -> None:
    try:
        started.set()
        gate.wait(timeout=_CHILD_DEADLINE)
        from pathlib import Path

        import app.eval.p2_3_cloud_capability.budget as budget_module
        from app.eval.p2_3_cloud_capability.budget import BudgetLedger

        ledger = BudgetLedger.for_stage(stage, output_dir=Path(output_dir))
        reservation = ledger.check_before_call(
            model_key="minimax:MiniMax-M3",
            input_tokens=1,
            max_output_tokens=1,
        )
        ledger.record(
            actual_usd=usd,
            usage={"input_tokens": 1, "output_tokens": 1},
            reservation_id=reservation,
        )
        errors.put(("ok", str(Path(budget_module.__file__).resolve())))
    except Exception as exc:  # surfaced by the parent
        errors.put(("error", f"{type(exc).__name__}: {exc}"))


def _reap(proc) -> None:
    """Wait for a child within the deadline, then kill it only if it outlives it."""

    try:
        pid = proc.pid
    except ValueError:  # already closed
        return
    if pid is None:
        return
    proc.join(timeout=_REAP_DEADLINE)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=_REAP_DEADLINE)
    if proc.is_alive():
        proc.kill()
        proc.join(timeout=_REAP_DEADLINE)


def _drain_messages(errors, expected: int):
    import queue

    messages = []
    for _ in range(expected):
        try:
            messages.append(errors.get(timeout=_CHILD_DEADLINE))
        except queue.Empty:
            messages.append(("missing", "no message before deadline"))
            break
    return messages


def test_concurrent_commits_do_not_lose_updates(tmp_path):
    import multiprocessing as mp
    from pathlib import Path

    import app.eval.p2_3_cloud_capability.budget as budget_module

    expected_module = str(Path(budget_module.__file__).resolve())
    BudgetLedger.for_stage("main", output_dir=tmp_path)
    ctx = mp.get_context("spawn")
    started_main = ctx.Event()
    started_appendix = ctx.Event()
    gate = ctx.Event()
    errors = ctx.Queue()
    procs = [
        ctx.Process(
            target=_concurrent_ledger_worker,
            args=(str(tmp_path), "main", 0.03, started_main, gate, errors),
        ),
        ctx.Process(
            target=_concurrent_ledger_worker,
            args=(str(tmp_path), "appendix", 0.04, started_appendix, gate, errors),
        ),
    ]
    main_proc, appendix_proc = procs
    try:
        main_proc.start()
        appendix_proc.start()
        try:
            assert started_main.wait(timeout=_CHILD_DEADLINE)
            assert started_appendix.wait(timeout=_CHILD_DEADLINE)
        finally:
            gate.set()
        _reap(main_proc)
        _reap(appendix_proc)
        assert main_proc.exitcode == 0
        assert appendix_proc.exitcode == 0
        # Both children must load budget.py from this release tree.
        assert _drain_messages(errors, len(procs)) == [("ok", expected_module)] * len(procs)
        restored = BudgetLedger.for_stage("main", output_dir=tmp_path)
        assert restored.spent_usd == pytest.approx(0.07)
        assert restored.spent_by_stage["main"] == pytest.approx(0.03)
        assert restored.spent_by_stage["appendix"] == pytest.approx(0.04)
    finally:
        for proc in procs:
            _reap(proc)
        errors.close()


def _write_ledger(tmp_path, **overrides):
    payload = {
        "schema_version": "cloud-capability-budget-ledger-v1",
        "version": 1,
        "legacy_spend": {"status": "unverifiable", "usd": None},
        "spent_by_stage": {"main": 0.0, "appendix": 0.0},
        "reservations": {},
        "matrix_allowed": True,
    }
    payload.update(overrides)
    (tmp_path / "budget_ledger_v1.json").write_text(json.dumps(payload), encoding="utf-8")


def test_negative_persisted_spend_fails_closed(tmp_path):
    _write_ledger(tmp_path, spent_by_stage={"main": -0.01, "appendix": 0.0})
    with pytest.raises(BudgetLedgerCorrupt):
        BudgetLedger.for_stage("main", output_dir=tmp_path)


def test_nonfinite_persisted_spend_fails_closed(tmp_path):
    for value in (float("nan"), float("inf"), float("-inf")):
        _write_ledger(tmp_path, spent_by_stage={"main": value, "appendix": 0.0})
        with pytest.raises(BudgetLedgerCorrupt):
            BudgetLedger.for_stage("main", output_dir=tmp_path)


def test_malformed_reservation_fails_closed(tmp_path):
    _write_ledger(
        tmp_path,
        reservations={"r1": {"stage": "main", "status": "open", "usd": "cheap"}},
    )
    with pytest.raises(BudgetLedgerCorrupt):
        BudgetLedger.for_stage("main", output_dir=tmp_path)


def test_negative_or_nonfinite_actual_cost_fails_closed(tmp_path):
    ledger = BudgetLedger.for_stage("main", output_dir=tmp_path)
    for value in (-0.01, float("nan"), float("inf")):
        with pytest.raises(BudgetLedgerCorrupt):
            ledger.record(
                actual_usd=value,
                usage={"input_tokens": 1, "output_tokens": 1},
            )
        restored = BudgetLedger.for_stage("main", output_dir=tmp_path)
        assert restored.spent_usd == pytest.approx(0.0)


@pytest.mark.parametrize(
    "usage",
    [
        {},
        "nope",
        True,
        {"input_tokens": True, "output_tokens": 1},
        {"input_tokens": 1, "output_tokens": True},
        {"input_tokens": -1, "output_tokens": 1},
        {"input_tokens": 1, "output_tokens": -1},
        {"input_tokens": 1, "output_tokens": 1, "total_tokens": -1},
        {"input_tokens": 1, "output_tokens": 1, "total_tokens": True},
        {"input_tokens": 1, "output_tokens": 1, "cache_read_input_tokens": -1},
        {"input_tokens": 1, "output_tokens": 1, "cache_read_input_tokens": True},
        {"input_tokens": 0, "output_tokens": 0},
    ],
)
def test_malformed_usage_blocks_reloaded_ledger_before_releasing_reservation(tmp_path, usage):
    ledger = BudgetLedger.for_stage("main", output_dir=tmp_path)
    reservation = ledger.check_before_call(
        model_key="minimax:MiniMax-M3",
        input_tokens=4000,
        max_output_tokens=4096,
    )
    with pytest.raises(MissingUsage):
        ledger.record(actual_usd=0.01, usage=usage, reservation_id=reservation)
    assert ledger.matrix_allowed is False
    payload = json.loads((tmp_path / "budget_ledger_v1.json").read_text())
    assert payload["matrix_allowed"] is False
    assert reservation in payload["reservations"]
    restored = BudgetLedger.for_stage("main", output_dir=tmp_path)
    assert restored.matrix_allowed is False
    with pytest.raises(MissingUsage):
        restored.check_before_call(
            model_key="minimax:MiniMax-M3",
            input_tokens=1,
            max_output_tokens=1,
        )


def test_unpriceable_token_usage_and_actual_usd_fail_closed():
    ledger = BudgetLedger.for_stage("main")
    huge = 10**400
    with pytest.raises(BudgetLedgerCorrupt):
        ledger.estimate_cost(
            model_key="minimax:MiniMax-M3",
            input_tokens=huge,
            max_output_tokens=1,
        )
    with pytest.raises(BudgetLedgerCorrupt):
        ledger.estimate_cost(
            model_key="minimax:MiniMax-M3",
            input_tokens=1,
            max_output_tokens=huge,
        )
    with pytest.raises(BudgetLedgerCorrupt):
        ledger.estimate_cost(
            model_key="minimax:MiniMax-M3",
            input_tokens=1,
            max_output_tokens=1,
            cache_read_tokens=huge,
        )
    with pytest.raises(BudgetLedgerCorrupt):
        ledger.record(actual_usd=huge, usage={"input_tokens": 1, "output_tokens": 1})
    for usage in (
        {"input_tokens": huge, "output_tokens": 1},
        {"input_tokens": 1, "output_tokens": huge},
        {"input_tokens": 1, "output_tokens": 1, "total_tokens": huge},
        {"input_tokens": 1, "output_tokens": 1, "cache_read_input_tokens": huge},
    ):
        with pytest.raises((BudgetLedgerCorrupt, MissingUsage)):
            ledger.record(actual_usd=0.01, usage=usage)


def test_unpriceable_actual_usd_persists_blocked_and_keeps_reservation(tmp_path):
    ledger = BudgetLedger.for_stage("main", output_dir=tmp_path)
    reservation = ledger.check_before_call(
        model_key="minimax:MiniMax-M3",
        input_tokens=4000,
        max_output_tokens=4096,
    )
    with pytest.raises(BudgetLedgerCorrupt):
        ledger.record(
            actual_usd=10**400,
            usage={"input_tokens": 1, "output_tokens": 1},
            reservation_id=reservation,
        )
    payload = json.loads((tmp_path / "budget_ledger_v1.json").read_text())
    assert payload["matrix_allowed"] is False
    assert reservation in payload["reservations"]
    restored = BudgetLedger.for_stage("main", output_dir=tmp_path)
    assert restored.matrix_allowed is False
    assert reservation in restored.reservations
    with pytest.raises(MissingUsage):
        restored.check_before_call(
            model_key="minimax:MiniMax-M3",
            input_tokens=1,
            max_output_tokens=1,
        )


def test_locked_does_not_overwrite_unreadable_ledger(tmp_path):
    ledger = BudgetLedger.for_stage("main", output_dir=tmp_path)
    reservation = ledger.check_before_call(
        model_key="minimax:MiniMax-M3",
        input_tokens=1,
        max_output_tokens=1,
    )
    path = tmp_path / "budget_ledger_v1.json"
    original = "{not-json"
    path.write_text(original, encoding="utf-8")
    with pytest.raises(BudgetLedgerCorrupt):
        ledger.record(
            actual_usd=0.01,
            usage={"input_tokens": 1, "output_tokens": 1},
            reservation_id=reservation,
        )
    assert path.read_text(encoding="utf-8") == original


def test_main_actual_spend_over_five_persists_and_blocks(tmp_path):
    ledger = BudgetLedger.for_stage("main", output_dir=tmp_path)
    with pytest.raises(BudgetExceeded):
        _commit(ledger, 5.01)
    payload = json.loads((tmp_path / "budget_ledger_v1.json").read_text())
    assert payload["spent_by_stage"]["main"] == pytest.approx(5.01)
    assert payload["matrix_allowed"] is False
    restored = BudgetLedger.for_stage("main", output_dir=tmp_path)
    assert restored.spent_usd == pytest.approx(5.01)
    assert restored.matrix_allowed is False
    with pytest.raises((BudgetExceeded, MissingUsage)):
        restored.check_before_call(
            model_key="minimax:MiniMax-M3",
            input_tokens=1,
            max_output_tokens=1,
        )
