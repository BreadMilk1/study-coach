"""Offline v1/v2 ledger fixtures. Does not read or write the real output ledger."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from app.eval.p2_3_cloud_capability.budget import (
    LEDGER_NAME,
    LOCK_NAME,
    BudgetExceeded,
    BudgetLedger,
    BudgetLedgerCorrupt,
    MissingUsage,
)

_KEYS = ("s066a4d2f779e9c1e", "s19f35a2c8949e11b", "se68ebcb769e9f2ac")
_SNAPSHOT_ID = "a" * 64


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _pins_for(blobs: dict[str, tuple[bytes, bytes]]) -> dict[str, dict[str, str]]:
    return {
        key: {"candidate_sha256": _sha(cand), "sqlite_sha256": _sha(db)}
        for key, (cand, db) in blobs.items()
    }


def _sample_blobs() -> dict[str, tuple[bytes, bytes]]:
    return {key: (f"candidate:{key}".encode(), f"sqlite:{key}".encode()) for key in _KEYS}


def _v2_document(originals: dict[str, dict[str, str]], **overrides: object) -> dict:
    document = {
        "schema_version": "cloud-capability-budget-ledger-v2",
        "version": 67,
        "legacy_spend": {"status": "unverifiable", "usd": None},
        "spent_by_stage": {"main": 0.0, "appendix": 0.06980472},
        "reservations": {
            "56b160d3c3dd4046a5ea10749e58f644": {
                "stage": "appendix",
                "status": "open",
                "usd": 0.0122304,
            }
        },
        "matrix_allowed": False,
        "isolated_storage_keys": list(_KEYS),
        "isolated_originals": originals,
        "source_snapshot_id": _SNAPSHOT_ID,
        "settlements": {},
    }
    document.update(overrides)
    return document


def _write_document(directory: Path, document: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / LEDGER_NAME
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _fingerprint(path: Path) -> tuple[bytes, int]:
    """Ledger bytes plus mtime: a rejection must not rewrite the document."""

    raw = path.read_bytes()
    return raw, path.stat().st_mtime_ns


def _assert_isolation_fields(payload: dict, originals: dict[str, dict[str, str]]) -> None:
    assert payload["isolated_storage_keys"] == list(_KEYS)
    assert payload["isolated_originals"] == originals
    assert payload["source_snapshot_id"] == _SNAPSHOT_ID
    assert payload["settlements"] == {}


def _v1_document(**overrides: object) -> dict:
    document = {
        "schema_version": "cloud-capability-budget-ledger-v1",
        "version": 66,
        "legacy_spend": {"status": "unverifiable", "usd": None},
        "spent_by_stage": {"main": 0.0, "appendix": 0.06980472},
        "reservations": {},
        "matrix_allowed": False,
    }
    document.update(overrides)
    return document


def test_v2_failure_persist_keeps_isolation_fields_and_unknown_cost(tmp_path):
    originals = _pins_for(_sample_blobs())
    path = _write_document(tmp_path, _v2_document(originals))
    before = path.read_bytes()
    ledger = BudgetLedger.for_stage("appendix", output_dir=tmp_path)
    assert path.read_bytes() == before
    assert ledger.legacy_spend["usd"] is None
    assert ledger.legacy_spend["usd"] != 0

    with pytest.raises(MissingUsage):
        ledger.record(actual_usd=0.01, usage=None)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "cloud-capability-budget-ledger-v2"
    assert payload["isolated_storage_keys"] == list(_KEYS)
    assert payload["isolated_originals"] == originals
    assert payload["source_snapshot_id"] == _SNAPSHOT_ID
    assert payload["settlements"] == {}
    assert payload["legacy_spend"]["status"] == "unverifiable"
    assert payload["legacy_spend"]["usd"] is None
    assert payload["spent_by_stage"]["appendix"] == pytest.approx(0.06980472)
    assert payload["matrix_allowed"] is False
    assert payload["version"] == 68
    restored = BudgetLedger.for_stage("appendix", output_dir=tmp_path)
    assert restored.legacy_spend["usd"] is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("isolated_storage_keys", list(_KEYS)),
        (
            "isolated_originals",
            {key: {"candidate_sha256": "a" * 64, "sqlite_sha256": "b" * 64} for key in _KEYS},
        ),
        ("source_snapshot_id", _SNAPSHOT_ID),
        ("settlements", {}),
    ],
)
def test_v1_document_with_v2_field_is_rejected_without_rewrite(tmp_path, field, value):
    document = _v1_document(**{field: value})
    path = _write_document(tmp_path, document)
    before = _fingerprint(path)
    with pytest.raises(BudgetLedgerCorrupt, match=field):
        BudgetLedger.for_stage("appendix", output_dir=tmp_path)
    assert _fingerprint(path) == before
    assert json.loads(path.read_text(encoding="utf-8"))["version"] == 66


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        ("bad-sha", "candidate_sha256"),
        ("missing-snapshot", "source_snapshot_id"),
        ("nonempty-settlements", "settlements"),
        ("uppercase-sha", "sqlite_sha256"),
        ("wrong-keys", "isolated_storage_keys"),
        ("missing-keys", "isolated_storage_keys"),
        ("missing-originals", "isolated_originals"),
        ("missing-settlements", "settlements"),
        ("missing-sha-member", "candidate_sha256 or sqlite_sha256"),
        ("non-string-sha", "candidate_sha256"),
    ],
)
def test_malformed_v2_is_rejected_without_rewrite(tmp_path, mutate, match):
    originals = _pins_for(_sample_blobs())
    document = _v2_document(originals)
    if mutate == "bad-sha":
        document["isolated_originals"][_KEYS[0]]["candidate_sha256"] = "ab"
    elif mutate == "missing-snapshot":
        del document["source_snapshot_id"]
    elif mutate == "nonempty-settlements":
        document["settlements"] = {
            "56b160d3c3dd4046a5ea10749e58f644": {
                "idempotency_key": "same",
                "evidence_sha256": "b" * 64,
                "actual_usd": 0.01,
                "usage_digest": "digest",
            }
        }
    elif mutate == "uppercase-sha":
        document["isolated_originals"][_KEYS[0]]["sqlite_sha256"] = "A" * 64
    elif mutate == "wrong-keys":
        document["isolated_storage_keys"] = list(_KEYS[:2])
    elif mutate == "missing-keys":
        del document["isolated_storage_keys"]
    elif mutate == "missing-originals":
        del document["isolated_originals"]
    elif mutate == "missing-settlements":
        del document["settlements"]
    elif mutate == "missing-sha-member":
        del document["isolated_originals"][_KEYS[0]]["sqlite_sha256"]
    elif mutate == "non-string-sha":
        document["isolated_originals"][_KEYS[0]]["candidate_sha256"] = 5
    path = _write_document(tmp_path, document)
    before = _fingerprint(path)
    with pytest.raises(BudgetLedgerCorrupt, match=match):
        BudgetLedger.for_stage("appendix", output_dir=tmp_path)
    # A refused load persists nothing: bytes, mtime and the on-disk version stay put.
    assert _fingerprint(path) == before
    assert json.loads(path.read_text(encoding="utf-8"))["version"] == 67


def test_v2_load_of_valid_document_does_not_rewrite_or_upgrade_ledger(tmp_path):
    """A valid v2 ledger loads read-only, without a rewrite or a schema change.

    The constructor is not a filesystem read-only API: taking the transaction lock
    creates the sibling lock file, so only the ledger file itself is asserted here.
    """

    originals = _pins_for(_sample_blobs())
    path = _write_document(tmp_path, _v2_document(originals))
    before = _fingerprint(path)
    ledger = BudgetLedger.for_stage("appendix", output_dir=tmp_path)
    assert _fingerprint(path) == before
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "cloud-capability-budget-ledger-v2"
    assert payload["version"] == 67
    assert ledger.ledger_schema == "cloud-capability-budget-ledger-v2"
    assert ledger.version == 67
    assert ledger.isolated_originals == originals
    assert (tmp_path / LOCK_NAME).exists()


def test_v2_record_updates_costs_and_reservation_and_keeps_isolation(tmp_path):
    originals = _pins_for(_sample_blobs())
    open_reservations = {
        "rid-a": {"stage": "appendix", "status": "open", "usd": 0.0122304},
        "rid-b": {"stage": "appendix", "status": "open", "usd": 0.5},
    }
    path = _write_document(
        tmp_path,
        _v2_document(originals, reservations=open_reservations),
    )
    ledger = BudgetLedger.for_stage("appendix", output_dir=tmp_path)
    ledger.record(
        actual_usd=0.01,
        usage={"input_tokens": 1, "output_tokens": 1},
        reservation_id="rid-a",
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["spent_by_stage"]["main"] == pytest.approx(0.0)
    assert payload["spent_by_stage"]["appendix"] == pytest.approx(0.07980472)
    assert payload["reservations"] == {"rid-b": open_reservations["rid-b"]}
    assert payload["version"] == 68
    assert payload["matrix_allowed"] is False
    assert payload["legacy_spend"]["usd"] is None
    _assert_isolation_fields(payload, originals)
    restored = BudgetLedger.for_stage("appendix", output_dir=tmp_path)
    assert restored.spent_usd == pytest.approx(0.07980472)
    assert restored.reserved_usd() == pytest.approx(0.5)
    assert restored.matrix_allowed is False
    assert restored.isolated_originals == originals


def test_v2_missing_usage_persist_keeps_isolation_fields(tmp_path):
    originals = _pins_for(_sample_blobs())
    reservation_id = "56b160d3c3dd4046a5ea10749e58f644"
    path = _write_document(tmp_path, _v2_document(originals, matrix_allowed=True))
    ledger = BudgetLedger.for_stage("appendix", output_dir=tmp_path)
    with pytest.raises(MissingUsage):
        ledger.record(actual_usd=0.01, usage=None, reservation_id=reservation_id)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["spent_by_stage"]["appendix"] == pytest.approx(0.06980472)
    assert reservation_id in payload["reservations"]
    assert payload["version"] == 68
    assert payload["matrix_allowed"] is False
    assert payload["legacy_spend"]["usd"] is None
    _assert_isolation_fields(payload, originals)


def test_v2_budget_exceeded_persist_keeps_isolation_fields(tmp_path):
    originals = _pins_for(_sample_blobs())
    reservation_id = "56b160d3c3dd4046a5ea10749e58f644"
    path = _write_document(tmp_path, _v2_document(originals, matrix_allowed=True))
    ledger = BudgetLedger.for_stage("appendix", output_dir=tmp_path)
    with pytest.raises(BudgetExceeded):
        ledger.record(
            actual_usd=10.0,
            usage={"input_tokens": 1, "output_tokens": 1},
            reservation_id=reservation_id,
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["spent_by_stage"]["appendix"] == pytest.approx(10.06980472)
    assert payload["reservations"] == {}
    assert payload["version"] == 68
    assert payload["matrix_allowed"] is False
    _assert_isolation_fields(payload, originals)


def test_v2_unpriceable_actual_cost_persist_keeps_isolation_fields(tmp_path):
    originals = _pins_for(_sample_blobs())
    reservation_id = "56b160d3c3dd4046a5ea10749e58f644"
    path = _write_document(tmp_path, _v2_document(originals, matrix_allowed=True))
    ledger = BudgetLedger.for_stage("appendix", output_dir=tmp_path)
    with pytest.raises(BudgetLedgerCorrupt):
        ledger.record(
            actual_usd=-0.01,
            usage={"input_tokens": 1, "output_tokens": 1},
            reservation_id=reservation_id,
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["spent_by_stage"]["appendix"] == pytest.approx(0.06980472)
    assert reservation_id in payload["reservations"]
    assert payload["version"] == 68
    assert payload["matrix_allowed"] is False
    _assert_isolation_fields(payload, originals)


def test_mixed_type_isolation_keys_raise_corrupt_without_rewrite(tmp_path):
    originals = _pins_for(_sample_blobs())
    document = _v2_document(originals)
    document["isolated_storage_keys"] = [_KEYS[0], 1, _KEYS[2]]
    path = _write_document(tmp_path, document)
    before = _fingerprint(path)
    with pytest.raises(BudgetLedgerCorrupt, match="isolated_storage_keys"):
        BudgetLedger.for_stage("appendix", output_dir=tmp_path)
    assert _fingerprint(path) == before


@pytest.mark.parametrize("field", ["stage", "status"])
@pytest.mark.parametrize("bad", [["appendix"], {"name": "appendix"}])
def test_v1_reservation_non_string_membership_is_corrupt_not_type_error(tmp_path, field, bad):
    """A list/dict stage or status must refuse as corrupt, never leak a TypeError."""

    document = _v1_document(
        reservations={"rid": {"stage": "appendix", "status": "open", "usd": 0.01}}
    )
    document["reservations"]["rid"][field] = bad
    path = _write_document(tmp_path, document)
    before = _fingerprint(path)
    with pytest.raises(BudgetLedgerCorrupt, match="reservation rid"):
        BudgetLedger.for_stage("appendix", output_dir=tmp_path)
    assert _fingerprint(path) == before


@pytest.mark.parametrize("field", ["stage", "status"])
@pytest.mark.parametrize("bad", [["appendix"], {"name": "appendix"}, None, 3, True])
def test_v2_reservation_non_string_membership_is_corrupt_without_rewrite(tmp_path, field, bad):
    document = _v2_document(
        _pins_for(_sample_blobs()),
        reservations={"rid": {"stage": "appendix", "status": "open", "usd": 0.01}},
    )
    document["reservations"]["rid"][field] = bad
    path = _write_document(tmp_path, document)
    before = _fingerprint(path)
    with pytest.raises(BudgetLedgerCorrupt, match="reservation rid"):
        BudgetLedger.for_stage("appendix", output_dir=tmp_path)
    assert _fingerprint(path) == before
