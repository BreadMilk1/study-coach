from __future__ import annotations

import json
import math
import os
import re
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping, TypeVar

try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None


class BudgetExceeded(Exception):
    """Next paid call would exceed the remaining stage or hard cap."""


class MissingUsage(Exception):
    """A paid call returned no usage; the full matrix must not start."""


class BudgetLedgerCorrupt(Exception):
    """Persisted ledger is unreadable; fail closed."""


LEDGER_SCHEMA = "cloud-capability-budget-ledger-v1"
LEDGER_SCHEMA_V2 = "cloud-capability-budget-ledger-v2"
LEDGER_NAME = "budget_ledger_v1.json"
LOCK_NAME = "budget_ledger_v1.lock"

_T = TypeVar("_T")
_STAGES = frozenset({"main", "appendix"})
_RESERVATION_STATUSES = frozenset({"open"})
_V2_ONLY_FIELDS = (
    "isolated_storage_keys",
    "isolated_originals",
    "source_snapshot_id",
    "settlements",
)
_ISOLATION_KEYS = (
    "s066a4d2f779e9c1e",
    "s19f35a2c8949e11b",
    "se68ebcb769e9f2ac",
)
_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")


def _token_is_priceable(value: Any) -> bool:
    if type(value) is not int or value < 0:
        return False
    try:
        amount = float(value)
    except OverflowError:
        return False
    return math.isfinite(amount)


def _nonneg_int(value: Any) -> bool:
    return _token_is_priceable(value)


def _usage_tokens_valid(usage: Any) -> bool:
    if not isinstance(usage, Mapping):
        return False
    inp = usage.get("input_tokens")
    out = usage.get("output_tokens")
    if not _nonneg_int(inp) or not _nonneg_int(out):
        return False
    if inp == 0 and out == 0:
        return False
    if "total_tokens" in usage and not _nonneg_int(usage.get("total_tokens")):
        return False
    if "cache_read_input_tokens" in usage and not _nonneg_int(usage.get("cache_read_input_tokens")):
        return False
    return True


def usage_has_unpriceable_tokens(usage: Any) -> bool:
    if not isinstance(usage, Mapping):
        return False
    for key in ("input_tokens", "output_tokens", "total_tokens", "cache_read_input_tokens"):
        if key not in usage:
            continue
        value = usage[key]
        if type(value) is int and value >= 0 and not _token_is_priceable(value):
            return True
    return False


def _require_finite_nonneg(value: Any, *, field: str) -> float:
    if type(value) is bool or type(value) not in (int, float):
        raise BudgetLedgerCorrupt(f"{field} is not a number")
    try:
        amount = float(value)
    except OverflowError as exc:
        raise BudgetLedgerCorrupt(f"{field} is not finite and non-negative") from exc
    if not math.isfinite(amount) or amount < 0:
        raise BudgetLedgerCorrupt(f"{field} is not finite and non-negative")
    return amount


def _load_prices() -> dict[str, Any]:
    path = Path(__file__).with_name("prices.json")
    return json.loads(path.read_text(encoding="utf-8"))


class BudgetLedger:
    def __init__(self, *, stage: str, prices: Mapping[str, Any], path: Path | None = None):
        if stage not in {"main", "appendix"}:
            raise ValueError(f"unknown stage: {stage}")
        self.stage = stage
        self.prices = prices
        self.path = Path(path) if path is not None else None
        self.spent_usd = 0.0
        self.spent_by_stage = {"main": 0.0, "appendix": 0.0}
        self.reservations: dict[str, dict[str, Any]] = {}
        self.matrix_allowed = True
        self.legacy_spend: dict[str, Any] = {"status": "unverifiable", "usd": None}
        self.version = 0
        self.ledger_schema = LEDGER_SCHEMA
        self.isolated_storage_keys: tuple[str, ...] = ()
        self.isolated_originals: dict[str, dict[str, str]] = {}
        self.source_snapshot_id: str | None = None
        self.settlements: dict[str, Any] = {}
        caps = prices["caps"]
        self.stage_cap = float(caps["main_usd"] if stage == "main" else caps["appendix_usd"])
        self.hard_cap = float(caps["hard_usd"])
        if self.path is not None:
            self._boot()

    @classmethod
    def for_stage(cls, stage: str, output_dir: Path | None = None) -> "BudgetLedger":
        path = None
        if output_dir is not None:
            dest = Path(output_dir)
            dest.mkdir(parents=True, exist_ok=True)
            path = dest / LEDGER_NAME
        return cls(stage=stage, prices=_load_prices(), path=path)

    def estimate_cost(
        self,
        *,
        model_key: str,
        input_tokens: int,
        max_output_tokens: int,
        cache_read_tokens: int = 0,
    ) -> float:
        rates = self.prices["per_million"][model_key]
        try:
            inp = _require_finite_nonneg(input_tokens, field="input_tokens")
            out = _require_finite_nonneg(max_output_tokens, field="max_output_tokens")
            cache = _require_finite_nonneg(cache_read_tokens, field="cache_read_tokens")
            cost = (
                inp / 1e6 * float(rates["input"])
                + out / 1e6 * float(rates["output"])
                + cache / 1e6 * float(rates.get("cache_read") or 0.0)
            )
        except (OverflowError, BudgetLedgerCorrupt) as exc:
            raise BudgetLedgerCorrupt("cost estimate is not finite") from exc
        if not math.isfinite(cost) or cost < 0:
            raise BudgetLedgerCorrupt("cost estimate is not finite")
        return cost

    def reserved_usd(self, stage: str | None = None) -> float:
        total = 0.0
        for item in self.reservations.values():
            if not isinstance(item, Mapping):
                continue
            if item.get("status") != "open":
                continue
            if stage is not None and item.get("stage") != stage:
                continue
            total += float(item.get("usd") or 0.0)
        return total

    def remaining_usd(self) -> float:
        stage_left = (
            self.stage_cap
            - float(self.spent_by_stage.get(self.stage) or 0.0)
            - self.reserved_usd(self.stage)
        )
        hard_left = self.hard_cap - self.spent_usd - self.reserved_usd()
        return min(stage_left, hard_left)

    def check_before_call(
        self,
        *,
        model_key: str,
        input_tokens: int,
        max_output_tokens: int,
    ) -> str:
        estimate = self.estimate_cost(
            model_key=model_key,
            input_tokens=input_tokens,
            max_output_tokens=max_output_tokens,
        )
        return self._with_lock(lambda: self._reserve(estimate))

    def record(
        self,
        *,
        actual_usd: float,
        usage: Mapping[str, Any] | None,
        reservation_id: str | None = None,
    ) -> None:
        def _commit() -> None:
            if not _usage_tokens_valid(usage):
                self.matrix_allowed = False
                raise MissingUsage("paid call missing usage")
            try:
                amount = _require_finite_nonneg(actual_usd, field="actual_usd")
            except BudgetLedgerCorrupt:
                self.matrix_allowed = False
                raise
            if reservation_id and reservation_id in self.reservations:
                del self.reservations[reservation_id]
            self.spent_by_stage[self.stage] = float(self.spent_by_stage.get(self.stage) or 0.0) + amount
            self.spent_usd = float(self.spent_by_stage["main"]) + float(self.spent_by_stage["appendix"])
            if (
                float(self.spent_by_stage[self.stage]) > self.stage_cap
                or self.spent_usd > self.hard_cap
            ):
                self.matrix_allowed = False
                raise BudgetExceeded(
                    f"{self.stage} spend {self.spent_by_stage[self.stage]:.4f} / "
                    f"combined {self.spent_usd:.4f} exceeds cap"
                )

        self._with_lock(_commit)

    def _reserve(self, estimate: float) -> str:
        if not self.matrix_allowed:
            raise MissingUsage("usage missing; matrix blocked")
        if estimate > self.remaining_usd():
            raise BudgetExceeded(
                f"{self.stage} remaining {self.remaining_usd():.4f} USD, estimate {estimate:.4f}"
            )
        rid = uuid.uuid4().hex
        self.reservations[rid] = {"stage": self.stage, "usd": float(estimate), "status": "open"}
        return rid

    def _empty_payload(self) -> dict[str, Any]:
        return {
            "schema_version": LEDGER_SCHEMA,
            "version": 0,
            "legacy_spend": {"status": "unverifiable", "usd": None},
            "spent_by_stage": {"main": 0.0, "appendix": 0.0},
            "reservations": {},
            "matrix_allowed": True,
        }

    def _load_unlocked(self) -> dict[str, Any]:
        if self.path is None or not self.path.exists():
            return self._empty_payload()
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BudgetLedgerCorrupt("unreadable budget ledger") from exc
        if not isinstance(payload, Mapping):
            raise BudgetLedgerCorrupt("budget ledger is not an object")
        schema = payload.get("schema_version")
        if schema == LEDGER_SCHEMA:
            for name in _V2_ONLY_FIELDS:
                if name in payload:
                    raise BudgetLedgerCorrupt(f"{name} is not valid on ledger v1")
        elif schema != LEDGER_SCHEMA_V2:
            raise BudgetLedgerCorrupt("unsupported budget ledger schema")
        return dict(payload)

    def _apply(self, payload: Mapping[str, Any]) -> None:
        if "schema_version" not in payload or "version" not in payload:
            raise BudgetLedgerCorrupt("ledger missing schema_version or version")
        if type(payload.get("version")) is not int or int(payload["version"]) < 0:
            raise BudgetLedgerCorrupt("ledger version is not a non-negative int")
        spent_by = payload.get("spent_by_stage")
        if not isinstance(spent_by, Mapping) or "main" not in spent_by or "appendix" not in spent_by:
            raise BudgetLedgerCorrupt("spent_by_stage missing")
        self.spent_by_stage = {
            "main": _require_finite_nonneg(spent_by["main"], field="spent_by_stage.main"),
            "appendix": _require_finite_nonneg(spent_by["appendix"], field="spent_by_stage.appendix"),
        }
        self.spent_usd = self.spent_by_stage["main"] + self.spent_by_stage["appendix"]
        raw_res = payload.get("reservations")
        if not isinstance(raw_res, Mapping):
            raise BudgetLedgerCorrupt("reservations missing")
        reservations: dict[str, dict[str, Any]] = {}
        for key, raw in raw_res.items():
            if not isinstance(raw, Mapping):
                raise BudgetLedgerCorrupt(f"reservation {key} is not an object")
            stage = raw.get("stage")
            status = raw.get("status")
            # JSON containers are unhashable: check the type before set membership so a
            # list/dict stage or status is refused as corrupt instead of leaking TypeError.
            if type(stage) is not str or stage not in _STAGES:
                raise BudgetLedgerCorrupt(f"reservation {key} has invalid stage")
            if type(status) is not str or status not in _RESERVATION_STATUSES:
                raise BudgetLedgerCorrupt(f"reservation {key} has invalid status")
            usd = _require_finite_nonneg(raw.get("usd"), field=f"reservation.{key}.usd")
            reservations[str(key)] = {"stage": stage, "status": status, "usd": usd}
        self.reservations = reservations
        allowed = payload.get("matrix_allowed")
        if type(allowed) is not bool:
            raise BudgetLedgerCorrupt("matrix_allowed is not bool")
        self.matrix_allowed = allowed
        legacy = payload.get("legacy_spend") or {"status": "unverifiable", "usd": None}
        self.legacy_spend = dict(legacy) if isinstance(legacy, Mapping) else {
            "status": "unverifiable",
            "usd": None,
        }
        self._apply_schema(payload)
        self.version = int(payload["version"])

    def _apply_schema(self, payload: Mapping[str, Any]) -> None:
        schema = payload.get("schema_version")
        if schema == LEDGER_SCHEMA:
            for name in _V2_ONLY_FIELDS:
                if name in payload:
                    raise BudgetLedgerCorrupt(f"{name} is not valid on ledger v1")
            self.ledger_schema = LEDGER_SCHEMA
            self.isolated_storage_keys = ()
            self.isolated_originals = {}
            self.source_snapshot_id = None
            self.settlements = {}
            return
        if schema != LEDGER_SCHEMA_V2:
            raise BudgetLedgerCorrupt("unsupported budget ledger schema")
        keys = payload.get("isolated_storage_keys")
        if (
            type(keys) is not list
            or len(keys) != len(_ISOLATION_KEYS)
            or any(type(item) is not str for item in keys)
            or sorted(keys) != list(_ISOLATION_KEYS)
        ):
            raise BudgetLedgerCorrupt("isolated_storage_keys is not the three HyDE originals")
        originals = payload.get("isolated_originals")
        if type(originals) is not dict or set(originals) != set(_ISOLATION_KEYS):
            raise BudgetLedgerCorrupt("isolated_originals does not match isolated_storage_keys")
        normalized: dict[str, dict[str, str]] = {}
        for key in _ISOLATION_KEYS:
            item = originals[key]
            if type(item) is not dict or set(item) != {"candidate_sha256", "sqlite_sha256"}:
                raise BudgetLedgerCorrupt("candidate_sha256 or sqlite_sha256 is missing")
            candidate_sha = item["candidate_sha256"]
            sqlite_sha = item["sqlite_sha256"]
            if type(candidate_sha) is not str or _SHA256_HEX.fullmatch(candidate_sha) is None:
                raise BudgetLedgerCorrupt("candidate_sha256 is not lowercase sha256")
            if type(sqlite_sha) is not str or _SHA256_HEX.fullmatch(sqlite_sha) is None:
                raise BudgetLedgerCorrupt("sqlite_sha256 is not lowercase sha256")
            normalized[key] = {
                "candidate_sha256": candidate_sha,
                "sqlite_sha256": sqlite_sha,
            }
        if "source_snapshot_id" not in payload:
            raise BudgetLedgerCorrupt("source_snapshot_id is missing")
        snapshot_id = payload.get("source_snapshot_id")
        if type(snapshot_id) is not str or _SHA256_HEX.fullmatch(snapshot_id) is None:
            raise BudgetLedgerCorrupt("source_snapshot_id is not lowercase sha256")
        if "settlements" not in payload:
            raise BudgetLedgerCorrupt("settlements is missing")
        settlements = payload.get("settlements")
        if type(settlements) is not dict or len(settlements) != 0:
            raise BudgetLedgerCorrupt("settlements must be empty")
        self.ledger_schema = LEDGER_SCHEMA_V2
        self.isolated_storage_keys = _ISOLATION_KEYS
        self.isolated_originals = normalized
        self.source_snapshot_id = snapshot_id
        self.settlements = {}

    def _dump(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.ledger_schema,
            "version": self.version,
            "legacy_spend": dict(self.legacy_spend),
            "spent_by_stage": dict(self.spent_by_stage),
            "reservations": dict(self.reservations),
            "matrix_allowed": self.matrix_allowed,
        }
        if self.ledger_schema == LEDGER_SCHEMA_V2:
            payload["isolated_storage_keys"] = list(self.isolated_storage_keys)
            payload["isolated_originals"] = {
                key: dict(self.isolated_originals[key]) for key in self.isolated_storage_keys
            }
            payload["source_snapshot_id"] = self.source_snapshot_id
            payload["settlements"] = {}
        return payload

    def _persist_unlocked(self) -> None:
        if self.path is None:
            return
        self.version += 1
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(self._dump(), indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        os.replace(tmp, self.path)

    def _boot(self) -> None:
        def _init() -> None:
            if self.path is not None and not self.path.exists():
                self._persist_unlocked()
            else:
                self._apply(self._load_unlocked())

        self._locked(_init, reload=False, persist=False)

    def _locked(self, fn: Callable[[], _T], *, reload: bool = True, persist: bool = True) -> _T:
        if self.path is None:
            return fn()
        lock_path = self.path.with_name(LOCK_NAME)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with open(lock_path, "a+", encoding="utf-8") as lock:
            if fcntl is not None:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                if reload:
                    self._apply(self._load_unlocked())
                try:
                    result = fn()
                except (MissingUsage, BudgetExceeded, BudgetLedgerCorrupt):
                    if persist:
                        self._persist_unlocked()
                    raise
                if persist:
                    self._persist_unlocked()
                return result
            finally:
                if fcntl is not None:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _with_lock(self, fn: Callable[[], _T]) -> _T:
        return self._locked(fn)
