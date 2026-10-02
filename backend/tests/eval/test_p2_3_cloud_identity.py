import pytest

from app.eval.p2_3_cloud_capability.identity import (
    ProviderIdentity,
    cloud_run_id,
    recover_run_idx,
    storage_key,
)


def _identity(**overrides):
    fields = {"provider": "minimax", "protocol": "anthropic", "model": "MiniMax-M3"}
    fields.update(overrides)
    return ProviderIdentity(**fields)


def test_storage_key_rejects_slash_and_pipe():
    ident = ProviderIdentity(provider="minimax", protocol="anthropic", model="org/MiniMax-M3")
    key = storage_key(ident, mode="agent_loop", thinking_profile="disabled", query_id="quiz_hyde", run_idx=0)
    assert "/" not in key
    assert "|" not in key
    assert "\\" not in key
    assert key.isalnum() or set(key) <= set("abcdefghijklmnopqrstuvwxyz0123456789-_")


def test_cloud_run_id_includes_provider_protocol_model():
    a = ProviderIdentity(provider="minimax", protocol="anthropic", model="MiniMax-M3")
    b = ProviderIdentity(provider="other", protocol="anthropic", model="MiniMax-M3")
    rid_a = cloud_run_id(a, "deterministic", "disabled", "quiz_hyde", 0, 0)
    rid_b = cloud_run_id(b, "deterministic", "disabled", "quiz_hyde", 0, 0)
    assert rid_a != rid_b
    assert len(rid_a) == 16


def test_thinking_profile_changes_run_id():
    ident = ProviderIdentity(provider="minimax", protocol="anthropic", model="MiniMax-M3")
    off = cloud_run_id(ident, "deterministic", "disabled", "quiz_hyde", 0, 0)
    on = cloud_run_id(ident, "deterministic", "fixed-budget-1024", "quiz_hyde", 0, 0)
    assert off != on


def test_recover_run_idx_from_hashed_run_id():
    from app.eval.p2_3_cloud_capability.identity import recover_run_idx

    ident = ProviderIdentity(provider="minimax", protocol="anthropic", model="MiniMax-M3")
    for run_idx in (0, 1, 2):
        rid = cloud_run_id(ident, "agent_loop", "disabled", "quiz_hyde", 0, run_idx)
        got = recover_run_idx({
            "run_id": rid,
            "mode": "agent_loop",
            "thinking_profile": "disabled",
            "query_id": "quiz_hyde",
            "turn_idx": 0,
        })
        assert got == run_idx


def test_recover_run_idx_from_session_storage_key():
    ident = ProviderIdentity(provider="minimax", protocol="anthropic", model="MiniMax-M3")
    key = storage_key(ident, mode="deterministic", thinking_profile="disabled", query_id="quiz_rrf", run_idx=2)
    got = recover_run_idx({
        "session_storage_key": key,
        "mode": "deterministic",
        "thinking_profile": "disabled",
        "query_id": "quiz_rrf",
        "turn_idx": 0,
    })
    assert got == 2


def test_ambiguous_separator_fields_are_rejected_by_both_builders():
    """('a|b','c') and ('a','b|c') must never hash to the same identity."""
    left = _identity(provider="a|b", protocol="c")
    right = _identity(provider="a", protocol="b|c")
    for ident in (left, right):
        with pytest.raises(ValueError):
            cloud_run_id(ident, "deterministic", "disabled", "quiz_hyde", 0, 0)
        with pytest.raises(ValueError):
            storage_key(ident, mode="deterministic", thinking_profile="disabled", query_id="quiz_hyde", run_idx=0)


def test_separator_is_rejected_in_every_joined_field_without_echoing_input():
    marker = "leak|marker"
    calls = [
        lambda: cloud_run_id(_identity(provider=marker), "det", "disabled", "quiz_hyde", 0, 0),
        lambda: cloud_run_id(_identity(protocol=marker), "det", "disabled", "quiz_hyde", 0, 0),
        lambda: cloud_run_id(_identity(model=marker), "det", "disabled", "quiz_hyde", 0, 0),
        lambda: cloud_run_id(_identity(), marker, "disabled", "quiz_hyde", 0, 0),
        lambda: cloud_run_id(_identity(), "det", marker, "quiz_hyde", 0, 0),
        lambda: cloud_run_id(_identity(), "det", "disabled", marker, 0, 0),
        lambda: cloud_run_id(_identity(), "det", "disabled", "quiz_hyde", marker, 0),
        lambda: cloud_run_id(_identity(), "det", "disabled", "quiz_hyde", 0, marker),
        lambda: storage_key(_identity(provider=marker), mode="det", thinking_profile="disabled", query_id="quiz_hyde", run_idx=0),
        lambda: storage_key(_identity(protocol=marker), mode="det", thinking_profile="disabled", query_id="quiz_hyde", run_idx=0),
        lambda: storage_key(_identity(model=marker), mode="det", thinking_profile="disabled", query_id="quiz_hyde", run_idx=0),
        lambda: storage_key(_identity(), mode=marker, thinking_profile="disabled", query_id="quiz_hyde", run_idx=0),
        lambda: storage_key(_identity(), mode="det", thinking_profile=marker, query_id="quiz_hyde", run_idx=0),
        lambda: storage_key(_identity(), mode="det", thinking_profile="disabled", query_id=marker, run_idx=0),
        lambda: storage_key(_identity(), mode="det", thinking_profile="disabled", query_id="quiz_hyde", run_idx=marker),
    ]
    for call in calls:
        with pytest.raises(ValueError) as raised:
            call()
        assert marker not in str(raised.value)
        assert "|" not in str(raised.value)


def test_identity_golden_values_are_stable():
    """Hardcoded compatibility goldens. Never regenerate these from the functions under test."""
    cases = [
        (
            _identity(),
            "deterministic",
            "disabled",
            "quiz_hyde",
            0,
            0,
            "1a2304bfe222f6ff",
            "sa4aa535a490f62c7",
        ),
        (
            _identity(),
            "agent_loop",
            "fixed-budget-1024",
            "quiz_rrf",
            1,
            2,
            "d74a64025261743a",
            "sa8db100ed6d9bef8",
        ),
        (
            _identity(model="org/MiniMax-M3"),
            "agent_loop",
            "disabled",
            "题目_一",
            0,
            1,
            "80c3a5e42e347a96",
            "scfddbe3a0cd08eb6",
        ),
    ]
    for ident, mode, profile, query_id, turn_idx, run_idx, expected_run, expected_key in cases:
        assert cloud_run_id(ident, mode, profile, query_id, turn_idx, run_idx) == expected_run
        assert storage_key(ident, mode=mode, thinking_profile=profile, query_id=query_id, run_idx=run_idx) == expected_key
        assert len(expected_run) == 16
        assert expected_key == "s" + expected_key[1:]
        assert len(expected_key) == 17


def test_recover_run_idx_falls_back_to_session_key_when_turn_idx_is_unusable():
    ident = _identity()
    key = storage_key(ident, mode="deterministic", thinking_profile="disabled", query_id="quiz_rrf", run_idx=2)
    for bad_turn_idx in ("not-an-int", float("inf"), object(), [0]):
        assert recover_run_idx({
            "session_storage_key": key,
            "mode": "deterministic",
            "thinking_profile": "disabled",
            "query_id": "quiz_rrf",
            "turn_idx": bad_turn_idx,
        }) == 2, bad_turn_idx


def test_recover_run_idx_skips_run_id_path_with_bad_turn_idx_then_uses_session_key():
    ident = _identity()
    run_id = cloud_run_id(ident, "agent_loop", "disabled", "quiz_hyde", 1, 2)
    key = storage_key(ident, mode="agent_loop", thinking_profile="disabled", query_id="quiz_hyde", run_idx=1)
    assert recover_run_idx({
        "run_id": run_id,
        "session_storage_key": key,
        "mode": "agent_loop",
        "thinking_profile": "disabled",
        "query_id": "quiz_hyde",
        "turn_idx": "bad",
    }) == 1


def test_recover_run_idx_returns_none_when_only_the_run_id_path_is_usable():
    assert recover_run_idx({
        "run_id": "whatever",
        "mode": "agent_loop",
        "thinking_profile": "disabled",
        "query_id": "quiz_hyde",
        "turn_idx": object(),
    }) is None


def _turn_one_conflict(turn_idx_value):
    """run_id(turn=1, run=2) competes with session_storage_key(run=1) in one context."""
    ident = _identity()
    return {
        "run_id": cloud_run_id(ident, "agent_loop", "disabled", "quiz_hyde", 1, 2),
        "session_storage_key": storage_key(
            ident, mode="agent_loop", thinking_profile="disabled", query_id="quiz_hyde", run_idx=1
        ),
        "mode": "agent_loop",
        "thinking_profile": "disabled",
        "query_id": "quiz_hyde",
        "turn_idx": turn_idx_value,
    }


def _turn_zero_conflict(turn_idx_value):
    """run_id(turn=0, run=2) competes with session_storage_key(run=1) in one context."""
    ident = _identity()
    return {
        "run_id": cloud_run_id(ident, "agent_loop", "disabled", "quiz_hyde", 0, 2),
        "session_storage_key": storage_key(
            ident, mode="agent_loop", thinking_profile="disabled", query_id="quiz_hyde", run_idx=1
        ),
        "mode": "agent_loop",
        "thinking_profile": "disabled",
        "query_id": "quiz_hyde",
        "turn_idx": turn_idx_value,
    }


def test_recover_run_idx_rejects_non_integer_turn_idx_type():
    for invalid in (1.5, 1.0, True, "1"):
        assert recover_run_idx(_turn_one_conflict(invalid)) == 1, invalid


def test_recover_run_idx_ignores_falsy_invalid_turn_idx_for_turn_zero_records():
    for invalid in (False, None, "", [], {}):
        assert recover_run_idx(_turn_zero_conflict(invalid)) == 1, invalid


def test_recover_run_idx_keeps_exact_integer_turn_idx_priority():
    assert recover_run_idx(_turn_one_conflict(1)) == 2


def test_recover_run_idx_returns_none_for_non_integer_turn_idx_without_session_key():
    assert recover_run_idx({
        "run_id": cloud_run_id(_identity(), "agent_loop", "disabled", "quiz_hyde", 1, 2),
        "mode": "agent_loop",
        "thinking_profile": "disabled",
        "query_id": "quiz_hyde",
        "turn_idx": 2.5,
    }) is None


def test_recover_run_idx_keeps_run_id_priority_and_existing_index_behavior():
    ident = _identity()
    run_id = cloud_run_id(ident, "agent_loop", "disabled", "quiz_hyde", 1, 2)
    key = storage_key(ident, mode="agent_loop", thinking_profile="disabled", query_id="quiz_hyde", run_idx=0)
    assert recover_run_idx({
        "run_id": run_id,
        "session_storage_key": key,
        "mode": "agent_loop",
        "thinking_profile": "disabled",
        "query_id": "quiz_hyde",
        "turn_idx": 1,
    }) == 2
    assert recover_run_idx({"run_idx": 0, "session_storage_key": key}) == 0
    assert recover_run_idx({
        "run_id": cloud_run_id(ident, "deterministic", "disabled", "quiz_hyde", 0, 1),
        "mode": "deterministic",
        "thinking_profile": "disabled",
        "query_id": "quiz_hyde",
    }) == 1
