from app.eval.p2_3_cloud_capability.identity import (
    ProviderIdentity,
    cloud_run_id,
    storage_key,
)


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
    from app.eval.p2_3_cloud_capability.identity import recover_run_idx

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
