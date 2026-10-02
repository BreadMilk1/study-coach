import json

import pytest

from app.eval.p2_3_cloud_capability.artifacts import (
    CandidateStore,
    ScoreStore,
    SessionIncomplete,
    encode_scorer_id,
    new_candidate,
)


def test_session_commit_is_atomic(tmp_path):
    store = CandidateStore(tmp_path)
    store.begin_session("s1")
    store.add_turn("s1", new_candidate(run_id="t0", turn_idx=0, final_text="quiz"))
    assert store.has_complete_session("s1") is False
    with pytest.raises(SessionIncomplete):
        store.commit_session("s1", expected_turns=2)
    assert store.load_session("s1") is None


def test_complete_session_is_immutable(tmp_path):
    store = CandidateStore(tmp_path)
    store.begin_session("s1")
    store.add_turn("s1", new_candidate(run_id="t0", turn_idx=0, final_text="quiz"))
    store.add_turn("s1", new_candidate(run_id="t1", turn_idx=1, final_text="grade"))
    store.commit_session("s1", expected_turns=2)
    with pytest.raises(ValueError, match="immutable"):
        store.add_turn("s1", new_candidate(run_id="t2", turn_idx=2, final_text="nope"))


def test_candidate_omits_thinking_text_and_secrets():
    c = new_candidate(
        run_id="t0",
        turn_idx=0,
        final_text="visible",
        thinking_text="SECRET_CHAIN",
        raw_response_bytes=b'{"content":"SECRET_CHAIN"}',
        api_key="sk-secret",
    )
    dumped = c.to_dict()
    blob = str(dumped)
    assert "SECRET_CHAIN" not in blob
    assert "sk-secret" not in blob
    assert dumped["thinking"]["present"] is True
    assert dumped["thinking"]["char_length"] == len("SECRET_CHAIN")
    assert dumped["raw_response_sha256"]
    assert dumped["final_text"] == "visible"
    assert dumped["schema_version"] == "cloud-capability-candidate-v1"
    assert dumped["usage"] == "unavailable" or isinstance(dumped["usage"], dict)
    assert "wall_time_s" in dumped


def test_candidate_records_wall_time():
    dumped = new_candidate(
        run_id="t0",
        turn_idx=0,
        final_text="quiz",
        wall_time_s=1.25,
    ).to_dict()
    assert dumped["wall_time_s"] == pytest.approx(1.25)
    assert "run_idx" in dumped


def test_candidate_records_run_idx():
    dumped = new_candidate(
        run_id="t0",
        turn_idx=0,
        final_text="quiz",
        run_idx=2,
    ).to_dict()
    assert dumped["run_idx"] == 2


def test_transport_failure_is_not_model_failure():
    c = new_candidate(
        run_id="t0",
        turn_idx=0,
        final_text="",
        failure_class="transport",
        finish_status="failed",
    )
    dumped = c.to_dict()
    assert dumped["failure_class"] == "transport"
    assert dumped["finish_status"] == "failed"


def test_error_type_is_class_name_only():
    c = new_candidate(
        run_id="t0",
        turn_idx=0,
        final_text="",
        failure_class="model",
        finish_status="failed",
        error_type="RuntimeError",
        error_message="Called get_stream_writer() outside of a stream.",
    )
    dumped = c.to_dict()
    assert dumped["error_type"] == "RuntimeError"
    blob = str(dumped)
    assert "get_stream_writer" not in blob
    assert "Called" not in blob


def test_scorer_id_encoding_is_stable_and_path_safe():
    scorer_id = "quiz-artifact-v1/qwen2.5:7b"
    encoded = encode_scorer_id(scorer_id)
    assert "/" not in encoded
    assert ":" not in encoded
    assert encode_scorer_id(scorer_id) == encoded
    lowered = encoded.lower()
    assert "%2f" in lowered
    assert "%3a" in lowered


def test_later_failure_does_not_replace_selected_success(tmp_path):
    store = ScoreStore(tmp_path)
    run_id = "abc123"
    sid = "quiz-artifact-v1/deepseek-v4-pro"
    (tmp_path / f"{run_id}.json").write_text(
        json.dumps([{"scorer_id": sid, "status": "success", "output": {"score": 0.84}}]),
        encoding="utf-8",
    )
    store.append(run_id, {"scorer_id": sid, "status": "failed", "error_code": "parse"})
    selected = store.selected(run_id, sid)
    assert selected is not None
    assert selected["status"] == "success"
    assert selected["output"]["score"] == 0.84
    assert len(store.history(run_id, sid)) == 2


def test_legacy_path_rejects_traversal_and_stays_inside_store(tmp_path):
    store = ScoreStore(tmp_path)
    with pytest.raises(ValueError, match="path"):
        store.legacy_path("..")
    with pytest.raises(ValueError, match="path"):
        store.load_legacy("..")
    for run_id in ("../evil", "/tmp/evil", "foo/bar", "foo\\bar"):
        path = store.legacy_path(run_id)
        resolved = path.resolve()
        assert resolved.is_relative_to(tmp_path.resolve())
        assert resolved.parent == tmp_path.resolve()


def test_append_fail_closed_when_fcntl_missing(tmp_path, monkeypatch):
    import app.eval.p2_3_cloud_capability.artifacts as artifacts

    monkeypatch.setattr(artifacts, "fcntl", None)
    store = ScoreStore(tmp_path)
    with pytest.raises(RuntimeError, match="lock"):
        store.append("t0", {"scorer_id": "quiz-artifact-v1/qwen2.5:7b", "status": "success"})
    executions = tmp_path / "executions"
    written = list(executions.rglob("*.json")) if executions.exists() else []
    assert written == []


def test_append_rejects_dotdot_run_id_and_encodes_slash_scorer(tmp_path):
    store = ScoreStore(tmp_path)
    with pytest.raises(ValueError, match="path"):
        store.append("..", {"scorer_id": "quiz-artifact-v1/qwen2.5:7b", "status": "success"})
    dest = store.append("t0", {"scorer_id": "quiz-artifact-v1/qwen2.5:7b", "status": "success"})
    assert dest.is_file()
    assert ".." not in dest.parts
    assert dest.resolve().is_relative_to(tmp_path.resolve())
    assert "/" not in dest.parent.name
    assert "%2F" in dest.parent.name.upper() or "%2f" in dest.parent.name


def _score_append_worker(root: str, run_id: str, sid: str, count: int, started, gate, errors, calls) -> None:
    try:
        from pathlib import Path

        from app.eval.p2_3_cloud_capability.artifacts import ScoreStore

        store = ScoreStore(Path(root))
        started.set()
        gate.wait(timeout=10)
        for idx in range(count):
            store.append(
                run_id,
                {"scorer_id": sid, "status": "failed", "error_code": f"n{idx}"},
            )
            with calls.get_lock():
                calls.value += 1
    except Exception as exc:  # pragma: no cover - surfaced by parent
        errors.put(f"{type(exc).__name__}: {exc}")


def test_concurrent_appends_do_not_lose_executions(tmp_path):
    import multiprocessing as mp

    sid = "quiz-artifact-v1/deepseek-v4-pro"
    ctx = mp.get_context("spawn")
    started_a = ctx.Event()
    started_b = ctx.Event()
    gate = ctx.Event()
    errors = ctx.Queue()
    calls = ctx.Value("i", 0)
    workers = [
        ctx.Process(
            target=_score_append_worker,
            args=(str(tmp_path), "run-a", sid, 8, started_a, gate, errors, calls),
        ),
        ctx.Process(
            target=_score_append_worker,
            args=(str(tmp_path), "run-a", sid, 8, started_b, gate, errors, calls),
        ),
    ]
    for proc in workers:
        proc.start()
    assert started_a.wait(timeout=10)
    assert started_b.wait(timeout=10)
    gate.set()
    for proc in workers:
        proc.join(timeout=15)
        assert proc.exitcode == 0
    assert errors.empty()
    store = ScoreStore(tmp_path)
    hist = store.history("run-a", sid)
    files = [
        path
        for path in (tmp_path / "executions").rglob("*.json")
        if path.stem.isdigit()
    ]
    assert calls.value == 16
    assert len(hist) == calls.value
    assert len(files) == calls.value


def test_structured_artifact_failure_is_not_retried(tmp_path):
    store = ScoreStore(tmp_path)
    run_id = "abc123"
    sid = "quiz-artifact-v1/deepseek-v4-pro"
    (tmp_path / f"{run_id}.json").write_text(
        json.dumps(
            [{"scorer_id": sid, "status": "failed", "error_code": "structured_artifact_failure"}]
        ),
        encoding="utf-8",
    )
    assert store.needs_retry(run_id, sid) is False


def test_candidate_store_loads_legacy_payload_without_new_fields(tmp_path):
    payload = [
        {
            "schema_version": "cloud-capability-candidate-v1",
            "run_id": "t0",
            "turn_idx": 0,
            "final_text": "quiz",
        }
    ]
    (tmp_path / "s-old.json").write_text(json.dumps(payload), encoding="utf-8")
    store = CandidateStore(tmp_path)
    rows = store.load_session("s-old")
    assert rows is not None
    assert rows[0]["final_text"] == "quiz"
    assert "response_count" not in rows[0]
    assert "langchain_response_sha256" not in rows[0]


def test_incomplete_partial_is_discarded_on_resume(tmp_path):
    store = CandidateStore(tmp_path)
    store.begin_session("s1")
    store.add_turn("s1", new_candidate(run_id="t0", turn_idx=0, final_text="quiz"))
    assert list(tmp_path.glob("*.partial.json"))
    store.discard_incomplete()
    assert store.load_session("s1") is None
    assert not list(tmp_path.glob("*.partial.json"))


_UNSAFE_SESSION_KEYS = [
    None,
    123,
    b"s1",
    "",
    "s\x001",
    "../escape",
    "sub/escape",
    "a\\b",
    "sub\\escape",
    "/absolute/path",
    ".",
    "..",
    "s.partial",
    "s.failed",
    "S.PARTIAL",
    "s.Failed",
]

_SAFE_SESSION_KEYS = ["s1", "s-old", "s-agent", "v1.2", "s.notes", "会话-中文"]


def _store_snapshot(tmp_path):
    files = {}
    for path in sorted(tmp_path.rglob("*")):
        if path.is_file():
            files[str(path.relative_to(tmp_path))] = (
                path.read_bytes(),
                path.stat().st_mtime_ns,
            )
    return files


def test_unsafe_keys_rejected_by_every_public_entry(tmp_path):
    store = CandidateStore(tmp_path / "store")
    candidate = new_candidate(run_id="t0", turn_idx=0, final_text="quiz")
    for key in _UNSAFE_SESSION_KEYS:
        with pytest.raises(ValueError, match="unsafe session storage key"):
            store.has_complete_session(key)
        with pytest.raises(ValueError, match="unsafe session storage key"):
            store.load_session(key)
        with pytest.raises(ValueError, match="unsafe session storage key"):
            store.begin_session(key)
        with pytest.raises(ValueError, match="unsafe session storage key"):
            store.add_turn(key, candidate)
        with pytest.raises(ValueError, match="unsafe session storage key"):
            store.commit_session(key, expected_turns=0)
        with pytest.raises(ValueError, match="unsafe session storage key"):
            store.discard_session(key)
        with pytest.raises(ValueError, match="unsafe session storage key"):
            store.record_failure(key, candidate)
        with pytest.raises(ValueError, match="unsafe session storage key"):
            store.load_failure(key)
    assert list((tmp_path / "store").iterdir()) == []


def test_rejection_message_does_not_echo_the_key(tmp_path):
    store = CandidateStore(tmp_path / "store")
    candidate = new_candidate(run_id="t0", turn_idx=0, final_text="quiz")
    canary = "../canary-secret-123"
    calls = [
        lambda: store.has_complete_session(canary),
        lambda: store.load_session(canary),
        lambda: store.begin_session(canary),
        lambda: store.add_turn(canary, candidate),
        lambda: store.commit_session(canary, expected_turns=0),
        lambda: store.discard_session(canary),
        lambda: store.record_failure(canary, candidate),
        lambda: store.load_failure(canary),
    ]
    for call in calls:
        with pytest.raises(ValueError) as excinfo:
            call()
        assert "canary-secret-123" not in str(excinfo.value)


def test_rejected_writes_leave_files_and_open_untouched(tmp_path):
    root = tmp_path / "store"
    store = CandidateStore(root)
    store.begin_session("s1")
    store.add_turn("s1", new_candidate(run_id="t0", turn_idx=0, final_text="quiz"))
    before = _store_snapshot(tmp_path)
    candidate = new_candidate(run_id="t1", turn_idx=9, final_text="nope")
    with pytest.raises(ValueError, match="unsafe session storage key"):
        store.begin_session("../escape")
    with pytest.raises(ValueError, match="unsafe session storage key"):
        store.add_turn("../escape", candidate)
    with pytest.raises(ValueError, match="unsafe session storage key"):
        store.commit_session("../escape", expected_turns=0)
    with pytest.raises(ValueError, match="unsafe session storage key"):
        store.discard_session("../escape")
    with pytest.raises(ValueError, match="unsafe session storage key"):
        store.record_failure("../escape", candidate)
    assert _store_snapshot(tmp_path) == before
    assert sorted(store._open) == ["s1"]
    assert len(store._open["s1"]) == 1
    assert not (tmp_path / "escape.json").exists()
    assert not (tmp_path / "escape.partial.json").exists()
    assert not (tmp_path / "escape.failed.json").exists()


def test_invalid_reads_do_not_access_escape_target(tmp_path):
    root = tmp_path / "store"
    store = CandidateStore(root)
    outside_rows = [{"run_id": "t0", "final_text": "OUTSIDE"}]
    (tmp_path / "target.json").write_text(json.dumps(outside_rows), encoding="utf-8")
    (tmp_path / "target.failed.json").write_text(
        json.dumps(outside_rows[0]), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="unsafe session storage key"):
        store.has_complete_session("../target")
    with pytest.raises(ValueError, match="unsafe session storage key"):
        store.load_session("../target")
    with pytest.raises(ValueError, match="unsafe session storage key"):
        store.load_failure("../target")
    assert store.load_session("missing") is None
    assert store.load_failure("missing") is None


def test_commit_rejects_invalid_key_before_turn_count_check(tmp_path):
    root = tmp_path / "store"
    store = CandidateStore(root)
    # expected_turns=0 trivially matches an unstarted session's count check,
    # so the key rule has to fire first.
    with pytest.raises(ValueError, match="unsafe session storage key"):
        store.commit_session("../escape", expected_turns=0)
    assert list(root.iterdir()) == []
    assert not (tmp_path / "escape.json").exists()


def test_reserved_suffix_keys_never_collide_with_internal_files(tmp_path):
    root = tmp_path / "store"
    store = CandidateStore(root)
    store.begin_session("s1")
    store.add_turn("s1", new_candidate(run_id="t0", turn_idx=0, final_text="quiz"))
    assert (root / "s1.partial.json").exists()
    assert store.has_complete_session("s1") is False
    with pytest.raises(ValueError, match="unsafe session storage key"):
        store.has_complete_session("s1.partial")
    with pytest.raises(ValueError, match="unsafe session storage key"):
        store.load_session("s1.partial")
    with pytest.raises(ValueError, match="unsafe session storage key"):
        store.commit_session("s1.partial", expected_turns=1)
    candidate = new_candidate(
        run_id="t1",
        turn_idx=0,
        final_text="boom",
        failure_class="model",
        finish_status="failed",
    )
    store.record_failure("s9", candidate)
    assert (root / "s9.failed.json").exists()
    assert store.load_failure("s9")["final_text"] == "boom"
    with pytest.raises(ValueError, match="unsafe session storage key"):
        store.has_complete_session("s9.failed")
    with pytest.raises(ValueError, match="unsafe session storage key"):
        store.load_session("s9.failed")


@pytest.mark.parametrize("key", _SAFE_SESSION_KEYS)
def test_safe_keys_keep_working_with_unchanged_filenames(key, tmp_path):
    root = tmp_path / "store"
    store = CandidateStore(root)
    store.begin_session(key)
    store.add_turn(key, new_candidate(run_id="t0", turn_idx=0, final_text="quiz"))
    store.add_turn(key, new_candidate(run_id="t1", turn_idx=1, final_text="grade"))
    assert (root / f"{key}.partial.json").exists()
    store.commit_session(key, expected_turns=2)
    assert store.has_complete_session(key) is True
    assert not (root / f"{key}.partial.json").exists()
    rows = store.load_session(key)
    assert rows is not None
    assert [row["final_text"] for row in rows] == ["quiz", "grade"]
    assert rows[0]["session_storage_key"] == key
    candidate = new_candidate(
        run_id="t2",
        turn_idx=2,
        final_text="boom",
        failure_class="transport",
        finish_status="failed",
    )
    store.record_failure(key, candidate)
    assert (root / f"{key}.failed.json").exists()
    assert store.load_failure(key)["final_text"] == "boom"


def test_recommit_rejections_leave_store_untouched(tmp_path):
    root = tmp_path / "store"
    store = CandidateStore(root)
    store.begin_session("s1")
    store.add_turn("s1", new_candidate(run_id="t0", turn_idx=0, final_text="quiz"))
    store.commit_session("s1", expected_turns=1)
    complete_rows = store.load_session("s1")
    store.begin_session("s2")
    store.add_turn("s2", new_candidate(run_id="t3", turn_idx=0, final_text="draft"))
    before = _store_snapshot(tmp_path)
    assert {key: len(value) for key, value in store._open.items()} == {"s2": 1}
    with pytest.raises(ValueError, match="immutable"):
        store.commit_session("s1", expected_turns=0)
    with pytest.raises(ValueError, match="immutable"):
        store.commit_session("s1", expected_turns=1)
    assert _store_snapshot(tmp_path) == before
    assert store.load_session("s1") == complete_rows
    assert {key: len(value) for key, value in store._open.items()} == {"s2": 1}
    assert (root / "s2.partial.json").exists()
    assert not list(root.glob("*.tmp"))


def test_fresh_instance_recommit_never_overwrites_complete(tmp_path):
    root = tmp_path / "store"
    first = CandidateStore(root)
    first.begin_session("s1")
    first.add_turn("s1", new_candidate(run_id="t0", turn_idx=0, final_text="quiz"))
    first.commit_session("s1", expected_turns=1)
    complete_rows = first.load_session("s1")
    before = _store_snapshot(tmp_path)
    second = CandidateStore(root)
    with pytest.raises(ValueError, match="immutable"):
        second.commit_session("s1", expected_turns=0)
    with pytest.raises(ValueError, match="session not started"):
        second.commit_session("fresh-key", expected_turns=0)
    assert _store_snapshot(tmp_path) == before
    assert second.load_session("s1") == complete_rows
    assert second._open == {}
    assert not list(root.glob("*.tmp"))


def test_commit_unstarted_key_never_creates_complete_file(tmp_path):
    root = tmp_path / "store"
    store = CandidateStore(root)
    before = _store_snapshot(tmp_path)
    with pytest.raises(ValueError, match="session not started"):
        store.commit_session("s-never-begun", expected_turns=0)
    assert _store_snapshot(tmp_path) == before
    assert not (root / "s-never-begun.json").exists()
    assert store._open == {}


def test_explicit_zero_turn_commit_still_succeeds(tmp_path):
    root = tmp_path / "store"
    store = CandidateStore(root)
    store.begin_session("s-empty")
    store.commit_session("s-empty", expected_turns=0)
    assert store.has_complete_session("s-empty") is True
    assert store.load_session("s-empty") == []
    assert not (root / "s-empty.partial.json").exists()
    assert "s-empty" not in store._open


def test_count_mismatch_then_completed_commit_succeeds(tmp_path):
    root = tmp_path / "store"
    store = CandidateStore(root)
    store.begin_session("s1")
    store.add_turn("s1", new_candidate(run_id="t0", turn_idx=0, final_text="quiz"))
    with pytest.raises(SessionIncomplete):
        store.commit_session("s1", expected_turns=2)
    assert store.has_complete_session("s1") is False
    assert "s1" in store._open
    store.add_turn("s1", new_candidate(run_id="t1", turn_idx=1, final_text="grade"))
    store.commit_session("s1", expected_turns=2)
    assert [row["final_text"] for row in store.load_session("s1")] == ["quiz", "grade"]


def test_invalid_key_rejection_precedes_immutable_check(tmp_path):
    root = tmp_path / "store"
    store = CandidateStore(root)
    store.begin_session("s1")
    store.add_turn("s1", new_candidate(run_id="t0", turn_idx=0, final_text="quiz"))
    store.commit_session("s1", expected_turns=1)
    with pytest.raises(ValueError, match="unsafe session storage key"):
        store.commit_session("../escape", expected_turns=0)
