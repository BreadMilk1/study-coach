"""Cut P2.2-①b — unit tests for the AgentTrace dataclass.

Trace is the only structured record the eval harness pulls from a run, so
the serialize() shape is contractual. These tests pin:
  - record_iteration / record_tool_call append correctly
  - serialize() emits all expected keys with correct types/counts
  - exit_reason transitions: natural_stop / budget_exhausted / llm_call_failed
"""
import json
import time

from langchain_core.messages import AIMessage

from app.agent.agent_trace import AgentTrace


def _ai(content: str = "ok", tool_calls=None, input_tokens=10, output_tokens=5):
    """Build a stub AIMessage with usage_metadata so trace can record tokens."""
    msg = AIMessage(content=content, tool_calls=tool_calls or [])
    msg.usage_metadata = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    return msg


def test_record_iteration_and_tool_call_then_serialize_natural_stop():
    trace = AgentTrace(t_start=time.monotonic())
    # First iter: model called retriever_search
    resp1 = _ai(tool_calls=[{"name": "retriever_search", "args": {"query": "HyDE"}, "id": "c1"}],
                input_tokens=100, output_tokens=20)
    trace.record_iteration(resp1, iteration=0)
    trace.record_tool_call("retriever_search", {"query": "HyDE"}, "[]", error=False)
    # Second iter: model emitted final summary (no tool calls → natural stop)
    resp2 = _ai(content="here is your plan", tool_calls=[], input_tokens=120, output_tokens=80)
    trace.record_iteration(resp2, iteration=1)
    trace.exit_reason = "natural_stop"

    out = trace.serialize()
    assert out["total_iterations"] == 2
    assert out["total_tool_calls"] == 1
    assert out["tool_call_breakdown"] == {"retriever_search": 1}
    assert out["tool_errors"] == 0
    assert out["input_tokens"] == 220
    assert out["output_tokens"] == 100
    assert out["exit_reason"] == "natural_stop"
    assert out["llm_error"] is None
    assert isinstance(out["wall_time_s"], float)
    assert out["wall_time_s"] >= 0.0


def test_record_budget_exhaustion_and_tool_error():
    trace = AgentTrace(t_start=time.monotonic())
    trace.record_iteration(_ai(tool_calls=[{"name": "update_study_plan", "args": {}, "id": "c1"}]),
                           iteration=0)
    trace.record_tool_call(
        "update_study_plan", {"milestones": "not-a-list"},
        "Error calling update_study_plan: bad arg type", error=True,
    )
    trace.record_budget_exhaustion(max_iter=10)

    out = trace.serialize()
    assert out["exit_reason"] == "budget_exhausted"
    assert out["tool_errors"] == 1
    assert out["llm_error"] is None


def test_record_llm_error_sets_exit_reason_and_message():
    trace = AgentTrace(t_start=time.monotonic())
    trace.record_llm_error("ConnectionRefusedError: ollama not running")

    out = trace.serialize()
    assert out["exit_reason"] == "llm_call_failed"
    assert "ConnectionRefusedError" in out["llm_error"]
    assert out["total_iterations"] == 0
    assert out["total_tool_calls"] == 0


def test_last_persisted_question_id_returns_id_from_latest_successful_call():
    """Quiz agent inference helper: walk tool_calls in reverse, return question_id
    from the most recent successful persist_quiz_question call. Skip errored calls."""
    trace = AgentTrace(t_start=time.monotonic())
    # First successful persist
    trace.record_tool_call(
        "persist_quiz_question",
        {"topic": "HyDE", "prompt": "...", "options": ["A) x", "B) y", "C) z", "D) w"], "answer": "A", "explanation": "..."},
        '{"question_id": "q-first", "topic_id": "t-1", "persisted": true}',
        error=False,
    )
    # Subsequent errored persist (validation failure)
    trace.record_tool_call(
        "persist_quiz_question",
        {"topic": "BM25", "prompt": "...", "options": ["A) x"], "answer": "A", "explanation": "..."},
        '{"error": "invalid at options: List should have at least 4 items"}',
        error=True,
    )
    # Then another successful persist (LLM self-corrected)
    trace.record_tool_call(
        "persist_quiz_question",
        {"topic": "BM25", "prompt": "...", "options": ["A) x", "B) y", "C) z", "D) w"], "answer": "B", "explanation": "..."},
        '{"question_id": "q-corrected", "topic_id": "t-2", "persisted": true}',
        error=False,
    )

    assert trace.last_persisted_question_id() == "q-corrected"


def test_last_persisted_question_id_returns_none_when_no_successful_persist():
    trace = AgentTrace(t_start=time.monotonic())
    trace.record_tool_call(
        "persist_quiz_question", {}, '{"error": "..."}', error=True,
    )
    # Only errored persists + a retriever_search → returns None
    trace.record_tool_call("retriever_search", {"query": "x"}, "[]", error=False)
    assert trace.last_persisted_question_id() is None


def test_serialize_public_includes_ui_summary_and_tool_details():
    trace = AgentTrace(t_start=time.monotonic())
    trace.record_iteration(
        _ai(tool_calls=[{
            "name": "retriever_search",
            "args": {"query": "HyDE", "top_k": 3},
            "id": "c1",
        }], input_tokens=100, output_tokens=20),
        iteration=0,
    )
    trace.record_tool_call(
        "retriever_search",
        {"query": "HyDE", "top_k": 3},
        '[{"chunk_id":"c1","content":"long source text","page":1}]',
        error=False,
    )
    trace.exit_reason = "natural_stop"

    out = trace.serialize_public(node="planner")

    assert out["node"] == "planner"
    assert out["mode"] == "agent_loop"
    assert out["exit_reason"] == "natural_stop"
    assert out["total_iterations"] == 1
    assert out["total_tool_calls"] == 1
    assert out["input_tokens"] == 100
    assert out["output_tokens"] == 20
    assert out["tool_calls"] == [{
        "name": "retriever_search",
        "error": False,
        "args_preview": '{"query":"HyDE","top_k":3}',
        "output_preview": "1 chunks",
    }]


def test_serialize_public_truncates_error_output():
    trace = AgentTrace(t_start=time.monotonic())
    trace.record_tool_call(
        "update_study_plan",
        {"milestones": "bad"},
        "Error calling update_study_plan: " + ("x" * 500),
        error=True,
    )
    out = trace.serialize_public(node="planner")

    assert out["tool_calls"][0]["error"] is True
    assert out["tool_calls"][0]["output_preview"].startswith("Error calling update_study_plan")
    assert len(out["tool_calls"][0]["output_preview"]) <= 220


def test_serialize_public_does_not_leak_truncated_retriever_text():
    trace = AgentTrace(t_start=time.monotonic())
    trace.record_tool_call(
        "retriever_search",
        {"query": "Prompt Engineering"},
        '[{"chunk_id":"c1","content":"' + ("source text " * 80),
        error=False,
    )

    out = trace.serialize_public(node="quiz")

    assert out["tool_calls"][0]["output_preview"] == "retriever chunks"
    assert "source text" not in out["tool_calls"][0]["output_preview"]


# --- Batch A: LLM error detail boundary ------------------------------------

_MARKER = "SECRET_OPAQUE_MARKER_7f3a"
_SAFE_CONNECTION = "ConnectionError: Could not connect to the model service."
_SAFE_AUTH = "AuthenticationError: Model authentication failed."
_SAFE_GENERIC = "LLMError: Model request failed."


def test_record_llm_error_accepts_exception_object_and_drops_detail():
    trace = AgentTrace(t_start=time.monotonic())
    trace.record_iteration(_ai(), iteration=0)
    trace.record_tool_call("retriever_search", {"query": "HyDE"}, "[]", error=False)

    trace.record_llm_error(ConnectionError(f"ollama unreachable {_MARKER}"))

    # Entry point stores safe text, not the exception object or its detail.
    assert trace.llm_error == _SAFE_CONNECTION
    out = trace.serialize()
    assert out["exit_reason"] == "llm_call_failed"
    assert out["llm_error"] == _SAFE_CONNECTION
    assert _MARKER not in out["llm_error"]
    # Counters are untouched by the error path.
    assert out["total_iterations"] == 1
    assert out["total_tool_calls"] == 1
    assert out["tool_call_breakdown"] == {"retriever_search": 1}


def test_record_llm_error_accepts_legacy_string_and_drops_detail():
    trace = AgentTrace(t_start=time.monotonic())
    trace.record_llm_error(
        f"AuthenticationError: Incorrect API key provided: sk-live-{_MARKER}"
    )

    out = trace.serialize()
    assert out["exit_reason"] == "llm_call_failed"
    assert out["llm_error"] == _SAFE_AUTH
    assert _MARKER not in json.dumps(out)


def test_record_llm_error_keeps_none_field_none():
    trace = AgentTrace(t_start=time.monotonic())
    trace.record_llm_error(None)

    out = trace.serialize()
    assert out["exit_reason"] == "llm_call_failed"
    assert out["llm_error"] is None


def test_serialize_paths_project_raw_llm_error_assigned_outside_the_entry_point():
    """Hand-built or rehydrated traces that bypass record_llm_error must still
    not leak raw detail through either serialization path."""
    trace = AgentTrace(t_start=time.monotonic())
    trace.exit_reason = "llm_call_failed"
    trace.llm_error = f"SyntheticSecretError: Bearer sk-live-{_MARKER}"

    internal = trace.serialize()
    public = trace.serialize_public(node="planner")

    assert internal["llm_error"] == _SAFE_GENERIC
    assert public["llm_error"] == _SAFE_GENERIC
    assert _MARKER not in json.dumps(internal)
    assert _MARKER not in json.dumps(public)


def test_serialize_paths_keep_already_safe_text_idempotent():
    trace = AgentTrace(t_start=time.monotonic())
    trace.exit_reason = "llm_call_failed"
    trace.llm_error = _SAFE_CONNECTION

    assert trace.serialize()["llm_error"] == _SAFE_CONNECTION
    assert trace.serialize_public(node="quiz")["llm_error"] == _SAFE_CONNECTION
