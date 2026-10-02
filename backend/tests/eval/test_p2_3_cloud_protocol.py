import copy
import socket

from app.eval.p2_3_cloud_capability.protocol import (
    canonical_json_bytes,
    extract_thinking_tokens,
    extract_usage,
    failure_class_for_exception,
    flatten_text_content,
    langchain_response_canonical,
    normalize_finish_status,
    raw_usage_from_response,
)


def test_flatten_drops_thinking_keeps_text():
    content = [
        {"type": "thinking", "thinking": "secret chain"},
        {"type": "text", "text": '[{"prompt":"Q"}]'},
        {"type": "tool_use", "name": "retriever_search", "id": "1"},
    ]
    assert flatten_text_content(content) == '[{"prompt":"Q"}]'


def test_flatten_string_passthrough():
    assert flatten_text_content('[{"prompt":"Q"}]') == '[{"prompt":"Q"}]'


def test_flatten_none_and_empty():
    assert flatten_text_content(None) == ""
    assert flatten_text_content([]) == ""


def test_anthropic_end_turn_is_completed():
    assert normalize_finish_status(protocol="anthropic", raw_stop="end_turn") == "completed"
    assert normalize_finish_status(protocol="anthropic", raw_stop="tool_use") == "completed"
    assert normalize_finish_status(protocol="anthropic", raw_stop="max_tokens") == "truncated"


def test_openai_stop_and_length():
    assert normalize_finish_status(protocol="openai_compatible", raw_stop="stop") == "completed"
    assert normalize_finish_status(protocol="openai_compatible", raw_stop="length") == "truncated"


def test_unknown_stop_is_failed():
    assert normalize_finish_status(protocol="anthropic", raw_stop="banana") == "failed"
    assert normalize_finish_status(protocol="openai_compatible", raw_stop=None) == "failed"


def test_thinking_tokens_by_protocol_nested_path():
    anthropic_usage = {"output_tokens": 681, "output_tokens_details": {"thinking_tokens": 309}}
    openai_usage = {"completion_tokens": 621, "completion_tokens_details": {"reasoning_tokens": 200}}
    assert extract_thinking_tokens(protocol="anthropic", usage=anthropic_usage) == 309
    assert extract_thinking_tokens(protocol="openai_compatible", usage=openai_usage) == 200


def test_thinking_tokens_missing_is_unavailable_not_zero():
    assert extract_thinking_tokens(protocol="anthropic", usage={"output_tokens": 10}) == "unavailable"
    assert extract_thinking_tokens(protocol="anthropic", usage=None) == "unavailable"


def test_extract_usage_missing_is_none():
    class Resp:
        usage_metadata = None
        response_metadata = {}
    assert extract_usage(Resp()) is None


def test_extract_usage_reads_usage_metadata():
    class Resp:
        usage_metadata = {"input_tokens": 11, "output_tokens": 4, "total_tokens": 15}
        response_metadata = {}
    assert extract_usage(Resp()) == {
        "input_tokens": 11, "output_tokens": 4, "total_tokens": 15
    }


def test_raw_usage_from_response_returns_nested_object_not_top_level_keys():
    class Resp:
        usage_metadata = {"input_tokens": 11, "output_tokens": 681}
        response_metadata = {
            "usage": {
                "output_tokens": 681,
                "output_tokens_details": {"thinking_tokens": 309},
            }
        }
    raw = raw_usage_from_response(Resp())
    assert raw["output_tokens_details"]["thinking_tokens"] == 309


def test_attribute_error_is_harness_not_model():
    assert failure_class_for_exception(AttributeError("astream")) == "harness"
    assert failure_class_for_exception(ConnectionError("refused")) == "transport"
    assert failure_class_for_exception(RuntimeError("boom")) == "model"


def test_provider_http_errors_are_transport():
    class AuthenticationError(Exception):
        pass

    class RateLimitError(Exception):
        pass

    assert failure_class_for_exception(AuthenticationError("401")) == "transport"
    assert failure_class_for_exception(RateLimitError("429")) == "transport"


_SENTINEL = "sk-synthetic-sentinel-not-a-real-credential"


class _Response:
    def __init__(self, **fields):
        self.__dict__.update(fields)


def _rendered(response) -> str:
    return canonical_json_bytes(langchain_response_canonical(response)).decode("utf-8")


def test_secret_keys_are_matched_after_hyphen_normalisation():
    headers = {
        "x-api-key": _SENTINEL,
        "X-API-KEY": _SENTINEL,
        "service-API-Key": _SENTINEL,
        "api_key": _SENTINEL,
        "API-KEY": _SENTINEL,
        "x_api_key": _SENTINEL,
        "X-Request-ID": "keep-request-id",
    }
    response = _Response(
        content=None,
        tool_calls=None,
        usage_metadata=None,
        response_metadata={"headers": headers},
    )
    payload = langchain_response_canonical(response)
    assert _SENTINEL not in _rendered(response)
    assert payload["response_metadata"]["headers"] == {"X-Request-ID": "keep-request-id"}


def test_secret_keys_are_redacted_in_nested_metadata_and_tool_call_input():
    response = _Response(
        content=[
            {
                "type": "tool_use",
                "id": "1",
                "name": "search",
                "input": {"x-api-key": _SENTINEL, "query": "keep-query"},
            }
        ],
        tool_calls=[
            {
                "id": "1",
                "name": "search",
                "input": {"service-api-key": _SENTINEL, "X-API-KEY": _SENTINEL},
            }
        ],
        usage_metadata=None,
        response_metadata={"usage": {"x-api-key": _SENTINEL, "output_tokens": 5}},
    )
    rendered = _rendered(response)
    assert _SENTINEL not in rendered
    assert "keep-query" in rendered
    assert '"output_tokens":5' in rendered


def test_only_secret_keys_are_dropped_and_input_is_not_mutated():
    metadata = {"note": _SENTINEL, "headers": {"x-api-key": _SENTINEL, "X-Request-ID": "keep"}}
    response = _Response(
        content=[{"type": "text", "text": _SENTINEL}],
        tool_calls=None,
        usage_metadata=None,
        response_metadata=metadata,
    )
    before = copy.deepcopy(response.__dict__)

    payload = langchain_response_canonical(response)

    assert response.__dict__ == before
    assert payload["response_metadata"]["note"] == _SENTINEL
    assert payload["content"][0]["text"] == _SENTINEL
    assert "keep" in canonical_json_bytes(payload).decode("utf-8")


def _usage_response(source: str, raw: dict):
    if source == "usage_metadata":
        return _Response(usage_metadata=raw, response_metadata={})
    if source == "response_metadata_usage":
        return _Response(usage_metadata=None, response_metadata={"usage": raw})
    return _Response(usage_metadata=None, response_metadata={"token_usage": raw})


def test_extract_usage_negative_total_falls_back_to_input_plus_output():
    for source in ("usage_metadata", "response_metadata_usage", "response_metadata_token_usage"):
        raw = {"input_tokens": 11, "output_tokens": 4, "total_tokens": -15}
        assert extract_usage(_usage_response(source, raw)) == {
            "input_tokens": 11,
            "output_tokens": 4,
            "total_tokens": 15,
        }, source


def test_extract_usage_missing_or_non_int_total_falls_back_to_input_plus_output():
    for total in (None, "15", 15.0, True, [], {}):
        raw = {"input_tokens": 11, "output_tokens": 4, "total_tokens": total}
        assert extract_usage(_usage_response("usage_metadata", raw)) == {
            "input_tokens": 11,
            "output_tokens": 4,
            "total_tokens": 15,
        }, total
    assert extract_usage(_usage_response("usage_metadata", {"input_tokens": 11, "output_tokens": 4})) == {
        "input_tokens": 11,
        "output_tokens": 4,
        "total_tokens": 15,
    }


def test_extract_usage_keeps_every_nonnegative_integer_total():
    for total in (0, 1, 99):
        raw = {"input_tokens": 11, "output_tokens": 4, "total_tokens": total}
        assert extract_usage(_usage_response("usage_metadata", raw))["total_tokens"] == total


def test_extract_usage_skips_sources_with_invalid_input_or_output():
    for raw in (
        {"input_tokens": -1, "output_tokens": 4},
        {"input_tokens": 11, "output_tokens": "4"},
        {"input_tokens": None, "output_tokens": 4},
        {"input_tokens": 11, "output_tokens": -4},
    ):
        assert extract_usage(_usage_response("usage_metadata", raw)) is None, raw


def test_raw_usage_from_response_keeps_the_original_nested_usage():
    raw_usage = {
        "input_tokens": 11,
        "output_tokens": 4,
        "total_tokens": -15,
        "output_tokens_details": {"thinking_tokens": 3},
    }
    before = copy.deepcopy(raw_usage)
    response = _Response(usage_metadata=None, response_metadata={"usage": raw_usage})

    got = raw_usage_from_response(response)

    assert got is raw_usage
    assert got["total_tokens"] == -15
    assert raw_usage == before


class APIConnectionError(Exception):
    pass


def test_local_file_permission_and_directory_errors_are_harness():
    for exc in (
        FileNotFoundError("missing"),
        PermissionError("denied"),
        IsADirectoryError("is a directory"),
        NotADirectoryError("not a directory"),
        OSError(28, "No space left on device"),
        OSError("plain local failure"),
    ):
        assert failure_class_for_exception(exc) == "harness", exc


def test_network_timeout_and_dns_errors_are_transport():
    for exc in (
        ConnectionError("refused"),
        TimeoutError("timed out"),
        socket.gaierror("name resolution"),
        socket.herror("host not found"),
    ):
        assert failure_class_for_exception(exc) == "transport", exc


def test_sdk_wrapped_generic_oserror_and_dns_stay_transport():
    for inner in (OSError("generic transport failure"), socket.gaierror("dns failure")):
        outer = APIConnectionError("connection error")
        outer.__cause__ = inner
        assert failure_class_for_exception(outer) == "transport", inner


def test_wrapper_around_local_error_or_attribute_error_is_harness():
    for inner in (FileNotFoundError("missing"), PermissionError("denied"), AttributeError("astream")):
        outer = APIConnectionError("connection error")
        outer.__cause__ = inner
        assert failure_class_for_exception(outer) == "harness", inner


def test_cause_is_preferred_over_context_and_cycles_terminate():
    outer = RuntimeError("outer")
    outer.__cause__ = AttributeError("astream")
    outer.__context__ = ConnectionError("refused")
    assert failure_class_for_exception(outer) == "harness"

    first = RuntimeError("first")
    second = APIConnectionError("second")
    first.__cause__ = second
    second.__cause__ = first
    assert failure_class_for_exception(first) == "transport"

    chained = RuntimeError("chained")
    chained.__context__ = FileNotFoundError("missing")
    assert failure_class_for_exception(chained) == "harness"

    assert failure_class_for_exception(RuntimeError("plain")) == "model"
