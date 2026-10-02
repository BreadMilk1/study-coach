from app.eval.p2_3_cloud_capability.protocol import (
    extract_thinking_tokens,
    extract_usage,
    failure_class_for_exception,
    flatten_text_content,
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
