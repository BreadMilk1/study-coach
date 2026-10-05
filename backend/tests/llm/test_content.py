"""Batch C — strict text boundary for LangChain message content.

`extract_text` / `require_text` are the single place where message-content
shapes are interpreted. These tests pin the accepted shapes, the fixed
rejection messages, the skip list, and the no-mutation guarantee. They use
real `AIMessage` / `AIMessageChunk` carriers so the tested values are the ones
a provider adapter actually produces.
"""

import copy

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk

from app.llm.content import (
    EMPTY_CONTENT_MESSAGE,
    MALFORMED_CONTENT_MESSAGE,
    SKIPPED_CONTENT_BLOCK_TYPES,
    extract_text,
    require_text,
)

_SENTINEL = "SENTINEL-DO-NOT-LEAK"

_THINKING = {"type": "thinking", "thinking": _SENTINEL}


def test_fixed_messages_are_the_published_contract():
    assert MALFORMED_CONTENT_MESSAGE == "LLM response content is not a supported text shape"
    assert EMPTY_CONTENT_MESSAGE == "LLM response text has no non-whitespace body"


def test_skip_list_is_the_fixed_contract():
    assert SKIPPED_CONTENT_BLOCK_TYPES == frozenset(
        {
            "reasoning",
            "thinking",
            "redacted_thinking",
            "tool_use",
            "tool_result",
            "tool_call",
            "tool_call_chunk",
            "invalid_tool_call",
            "server_tool_call",
            "server_tool_call_chunk",
            "server_tool_result",
            "image",
            "image_url",
            "audio",
            "file",
            "text-plain",
            "video",
        }
    )


def test_plain_string_is_returned_byte_for_byte():
    content = (
        "  keep leading space\n"
        "\tkeep tab\n"
        "  <think>not stripped</think>\n"
        '{"json": "not reparsed"}\n'
        "trailing newline stays\n\n"
    )

    assert extract_text(content) == content
    assert require_text(content) == content


def test_plain_string_is_not_stripped_and_separators_are_not_added():
    assert extract_text("") == ""
    assert extract_text("   ") == "   "
    assert extract_text("\n\n") == "\n\n"


def test_bare_strings_and_text_blocks_concatenate_in_order_without_separator():
    content = [
        "alpha",
        {"type": "text", "text": "beta"},
        "gamma",
        {"type": "text", "text": "delta"},
    ]

    assert extract_text(content) == "alphabetagammadelta"


def test_json_split_across_blocks_and_bare_strings_is_preserved():
    content = [
        {"type": "text", "text": '{"prompt": "Q?"'},
        ',"options": ["A) x", ',
        {"type": "text", "text": '"B) y"]'},
    ]

    assert extract_text(content) == '{"prompt": "Q?","options": ["A) x", "B) y"]'


def test_whitespace_and_newlines_inside_and_between_blocks_are_preserved():
    content = [
        {"type": "text", "text": "\n  first\n"},
        "\n",
        {"type": "text", "text": "  second  \n"},
    ]

    assert extract_text(content) == "\n  first\n\n  second  \n"


def test_text_block_metadata_never_enters_the_body():
    content = [
        {
            "type": "text",
            "text": "body",
            "id": "block-1",
            "index": 0,
            "annotations": [{"type": "citation", "url": "https://example.invalid"}],
            "extras": {"trace": _SENTINEL},
        }
    ]

    assert extract_text(content) == "body"


def test_known_non_text_block_without_text_field_is_skipped():
    content = [
        _THINKING,
        {"type": "tool_use", "id": "t1", "name": "persist", "input": {"a": 1}},
        {"type": "text", "text": "visible"},
    ]

    assert extract_text(content) == "visible"


def test_known_non_text_block_with_text_like_field_is_not_consumed():
    """Skip-list blocks are opaque: a `text`-shaped key must not be guessed."""

    content = [
        {"type": "reasoning", "text": _SENTINEL},
        {"type": "tool_result", "text": _SENTINEL},
        {"type": "image_url", "text": _SENTINEL},
        {"type": "text", "text": "visible"},
    ]

    extracted = extract_text(content)

    assert extracted == "visible"
    assert _SENTINEL not in extracted


@pytest.mark.parametrize("block_type", sorted(SKIPPED_CONTENT_BLOCK_TYPES))
def test_every_skip_list_type_is_ignored_without_inspecting_its_payload(block_type):
    content = [
        {"type": block_type, "text": _SENTINEL, "content": _SENTINEL},
        {"type": "text", "text": "visible"},
    ]

    assert extract_text(content) == "visible"


def test_only_skipped_blocks_extract_to_empty_string():
    content = [_THINKING, {"type": "tool_use", "id": "t1", "name": "x", "input": {}}]

    assert extract_text(content) == ""


def test_empty_list_extracts_to_empty_string():
    assert extract_text([]) == ""


def test_extract_text_does_not_mutate_the_content():
    content = [
        _THINKING,
        {"type": "text", "text": "visible", "extra": {"nested": [1, 2]}},
        "bare",
    ]
    before = copy.deepcopy(content)

    extract_text(content)

    assert content == before


def test_real_ai_message_and_chunk_blocks_are_extracted_in_order():
    blocks = [
        _THINKING,
        {"type": "text", "text": "Hello "},
        "world",
        {"type": "text", "text": "!"},
    ]

    message = AIMessage(content=copy.deepcopy(blocks))
    chunk = AIMessageChunk(content=copy.deepcopy(blocks))

    assert extract_text(message.content) == "Hello world!"
    assert extract_text(chunk.content) == "Hello world!"


@pytest.mark.parametrize(
    "malformed",
    [
        {"type": "unknown_vendor_block", "text": _SENTINEL},
        {"type": "non_standard", "value": {"type": "text", "text": _SENTINEL}},
        {"text": _SENTINEL},
        {"type": None, "text": _SENTINEL},
        {"type": 7, "text": _SENTINEL},
        {"type": ["text"], "text": _SENTINEL},
        {"type": "text", "text": None},
        {"type": "text", "text": 5},
        {"type": "text", "text": {"nested": _SENTINEL}},
        {"type": "text", "text": ["a", "b"]},
        {"type": "text"},
        {"type_typo": "text", "text": _SENTINEL},
    ],
)
def test_malformed_list_block_is_rejected_with_the_fixed_message(malformed):
    with pytest.raises(ValueError) as exc_info:
        extract_text([malformed])

    assert str(exc_info.value) == MALFORMED_CONTENT_MESSAGE
    assert _SENTINEL not in str(exc_info.value)


@pytest.mark.parametrize(
    "element",
    [1, None, True, b"bytes", ("a",), ["a"], object()],
)
def test_non_string_or_dict_list_element_is_rejected(element):
    with pytest.raises(ValueError) as exc_info:
        extract_text(["ok", element])

    assert str(exc_info.value) == MALFORMED_CONTENT_MESSAGE


def test_a_later_bad_element_rejects_the_whole_content():
    with pytest.raises(ValueError) as exc_info:
        extract_text(
            [
                {"type": "text", "text": "fine"},
                {"type": "unknown_vendor_block", "text": _SENTINEL},
            ]
        )

    assert str(exc_info.value) == MALFORMED_CONTENT_MESSAGE
    assert _SENTINEL not in str(exc_info.value)


@pytest.mark.parametrize(
    "unsupported",
    [None, 0, 1.5, True, {}, {"type": "text", "text": "x"}, (), object()],
)
def test_unsupported_top_level_content_is_rejected(unsupported):
    with pytest.raises(ValueError) as exc_info:
        extract_text(unsupported)

    assert str(exc_info.value) == MALFORMED_CONTENT_MESSAGE


def test_rejection_never_includes_the_offending_payload():
    with pytest.raises(ValueError) as exc_info:
        extract_text([{"type": "unknown_vendor_block", "text": _SENTINEL}])

    message = str(exc_info.value)
    assert message == MALFORMED_CONTENT_MESSAGE
    assert _SENTINEL not in message
    assert "unknown_vendor_block" not in message


@pytest.mark.parametrize(
    "blank",
    [
        "",
        "   ",
        "\n\t \n",
        [],
        [_THINKING],
        [{"type": "text", "text": ""}],
        "  \n  ",
    ],
)
def test_require_text_rejects_content_without_a_non_whitespace_body(blank):
    with pytest.raises(ValueError) as exc_info:
        require_text(blank)

    assert str(exc_info.value) == EMPTY_CONTENT_MESSAGE


def test_require_text_returns_the_unstripped_body():
    assert require_text("  body with padding  ") == "  body with padding  "
    assert require_text([{"type": "text", "text": "  x  "}]) == "  x  "


def test_require_text_uses_the_same_malformed_rejection_as_extract_text():
    with pytest.raises(ValueError) as exc_info:
        require_text([{"type": "non_standard", "value": _SENTINEL}])

    assert str(exc_info.value) == MALFORMED_CONTENT_MESSAGE
    assert _SENTINEL not in str(exc_info.value)
