"""Strict text extraction from LangChain message content.

`BaseMessage.content` is either a plain string or a list of content blocks.
`BaseMessage.text` concatenates the text blocks, but it silently ignores block
types it does not recognise and text blocks whose `text` is not a string, so it
cannot be used where "the model returned no usable answer" must be an error
instead of an empty answer. This module is the one place that interprets the
content shape for the text consumers (Quiz, Judge, Tutor, Planner: the
deterministic GENERATE / CHECK-IN node, the `generate_mindmap` tool and the
final turn of the Planner agent loop) and for the model detection routes
(`/api/models/ping` requires a usable response body, and
`/api/models/tool-check` requires a usable body on both turns of its round
trip). Each consumer owns its own failure
policy — see the consumer sections of ARCHITECTURE.md.

Contract:
- A plain string is returned unchanged: no strip, no separator, no newline or
  `think`-tag rewriting, no JSON re-parsing.
- A list concatenates its elements in order with `""` as separator, keeping all
  whitespace, so JSON may span several blocks.
- A list element is either a bare string, or a dict whose `type` is exactly
  `"text"` and whose `text` is a `str`. Blocks whose `type` is in
  `SKIPPED_CONTENT_BLOCK_TYPES` are opaque non-answer payloads and are skipped
  without inspecting any of their other fields.
- Everything else — unknown or catch-all types, a missing/non-string `type`, an
  invalid `text` field, a non string/dict element, an unsupported top-level
  object — is rejected with `ValueError(MALFORMED_CONTENT_MESSAGE)`. The
  message is fixed and never contains the offending payload.
- The input content is never mutated.

This deliberately diverges from the tolerant evaluator-side
`app.eval.p2_3_cloud_capability.protocol.flatten_text_content`, which never
raises and is not used on the production answer path. Production must not
import from `app.eval`.
"""

from typing import Any

# Fixed list of recognised non-answer block types. Their payload is opaque: it
# is neither interpreted nor shown, and no tool/multimodal schema is validated.
SKIPPED_CONTENT_BLOCK_TYPES: frozenset[str] = frozenset(
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

MALFORMED_CONTENT_MESSAGE = "LLM response content is not a supported text shape"
EMPTY_CONTENT_MESSAGE = "LLM response text has no non-whitespace body"


def extract_text(content: Any) -> str:
    """Return the answer text carried by `content`, or `""` when it carries none.

    Raises `ValueError` when the content shape cannot be interpreted.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for element in content:
            if isinstance(element, str):
                parts.append(element)
                continue
            if isinstance(element, dict):
                block_type = element.get("type")
                if not isinstance(block_type, str):
                    raise ValueError(MALFORMED_CONTENT_MESSAGE)
                if block_type in SKIPPED_CONTENT_BLOCK_TYPES:
                    continue
                if block_type == "text":
                    text = element.get("text")
                    if not isinstance(text, str):
                        raise ValueError(MALFORMED_CONTENT_MESSAGE)
                    parts.append(text)
                    continue
                raise ValueError(MALFORMED_CONTENT_MESSAGE)
            raise ValueError(MALFORMED_CONTENT_MESSAGE)
        return "".join(parts)
    raise ValueError(MALFORMED_CONTENT_MESSAGE)


def require_text(content: Any) -> str:
    """Return the extracted body, rejecting a response with no usable answer.

    "No usable answer" means the extracted text has no non-whitespace
    character. The returned value is the unstripped text; the whitespace-only
    check never replaces the successful body.
    """
    text = extract_text(content)
    if not text.strip():
        raise ValueError(EMPTY_CONTENT_MESSAGE)
    return text


__all__ = [
    "EMPTY_CONTENT_MESSAGE",
    "MALFORMED_CONTENT_MESSAGE",
    "SKIPPED_CONTENT_BLOCK_TYPES",
    "extract_text",
    "require_text",
]
