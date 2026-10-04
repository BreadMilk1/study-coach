"""Batch A — LLM error boundary: `app.llm.errors.normalize_llm_error` contract.

This is the single projection point that turns an exception object or a legacy
"<Category>: <detail>" string into fixed, category-only safe text. These tests
pin:

  - every whitelisted category (builtin exceptions plus SDK-style *class names*,
    which must never require importing the SDK) maps to its fixed sentence
  - unknown categories, unknown/empty legacy strings, non-string and None
  - exception internals (``str`` / ``repr`` / ``args`` / request / response) are
    never read — hostile objects that explode on any of them still normalize
  - detail markers (opaque, Bearer, credential assignment, URL query, JSON
    request body, multiline, long tail) never survive projection
  - re-projecting already-safe text is idempotent

The proxy fakes here are deliberately dependency-free: no SDK, logger or
provider client is constructed.
"""
from __future__ import annotations

import pytest

from app.llm.errors import normalize_llm_error

_CONNECT_TEXT = "Could not connect to the model service."
_TIMEOUT_TEXT = "The model request timed out."

EXPECTED = {
    "ConnectionRefusedError": f"ConnectionRefusedError: {_CONNECT_TEXT}",
    "ConnectionError": f"ConnectionError: {_CONNECT_TEXT}",
    "APIConnectionError": f"APIConnectionError: {_CONNECT_TEXT}",
    "TimeoutError": f"TimeoutError: {_TIMEOUT_TEXT}",
    "APITimeoutError": f"APITimeoutError: {_TIMEOUT_TEXT}",
    "AuthenticationError": "AuthenticationError: Model authentication failed.",
    "PermissionDeniedError": "PermissionDeniedError: Model access was denied.",
    "RateLimitError": "RateLimitError: The model service rate limit was reached.",
}
GENERIC = "LLMError: Model request failed."
ALL_SAFE = set(EXPECTED.values()) | {GENERIC}

# Builtin classes we can instantiate directly; SDK-style categories are matched
# by class *name* only, so they are synthesized instead of imported.
BUILTIN_TYPES = {
    "ConnectionRefusedError": ConnectionRefusedError,
    "ConnectionError": ConnectionError,
    "TimeoutError": TimeoutError,
}
SYNTHETIC_NAMES = sorted(set(EXPECTED) - set(BUILTIN_TYPES))

MARKER = "SECRET_OPAQUE_MARKER_7f3a"
DETAILS = [
    MARKER,
    f"Authorization: Bearer sk-live-{MARKER}",
    f'api_key="sk-live-{MARKER}"',
    f"https://api.openai.com/v1/chat/completions?api_key=sk-live-{MARKER}&trace={MARKER}",
    f'{{"api_key": "sk-live-{MARKER}", "input": "{MARKER}"}}',
    f"line one\nline two\n{MARKER}",
    "tail-" + ("x" * 5000) + f"-{MARKER}",
]


def _named_exception(name: str, message: str, *, base=Exception):
    return type(name, (base,), {})(message)


class HostileException(Exception):
    """Any attempt to read the detail explodes — normalization must not try."""

    _FORBIDDEN = ("__str__", "__repr__", "args", "request", "response",
                  "body", "headers", "endpoint")

    def __str__(self):  # pragma: no cover - only runs if the contract breaks
        raise AssertionError("normalize_llm_error called str(exc)")

    def __repr__(self):  # pragma: no cover - only runs if the contract breaks
        raise AssertionError("normalize_llm_error called repr(exc)")

    @property
    def args(self):  # pragma: no cover - only runs if the contract breaks
        raise AssertionError("normalize_llm_error read exc.args")

    @property
    def request(self):  # pragma: no cover - only runs if the contract breaks
        raise AssertionError("normalize_llm_error read exc.request")

    @property
    def response(self):  # pragma: no cover - only runs if the contract breaks
        raise AssertionError("normalize_llm_error read exc.response")

    @property
    def body(self):  # pragma: no cover - only runs if the contract breaks
        raise AssertionError("normalize_llm_error read exc.body")

    @property
    def headers(self):  # pragma: no cover - only runs if the contract breaks
        raise AssertionError("normalize_llm_error read exc.headers")


def test_builtin_whitelisted_exception_objects_map_to_fixed_text():
    for name, exc_type in BUILTIN_TYPES.items():
        assert normalize_llm_error(exc_type(f"detail {MARKER}")) == EXPECTED[name]


def test_sdk_style_whitelisted_class_names_map_to_fixed_text():
    # APIConnectionError / APITimeoutError / AuthenticationError /
    # PermissionDeniedError / RateLimitError are matched by class name so that
    # app code never has to import a provider SDK.
    for name in SYNTHETIC_NAMES:
        assert normalize_llm_error(_named_exception(name, f"detail {MARKER}")) == EXPECTED[name]


@pytest.mark.parametrize("name", sorted(EXPECTED))
@pytest.mark.parametrize("detail", DETAILS)
def test_every_whitelisted_category_discards_the_detail(name, detail):
    if name in BUILTIN_TYPES:
        out = normalize_llm_error(BUILTIN_TYPES[name](detail))
    else:
        out = normalize_llm_error(_named_exception(name, detail))
    assert out == EXPECTED[name]
    assert MARKER not in out
    assert detail not in out


def test_unknown_custom_exception_returns_generic_without_class_name():
    class SyntheticSecretError(Exception):
        pass

    out = normalize_llm_error(SyntheticSecretError(f"boom {MARKER}"))
    assert out == GENERIC
    assert "SyntheticSecretError" not in out
    assert MARKER not in out


def test_unknown_class_name_is_not_echoed_even_if_it_contains_a_category():
    hostile_name = f"ConnectionError{MARKER}"
    exc = _named_exception(hostile_name, f"boom {MARKER}")
    out = normalize_llm_error(exc)
    assert out == GENERIC
    assert MARKER not in out
    assert hostile_name not in out


def test_exception_object_taking_no_args_still_normalizes():
    class Bare(Exception):
        def __init__(self):
            super().__init__()

    assert normalize_llm_error(Bare()) == GENERIC
    assert normalize_llm_error(_named_exception("AuthenticationError", "")) == (
        EXPECTED["AuthenticationError"]
    )


@pytest.mark.parametrize(
    "exc",
    [
        HostileException(),
        _named_exception("ConnectionError", "detail", base=HostileException),
        _named_exception("AuthenticationError", "detail", base=HostileException),
    ],
)
def test_exception_detail_is_never_read(exc):
    # HostileException raises from __str__/__repr__/args/request/response; a
    # successful call proves the projection only used type information.
    out = normalize_llm_error(exc)
    assert out in ALL_SAFE
    if type(exc).__name__ in EXPECTED:
        assert out == EXPECTED[type(exc).__name__]


@pytest.mark.parametrize("name", sorted(EXPECTED))
@pytest.mark.parametrize("detail", DETAILS)
def test_legacy_whitelisted_category_drops_everything_after_the_colon(name, detail):
    assert normalize_llm_error(f"{name}: {detail}") == EXPECTED[name]


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_legacy_whitelisted_category_tolerates_surrounding_whitespace(name):
    assert normalize_llm_error(f"  {name}: secret tail  ") == EXPECTED[name]
    assert normalize_llm_error(f"\n{name}:\tsecret tail\n") == EXPECTED[name]
    assert normalize_llm_error(f"  {name}  : secret tail") == EXPECTED[name]


@pytest.mark.parametrize(
    "legacy",
    [
        f"SyntheticSecretError: {MARKER}",
        f"Unknown: Authorization: Bearer sk-live-{MARKER}",
        f"error: {MARKER}",
        "ConnectionRefusedError",          # no colon → not the legacy form
        "ConnectionError detail only",
        "AuthenticationErrorModel authentication failed.",  # no colon
        "",
        "   ",
        "\n\t",
        ":",
        ": detail without category",
        f"Bearer sk-live-{MARKER}",
        f"https://api.example.com/v1?api_key=sk-live-{MARKER}",
        f'{{"api_key": "sk-live-{MARKER}"}}',
    ],
)
def test_unknown_or_malformed_legacy_strings_return_generic(legacy):
    assert normalize_llm_error(legacy) == GENERIC


def test_generic_safe_text_is_returned_unchanged():
    # "LLMError" is intentionally not a whitelisted detail category; the
    # generic sentence must still round-trip.
    assert normalize_llm_error(GENERIC) == GENERIC


@pytest.mark.parametrize(
    "value",
    [0, 1, 1.5, True, b"bytes-secret", ["list"], {"api_key": "x"}, object()],
)
def test_non_string_non_none_values_return_generic(value):
    assert normalize_llm_error(value) == GENERIC


def test_none_field_stays_none():
    assert normalize_llm_error(None) is None


@pytest.mark.parametrize("value", [
    ConnectionError(f"detail {MARKER}"),
    "ConnectionRefusedError: ollama not running",
    "AuthenticationError: sk-live-abc",
    None,
    "unknown legacy text",
    42,
])
def test_projection_is_idempotent(value):
    once = normalize_llm_error(value)
    assert normalize_llm_error(once) == once


def test_projection_output_is_always_one_of_the_fixed_sentences():
    for name in sorted(EXPECTED):
        values = [
            _named_exception(name, f"{MARKER}"),
            f"{name}: {MARKER}",
            f"{name}",
        ]
        for value in values:
            assert normalize_llm_error(value) in ALL_SAFE
    for value in [HostileException(), MARKER, 3.14, None]:
        assert normalize_llm_error(value) in ALL_SAFE | {None}
