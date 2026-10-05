import importlib
import json
import os
import socket

import httpx
import pytest
from langchain_core.messages import AIMessage

from app.llm.provider import LLMConfig, get_chat_model, parse_llm_config


def test_parse_llm_config_uses_defaults_when_all_headers_missing():
    cfg = parse_llm_config()

    assert cfg.provider == "ollama"
    assert cfg.model == "gemma3:4b"
    assert cfg.api_key is None


def test_parse_llm_config_requires_api_key_for_cloud_providers():
    with pytest.raises(ValueError, match="api[-_ ]?key"):
        parse_llm_config(x_provider="openai", x_model="gpt-4o-mini")


def test_parse_llm_config_accepts_cloud_provider_with_key():
    cfg = parse_llm_config(
        x_provider="anthropic",
        x_model="claude-haiku-4-5",
        x_api_key="sk-xxx",
    )
    assert cfg.provider == "anthropic"
    assert cfg.api_key == "sk-xxx"


def test_judge_model_falls_back_to_main_model_when_not_set():
    cfg = parse_llm_config(x_provider="ollama", x_model="qwen2.5:7b")
    assert cfg.effective_judge_model() == "qwen2.5:7b"


def test_judge_model_uses_override_when_set():
    cfg = parse_llm_config(
        x_provider="ollama",
        x_model="qwen2.5:7b",
        x_judge_model="llama3.1:8b",
    )
    assert cfg.effective_judge_model() == "llama3.1:8b"


def test_ollama_chat_model_ignores_process_proxy_settings(monkeypatch):
    captured: dict[str, object] = {}

    def fake_init_chat_model(**kwargs):
        captured.update(kwargs)
        return "chat-model"

    monkeypatch.setattr("langchain.chat_models.init_chat_model", fake_init_chat_model)
    config = LLMConfig(
        provider="ollama",
        model="gemma4:e4b",
        base_url="http://127.0.0.1:11434",
    )

    assert get_chat_model(config) == "chat-model"
    assert captured == {
        "model": "gemma4:e4b",
        "model_provider": "ollama",
        "base_url": "http://127.0.0.1:11434",
        "client_kwargs": {"trust_env": False},
    }


def test_cloud_chat_model_kwargs_are_unchanged(monkeypatch):
    captured: dict[str, object] = {}

    def fake_init_chat_model(**kwargs):
        captured.update(kwargs)
        return "chat-model"

    monkeypatch.setattr("langchain.chat_models.init_chat_model", fake_init_chat_model)
    config = LLMConfig(
        provider="openai",
        model="gpt-4o-mini",
        api_key="sk-test",
        base_url="https://api.openai.test/v1",
    )

    assert get_chat_model(config, temperature=0.2) == "chat-model"
    assert captured == {
        "model": "gpt-4o-mini",
        "model_provider": "openai",
        "api_key": "sk-test",
        "base_url": "https://api.openai.test/v1",
        "temperature": 0.2,
    }


# ---------------------------------------------------------------------------
# Offline runtime-adapter verification
#
# Batch B ships the OpenAI and Google Gemini adapters as default runtime
# dependencies. Everything below runs without a provider: the whole module
# blocks real sockets, only synthetic keys are used, and the two request
# contracts exercise the real SDKs through an httpx.MockTransport. A missing
# runtime package must fail these tests, never skip them.
# ---------------------------------------------------------------------------

_OPENAI_KEY = "sk-offline-openai"
_GEMINI_KEY = "gm-offline-key"


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    """Turn any real socket connection into an immediate test failure."""

    def deny(*args, **kwargs):
        raise AssertionError("offline provider test attempted real network access")

    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket, "create_connection", deny)


def _single_request_transport(requests, respond):
    """Let the real client serve exactly one request; anything else fails."""

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert len(requests) == 1, f"unexpected extra provider request: {request.url}"
        return respond(request)

    return httpx.MockTransport(handler)


_RUNTIME_ADAPTERS = [
    pytest.param(
        "ollama", "gemma3:4b", None,
        "langchain_ollama", "ChatOllama", None,
        id="ollama",
    ),
    pytest.param(
        "anthropic", "claude-haiku-4-5", "sk-ant-offline",
        "langchain_anthropic", "ChatAnthropic", "anthropic_api_key",
        id="anthropic",
    ),
    pytest.param(
        "openai", "gpt-4o-mini", _OPENAI_KEY,
        "langchain_openai", "ChatOpenAI", "openai_api_key",
        id="openai",
    ),
    pytest.param(
        "gemini", "gemini-2.5-flash", _GEMINI_KEY,
        "langchain_google_genai", "ChatGoogleGenerativeAI", "google_api_key",
        id="gemini",
    ),
    pytest.param(
        "google_genai", "gemini-2.5-flash", _GEMINI_KEY,
        "langchain_google_genai", "ChatGoogleGenerativeAI", "google_api_key",
        id="google_genai",
    ),
]


@pytest.mark.parametrize(
    "provider,model,api_key,module_name,class_name,key_attr", _RUNTIME_ADAPTERS
)
def test_runtime_adapter_is_installed_and_constructs_offline(
    provider, model, api_key, module_name, class_name, key_attr
):
    adapter_cls = getattr(importlib.import_module(module_name), class_name)

    built = get_chat_model(LLMConfig(provider=provider, model=model, api_key=api_key))

    assert isinstance(built, adapter_cls)
    assert built.model == model
    if key_attr is not None:
        assert getattr(built, key_attr).get_secret_value() == api_key


def test_anthropic_adapter_keeps_its_default_base_url():
    built = get_chat_model(
        LLMConfig(
            provider="anthropic",
            model="claude-haiku-4-5",
            api_key="sk-ant-offline",
        )
    )

    assert built.anthropic_api_url == "https://api.anthropic.com"


def test_gemini_adapter_selects_developer_api_backend():
    built = get_chat_model(
        LLMConfig(provider="gemini", model="gemini-2.5-flash", api_key=_GEMINI_KEY)
    )

    assert built.vertexai is False


def _openai_chat_completion(text: str) -> dict:
    return {
        "id": "chatcmpl-offline",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
    }


def _assert_openai_chat_completions_request(request: httpx.Request) -> httpx.Response:
    assert request.method == "POST"
    assert request.url.scheme == "https"
    assert request.url.host == "api.openai.com"
    assert request.url.path == "/v1/chat/completions"
    assert request.headers["authorization"] == f"Bearer {_OPENAI_KEY}"
    body = json.loads(request.content)
    assert body["model"] == "gpt-4o-mini"
    assert body["messages"] == [{"role": "user", "content": "ping"}]
    assert body.get("stream", False) is False
    return httpx.Response(200, json=_openai_chat_completion("pong"))


def test_openai_chat_completions_request_contract_offline():
    requests: list[httpx.Request] = []
    config = LLMConfig(provider="openai", model="gpt-4o-mini", api_key=_OPENAI_KEY)

    with httpx.Client(
        transport=_single_request_transport(
            requests, _assert_openai_chat_completions_request
        )
    ) as http_client:
        model = get_chat_model(config, http_client=http_client)
        result = model.invoke("ping")

    assert len(requests) == 1
    assert isinstance(result, AIMessage)
    assert result.content == "pong"


def _gemini_generate_content(text: str) -> dict:
    return {
        "candidates": [
            {
                "content": {"parts": [{"text": text}], "role": "model"},
                "finishReason": "STOP",
                "index": 0,
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 3,
            "candidatesTokenCount": 1,
            "totalTokenCount": 4,
        },
        "modelVersion": "gemini-2.5-flash",
    }


def _assert_gemini_developer_api_request(request: httpx.Request) -> httpx.Response:
    assert request.method == "POST"
    assert request.url.scheme == "https"
    assert request.url.host == "generativelanguage.googleapis.com"
    assert request.url.path == "/v1beta/models/gemini-2.5-flash:generateContent"
    assert request.headers["x-goog-api-key"] == _GEMINI_KEY
    body = json.loads(request.content)
    assert body["contents"] == [{"parts": [{"text": "ping"}], "role": "user"}]
    return httpx.Response(200, json=_gemini_generate_content("pong"))


def test_gemini_developer_api_request_contract_offline():
    requests: list[httpx.Request] = []
    config = LLMConfig(provider="gemini", model="gemini-2.5-flash", api_key=_GEMINI_KEY)
    transport = _single_request_transport(requests, _assert_gemini_developer_api_request)

    model = get_chat_model(config, client_args={"transport": transport})
    result = model.invoke("ping")

    assert len(requests) == 1
    assert isinstance(result, AIMessage)
    assert result.content == "pong"


# The Google SDK resolves its backend from GOOGLE_GENAI_USE_ENTERPRISE first and
# GOOGLE_GENAI_USE_VERTEXAI second, regardless of the adapter-level flag.
_GEMINI_BACKEND_ENV_VARS = (
    "GOOGLE_GENAI_USE_ENTERPRISE",
    "GOOGLE_GENAI_USE_VERTEXAI",
)

_GEMINI_BACKEND_ENV_CASES = [
    pytest.param({}, id="unset"),
    pytest.param({"GOOGLE_GENAI_USE_ENTERPRISE": "true"}, id="enterprise"),
    pytest.param({"GOOGLE_GENAI_USE_VERTEXAI": "true"}, id="vertexai"),
    pytest.param(
        {"GOOGLE_GENAI_USE_ENTERPRISE": "true", "GOOGLE_GENAI_USE_VERTEXAI": "true"},
        id="enterprise+vertexai",
    ),
]


def _set_gemini_backend_env(monkeypatch, values):
    for name in _GEMINI_BACKEND_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    for name, value in values.items():
        monkeypatch.setenv(name, value)


def _assert_gemini_backend_env_restored(values):
    for name in _GEMINI_BACKEND_ENV_VARS:
        if name in values:
            assert os.environ[name] == values[name]
        else:
            assert name not in os.environ


@pytest.mark.parametrize("provider", ["gemini", "google_genai"])
@pytest.mark.parametrize("backend_env", _GEMINI_BACKEND_ENV_CASES)
def test_gemini_pins_developer_api_against_backend_environment(
    monkeypatch, provider, backend_env
):
    _set_gemini_backend_env(monkeypatch, backend_env)
    requests: list[httpx.Request] = []
    config = LLMConfig(provider=provider, model="gemini-2.5-flash", api_key=_GEMINI_KEY)
    transport = _single_request_transport(requests, _assert_gemini_developer_api_request)

    model = get_chat_model(config, client_args={"transport": transport})

    assert model.vertexai is False
    assert model.client.vertexai is False
    result = model.invoke("ping")

    assert len(requests) == 1
    assert isinstance(result, AIMessage)
    assert result.content == "pong"
    _assert_gemini_backend_env_restored(backend_env)


@pytest.mark.parametrize("provider", ["gemini", "google_genai"])
def test_gemini_construction_failure_restores_backend_environment(monkeypatch, provider):
    backend_env = {
        "GOOGLE_GENAI_USE_ENTERPRISE": "true",
        "GOOGLE_GENAI_USE_VERTEXAI": "true",
    }
    _set_gemini_backend_env(monkeypatch, backend_env)
    config = LLMConfig(provider=provider, model="gemini-2.5-flash", api_key=_GEMINI_KEY)

    with pytest.raises(ValueError, match="temperature"):
        get_chat_model(config, temperature=5.0)

    _assert_gemini_backend_env_restored(backend_env)
