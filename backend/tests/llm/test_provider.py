import pytest

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
