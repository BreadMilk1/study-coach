from pydantic import BaseModel

DEFAULT_PROVIDER = "ollama"
DEFAULT_MODEL = "gemma3:4b"
PROVIDERS_REQUIRING_KEY = {"openai", "anthropic", "google_genai", "gemini"}


class LLMConfig(BaseModel):
    provider: str
    model: str
    api_key: str | None = None
    base_url: str | None = None
    judge_model: str | None = None

    def effective_judge_model(self) -> str:
        return self.judge_model or self.model


def parse_llm_config(
    x_provider: str | None = None,
    x_model: str | None = None,
    x_api_key: str | None = None,
    x_base_url: str | None = None,
    x_judge_model: str | None = None,
) -> LLMConfig:
    provider = (x_provider or DEFAULT_PROVIDER).lower()
    model = x_model or DEFAULT_MODEL
    if provider in PROVIDERS_REQUIRING_KEY and not x_api_key:
        raise ValueError(f"x-api-key required for provider '{provider}'")
    return LLMConfig(
        provider=provider,
        model=model,
        api_key=x_api_key,
        base_url=x_base_url,
        judge_model=x_judge_model,
    )


_PROVIDER_TO_LANGCHAIN = {"gemini": "google_genai"}


def get_chat_model(config: LLMConfig, **kwargs):
    from langchain.chat_models import init_chat_model

    lc_provider = _PROVIDER_TO_LANGCHAIN.get(config.provider, config.provider)
    extras: dict = {}
    if config.api_key:
        extras["api_key"] = config.api_key
    if config.base_url:
        extras["base_url"] = config.base_url
    return init_chat_model(
        model=config.model,
        model_provider=lc_provider,
        **extras,
        **kwargs,
    )
