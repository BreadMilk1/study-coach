import os
import threading
from contextlib import contextmanager, nullcontext

from pydantic import BaseModel

DEFAULT_PROVIDER = "ollama"
DEFAULT_MODEL = "gemma3:4b"
PROVIDERS_REQUIRING_KEY = {"openai", "anthropic", "google_genai", "gemini"}
SUPPORTED_PROVIDERS = frozenset({DEFAULT_PROVIDER, *PROVIDERS_REQUIRING_KEY})

_INVALID_PROVIDER_MESSAGE = "Provider must be ollama, openai, anthropic, or gemini."
_INVALID_MODEL_MESSAGE = "A valid model is required for the selected provider."
_INVALID_API_KEY_MESSAGE = "An API key is required for the selected provider."

_BACKEND_ENV_VARS = ("GOOGLE_GENAI_USE_ENTERPRISE", "GOOGLE_GENAI_USE_VERTEXAI")
_DEVELOPER_API_LOCK = threading.Lock()


class InvalidLLMConfigError(ValueError):
    """Narrow connection-config rejection carrying a fixed field and message.

    ``app.api.deps.get_llm_config`` maps exactly this type to the stable HTTP
    400 body; no other exception type crosses that boundary. The message is a
    fixed safe sentence and never contains the rejected header value.
    """

    def __init__(self, field: str, message: str) -> None:
        super().__init__(message)
        self.field = field
        self.message = message


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
    # Omitted provider keeps the Ollama default; an explicit empty/blank or
    # unknown provider is rejected. Values are only trimmed and lower-cased —
    # no SDK-level provider list widens product support.
    if x_provider is None:
        provider = DEFAULT_PROVIDER
    else:
        provider = x_provider.strip().lower()
        if provider not in SUPPORTED_PROVIDERS:
            raise InvalidLLMConfigError("provider", _INVALID_PROVIDER_MESSAGE)

    # An omitted model keeps the Ollama default only for Ollama; cloud
    # providers must name a model. An explicit blank model is rejected for
    # every provider. A non-empty model keeps its case and inner characters —
    # no remote model catalog is consulted.
    if x_model is None:
        if provider != DEFAULT_PROVIDER:
            raise InvalidLLMConfigError("model", _INVALID_MODEL_MESSAGE)
        model = DEFAULT_MODEL
    else:
        model = x_model.strip()
        if not model:
            raise InvalidLLMConfigError("model", _INVALID_MODEL_MESSAGE)

    # Cloud keys must be present and non-blank; a non-empty key is forwarded
    # byte-for-byte — no prefix guessing, no trimming. Ollama needs no key.
    if provider in PROVIDERS_REQUIRING_KEY and (
        x_api_key is None or not x_api_key.strip()
    ):
        raise InvalidLLMConfigError("api_key", _INVALID_API_KEY_MESSAGE)

    return LLMConfig(
        provider=provider,
        model=model,
        api_key=x_api_key,
        base_url=x_base_url,
        judge_model=x_judge_model,
    )


_PROVIDER_TO_LANGCHAIN = {"gemini": "google_genai"}


@contextmanager
def _developer_api_pinned():
    """Pin the API-key Developer API while a Google client is constructed.

    langchain-google-genai resolves the ``vertexai`` flag but does not forward
    it to the google-genai client, which re-reads GOOGLE_GENAI_USE_ENTERPRISE
    and GOOGLE_GENAI_USE_VERTEXAI from the process environment on its own.
    """
    with _DEVELOPER_API_LOCK:
        previous = {name: os.environ.get(name) for name in _BACKEND_ENV_VARS}
        for name in _BACKEND_ENV_VARS:
            os.environ[name] = "false"
        try:
            yield
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


def get_chat_model(config: LLMConfig, **kwargs):
    from langchain.chat_models import init_chat_model

    lc_provider = _PROVIDER_TO_LANGCHAIN.get(config.provider, config.provider)
    extras: dict = {}
    if config.api_key:
        extras["api_key"] = config.api_key
    if config.base_url:
        extras["base_url"] = config.base_url
    if lc_provider == "ollama":
        extras["client_kwargs"] = {"trust_env": False}

    developer_api = nullcontext()
    if lc_provider == "google_genai":
        extras["vertexai"] = False
        developer_api = _developer_api_pinned()

    with developer_api:
        return init_chat_model(
            model=config.model,
            model_provider=lc_provider,
            **extras,
            **kwargs,
        )
