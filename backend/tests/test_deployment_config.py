from pathlib import Path
import tomllib

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _environment(service: dict) -> dict[str, str]:
    environment = service.get("environment", {})
    if isinstance(environment, dict):
        return environment
    return dict(entry.split("=", 1) for entry in environment)


def _dotenv(path: Path) -> dict[str, str]:
    entries = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            key, value = line.split("=", 1)
            entries[key] = value
    return entries


def test_compose_limits_destructive_reset_to_loopback():
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    backend = compose["services"]["backend"]
    environment = _environment(backend)

    assert backend["ports"] == ["127.0.0.1:8000:8000"]
    assert environment["STUDY_COACH_LOCAL_MODE"] == "1"
    assert environment["CHROMA_PATH"] == "/app/data/chroma"
    assert "CHROMA_PERSIST_DIR" not in environment


def test_example_environment_keeps_destructive_reset_disabled():
    environment = _dotenv(ROOT / ".env.example")

    assert environment["STUDY_COACH_LOCAL_MODE"] == "0"
    assert environment["CHROMA_PATH"] == "./chroma_data"


def test_fly_explicitly_disables_destructive_reset():
    fly = tomllib.loads((ROOT / "fly.toml").read_text())

    assert fly["env"]["STUDY_COACH_LOCAL_MODE"] == "0"


def test_backend_container_still_listens_on_all_container_interfaces():
    dockerfile = (ROOT / "Dockerfile.backend").read_text()

    assert '"--host", "0.0.0.0"' in dockerfile


def test_backend_readme_is_available_before_dependency_sync():
    dockerfile = (ROOT / "Dockerfile.backend").read_text().splitlines()

    readme_copy = dockerfile.index("COPY backend/README.md ./README.md")
    dependency_sync = dockerfile.index("RUN uv sync --frozen --no-dev")
    assert readme_copy < dependency_sync


def test_docker_context_excludes_nested_build_and_runtime_artifacts():
    patterns = {
        line.strip()
        for line in (ROOT / ".dockerignore").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert {
        "**/.venv",
        "**/node_modules",
        "**/dist",
        "**/__pycache__",
        "**/*.pyc",
        "**/chroma_data",
        "**/*.db",
        "/data",
    } <= patterns


def test_compose_frontend_is_loopback_only_and_proxies_to_backend_service():
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    frontend = compose["services"]["frontend"]
    environment = _environment(frontend)

    assert frontend["ports"] == ["127.0.0.1:5173:5173"]
    assert environment["VITE_API_PROXY_TARGET"] == "http://backend:8000"


def test_compose_backend_uses_ollama_service_endpoint():
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    environment = _environment(compose["services"]["backend"])

    assert environment["OLLAMA_HOST"] == "http://ollama:11434"


def test_compose_ollama_is_loopback_only():
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    ollama = compose["services"]["ollama"]

    assert ollama["ports"] == ["127.0.0.1:11434:11434"]


def test_compose_ollama_pulls_required_models():
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    entrypoint = compose["services"]["ollama"]["entrypoint"]

    for model in ("nomic-embed-text", "gemma3:4b", "qwen2.5:7b"):
        assert f"ollama pull {model}" in entrypoint


def test_frontend_container_dev_server_listens_on_all_container_interfaces():
    dockerfile = (ROOT / "Dockerfile.frontend").read_text()

    assert 'CMD ["pnpm", "dev", "--host", "0.0.0.0"]' in dockerfile


def test_vite_proxy_target_uses_environment_with_localhost_fallback():
    vite_config = (ROOT / "frontend" / "vite.config.ts").read_text()

    assert "const env = loadEnv(mode, process.cwd())" in vite_config
    assert (
        "target: env.VITE_API_PROXY_TARGET || 'http://localhost:8000'"
        in vite_config
    )
