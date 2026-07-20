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
