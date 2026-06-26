from dataclasses import dataclass
import os
from pathlib import Path
from typing import Optional, Tuple


def _parse_env_line(line: str) -> Optional[Tuple[str, str]]:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None

    if stripped.startswith("export "):
        stripped = stripped[len("export ") :].lstrip()

    if "=" not in stripped:
        return None

    key, value = stripped.split("=", 1)
    key = key.strip()
    if not key or not all(char.isalnum() or char == "_" for char in key):
        return None

    value = _strip_inline_comment(value.strip())
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]

    return key, value


def _strip_inline_comment(value: str) -> str:
    quote: Optional[str] = None
    for index, char in enumerate(value):
        if char in {"'", '"'}:
            quote = None if quote == char else char
        if char == "#" and quote is None and (index == 0 or value[index - 1].isspace()):
            return value[:index].rstrip()
    return value


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_env_line(line)
        if parsed is None:
            continue

        key, value = parsed
        os.environ.setdefault(key, value)


def _get_bool_env(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    return raw_value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _debug_routes_default() -> bool:
    environment = os.getenv("OFFERPILOT_ENV", "local").strip().lower()
    return environment in {"local", "dev", "development", "test"}


def _default_env_file() -> Path:
    configured_path = os.getenv("OFFERPILOT_ENV_FILE")
    if configured_path:
        return Path(configured_path).expanduser()

    return Path(__file__).resolve().parents[2] / ".env"


def _default_sqlite_path() -> str:
    return str(Path(__file__).resolve().parents[2] / "data" / "offerpilot.db")


_load_env_file(_default_env_file())


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("OFFERPILOT_APP_NAME", "OfferPilot API")
    app_version: str = os.getenv("OFFERPILOT_APP_VERSION", "0.1.0")
    api_prefix: str = os.getenv("OFFERPILOT_API_PREFIX", "/api")
    environment: str = os.getenv("OFFERPILOT_ENV", "local")
    llm_provider: str = os.getenv("OFFERPILOT_LLM_PROVIDER", "deepseek")
    llm_api_key: str = os.getenv(
        "OFFERPILOT_LLM_API_KEY",
        os.getenv("DEEPSEEK_API_KEY", ""),
    )
    llm_base_url: str = os.getenv("OFFERPILOT_LLM_BASE_URL", "https://api.deepseek.com")
    llm_model: str = os.getenv("OFFERPILOT_LLM_MODEL", "deepseek-v4-flash")
    llm_timeout_seconds: float = float(os.getenv("OFFERPILOT_LLM_TIMEOUT_SECONDS", "30"))
    debug_routes_enabled: bool = _get_bool_env(
        "OFFERPILOT_ENABLE_DEBUG_ROUTES",
        _debug_routes_default(),
    )
    storage_backend: str = os.getenv("OFFERPILOT_STORAGE_BACKEND", "memory")
    sqlite_path: str = os.getenv("OFFERPILOT_SQLITE_PATH", _default_sqlite_path())
    feishu_verification_token: str = os.getenv("FEISHU_VERIFICATION_TOKEN", "")
    feishu_app_id: str = os.getenv("FEISHU_APP_ID", "")
    feishu_app_secret: str = os.getenv("FEISHU_APP_SECRET", "")
    feishu_api_base_url: str = os.getenv(
        "FEISHU_API_BASE_URL",
        "https://open.feishu.cn/open-apis",
    )
    feishu_timeout_seconds: float = float(os.getenv("FEISHU_TIMEOUT_SECONDS", "15"))


settings = Settings()
