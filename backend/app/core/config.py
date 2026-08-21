from dataclasses import dataclass
import os
from pathlib import Path
from typing import Optional, Tuple


# 解析 env line。
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


# 处理 strip_inline_comment 相关逻辑。
def _strip_inline_comment(value: str) -> str:
    quote: Optional[str] = None
    for index, char in enumerate(value):
        if char in {"'", '"'}:
            quote = None if quote == char else char
        if char == "#" and quote is None and (index == 0 or value[index - 1].isspace()):
            return value[:index].rstrip()
    return value


# 处理 load_env_file 相关逻辑。
def _load_env_file(path: Path) -> None:
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_env_line(line)
        if parsed is None:
            continue

        key, value = parsed
        os.environ.setdefault(key, value)


# 获取 bool env。
def _get_bool_env(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    return raw_value.strip().lower() in {"1", "true", "yes", "y", "on"}


# 获取 int env。
def _get_int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    try:
        return int(raw_value)
    except ValueError:
        return default


# 处理 debug_routes_default 相关逻辑。
def _debug_routes_default() -> bool:
    environment = os.getenv("OFFERPILOT_ENV", "local").strip().lower()
    return environment in {"local", "dev", "development", "test"}


# 处理 default_env_file 相关逻辑。
def _default_env_file() -> Path:
    configured_path = os.getenv("OFFERPILOT_ENV_FILE")
    if configured_path:
        return Path(configured_path).expanduser()

    return Path(__file__).resolve().parents[2] / ".env"


# 处理 default_sqlite_path 相关逻辑。
def _default_sqlite_path() -> str:
    return str(Path(__file__).resolve().parents[2] / "data" / "offerpilot.db")


def _default_knowledge_source_path() -> str:
    return str(Path(__file__).resolve().parents[3] / "data" / "knowledge")


def _default_qdrant_path() -> str:
    return str(Path(__file__).resolve().parents[2] / "data" / "qdrant")


def _default_fastembed_cache_path() -> str:
    return str(Path(__file__).resolve().parents[2] / "data" / "fastembed-cache")


_load_env_file(_default_env_file())


# 集中读取和保存 OfferPilot 后端运行配置。
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
    project_analysis_llm_timeout_seconds: float = float(
        os.getenv(
            "OFFERPILOT_PROJECT_ANALYSIS_LLM_TIMEOUT_SECONDS",
            "180",
        )
    )
    project_analysis_max_model_turns: int = _get_int_env(
        "OFFERPILOT_PROJECT_ANALYSIS_MAX_MODEL_TURNS",
        30,
    )
    project_analysis_max_tool_calls: int = _get_int_env(
        "OFFERPILOT_PROJECT_ANALYSIS_MAX_TOOL_CALLS",
        80,
    )
    project_analysis_max_total_tokens: int = _get_int_env(
        "OFFERPILOT_PROJECT_ANALYSIS_MAX_TOTAL_TOKENS",
        1_500_000,
    )
    project_analysis_max_request_chars: int = _get_int_env(
        "OFFERPILOT_PROJECT_ANALYSIS_MAX_REQUEST_CHARS",
        400_000,
    )
    project_analysis_max_findings: int = _get_int_env(
        "OFFERPILOT_PROJECT_ANALYSIS_MAX_FINDINGS",
        12,
    )
    project_analysis_max_quote_chars: int = _get_int_env(
        "OFFERPILOT_PROJECT_ANALYSIS_MAX_QUOTE_CHARS",
        400,
    )
    llm_planner_enabled: bool = _get_bool_env("OFFERPILOT_LLM_PLANNER_ENABLED", True)
    llm_planner_fallback_enabled: bool = _get_bool_env(
        "OFFERPILOT_LLM_PLANNER_FALLBACK_ENABLED",
        True,
    )
    context_compaction_enabled: bool = _get_bool_env(
        "OFFERPILOT_CONTEXT_COMPACTION_ENABLED",
        True,
    )
    context_max_input_tokens: int = _get_int_env(
        "OFFERPILOT_CONTEXT_MAX_INPUT_TOKENS",
        24_000,
    )
    context_compaction_trigger_tokens: int = _get_int_env(
        "OFFERPILOT_CONTEXT_COMPACTION_TRIGGER_TOKENS",
        16_000,
    )
    context_keep_recent_turns: int = _get_int_env(
        "OFFERPILOT_CONTEXT_KEEP_RECENT_TURNS",
        6,
    )
    context_summary_max_tokens: int = _get_int_env(
        "OFFERPILOT_CONTEXT_SUMMARY_MAX_TOKENS",
        2_000,
    )
    debug_routes_enabled: bool = _get_bool_env(
        "OFFERPILOT_ENABLE_DEBUG_ROUTES",
        _debug_routes_default(),
    )
    storage_backend: str = os.getenv("OFFERPILOT_STORAGE_BACKEND", "memory")
    sqlite_path: str = os.getenv("OFFERPILOT_SQLITE_PATH", _default_sqlite_path())
    knowledge_source_path: str = os.getenv(
        "OFFERPILOT_KNOWLEDGE_SOURCE_PATH",
        _default_knowledge_source_path(),
    )
    knowledge_markdown_sync_enabled: bool = _get_bool_env(
        "OFFERPILOT_KNOWLEDGE_MARKDOWN_SYNC_ENABLED",
        True,
    )
    rag_enabled: bool = _get_bool_env("OFFERPILOT_RAG_ENABLED", True)
    rag_qdrant_path: str = os.getenv(
        "OFFERPILOT_RAG_QDRANT_PATH",
        _default_qdrant_path(),
    )
    rag_collection_name: str = os.getenv(
        "OFFERPILOT_RAG_COLLECTION_NAME",
        "career_knowledge",
    )
    rag_dense_model: str = os.getenv(
        "OFFERPILOT_RAG_DENSE_MODEL",
        "BAAI/bge-small-zh-v1.5",
    )
    rag_sparse_model: str = os.getenv(
        "OFFERPILOT_RAG_SPARSE_MODEL",
        "Qdrant/bm25",
    )
    rag_fastembed_cache_path: str = os.getenv(
        "OFFERPILOT_RAG_FASTEMBED_CACHE_PATH",
        _default_fastembed_cache_path(),
    )
    rag_top_k: int = _get_int_env("OFFERPILOT_RAG_TOP_K", 5)
    project_discovery_enabled: bool = _get_bool_env(
        "OFFERPILOT_PROJECT_DISCOVERY_ENABLED", True
    )
    project_discovery_max_concurrency: int = _get_int_env(
        "OFFERPILOT_PROJECT_DISCOVERY_MAX_CONCURRENCY", 1
    )
    project_discovery_clone_timeout_seconds: int = _get_int_env(
        "OFFERPILOT_PROJECT_DISCOVERY_CLONE_TIMEOUT_SECONDS", 120
    )
    project_discovery_max_git_metadata_mb: int = _get_int_env(
        "OFFERPILOT_PROJECT_DISCOVERY_MAX_GIT_METADATA_MB", 100
    )
    asr_provider: str = os.getenv("OFFERPILOT_ASR_PROVIDER", "bailian")
    asr_api_key: str = os.getenv("OFFERPILOT_ASR_API_KEY", "")
    asr_workspace_id: str = os.getenv("OFFERPILOT_ASR_WORKSPACE_ID", "")
    asr_model: str = os.getenv(
        "OFFERPILOT_ASR_MODEL",
        "fun-asr-flash-2026-06-15",
    )
    asr_base_url: str = os.getenv("OFFERPILOT_ASR_BASE_URL", "")
    asr_timeout_seconds: float = float(
        os.getenv("OFFERPILOT_ASR_TIMEOUT_SECONDS", "60")
    )
    feishu_verification_token: str = os.getenv("FEISHU_VERIFICATION_TOKEN", "")
    feishu_app_id: str = os.getenv("FEISHU_APP_ID", "")
    feishu_app_secret: str = os.getenv("FEISHU_APP_SECRET", "")
    feishu_api_base_url: str = os.getenv(
        "FEISHU_API_BASE_URL",
        "https://open.feishu.cn/open-apis",
    )
    feishu_timeout_seconds: float = float(os.getenv("FEISHU_TIMEOUT_SECONDS", "15"))
    feishu_authorize_url: str = os.getenv(
        "FEISHU_AUTHORIZE_URL",
        "https://accounts.feishu.cn/open-apis/authen/v1/authorize",
    )
    dashboard_oauth_scope: str = os.getenv(
        "OFFERPILOT_DASHBOARD_OAUTH_SCOPE",
        "auth:user.id:read",
    )
    dashboard_public_base_url: str = os.getenv(
        "OFFERPILOT_DASHBOARD_PUBLIC_BASE_URL",
        "",
    )
    dashboard_session_secret: str = os.getenv(
        "OFFERPILOT_DASHBOARD_SESSION_SECRET",
        "",
    )
    dashboard_session_ttl_seconds: int = _get_int_env(
        "OFFERPILOT_DASHBOARD_SESSION_TTL_SECONDS",
        7 * 24 * 60 * 60,
    )
    feishu_calendar_sync_enabled: bool = _get_bool_env("FEISHU_CALENDAR_SYNC_ENABLED", False)
    feishu_calendar_id: str = os.getenv("FEISHU_CALENDAR_ID", "primary")
    feishu_calendar_auto_create_enabled: bool = _get_bool_env(
        "FEISHU_CALENDAR_AUTO_CREATE_ENABLED",
        True,
    )
    feishu_offerpilot_calendar_summary: str = os.getenv(
        "FEISHU_OFFERPILOT_CALENDAR_SUMMARY",
        "OfferPilot 秋招日历",
    )
    feishu_offerpilot_calendar_description: str = os.getenv(
        "FEISHU_OFFERPILOT_CALENDAR_DESCRIPTION",
        "OfferPilot 自动创建，用于记录秋招投递、笔试和面试提醒。",
    )
    feishu_offerpilot_calendar_permissions: str = os.getenv(
        "FEISHU_OFFERPILOT_CALENDAR_PERMISSIONS",
        "private",
    )
    feishu_calendar_timezone: str = os.getenv("FEISHU_CALENDAR_TIMEZONE", "Asia/Shanghai")
    feishu_interview_event_duration_minutes: int = _get_int_env(
        "FEISHU_INTERVIEW_EVENT_DURATION_MINUTES",
        60,
    )
    feishu_bitable_sync_enabled: bool = _get_bool_env("FEISHU_BITABLE_SYNC_ENABLED", False)
    feishu_bitable_app_token: str = os.getenv("FEISHU_BITABLE_APP_TOKEN", "")
    feishu_bitable_table_id: str = os.getenv("FEISHU_BITABLE_TABLE_ID", "")
    feishu_bitable_auto_create_enabled: bool = _get_bool_env(
        "FEISHU_BITABLE_AUTO_CREATE_ENABLED",
        True,
    )
    feishu_offerpilot_bitable_name: str = os.getenv(
        "FEISHU_OFFERPILOT_BITABLE_NAME",
        "OfferPilot 秋招投递表",
    )
    feishu_offerpilot_bitable_table_name: str = os.getenv(
        "FEISHU_OFFERPILOT_BITABLE_TABLE_NAME",
        "投递记录",
    )
    feishu_bitable_web_base_url: str = os.getenv(
        "FEISHU_BITABLE_WEB_BASE_URL",
        "https://feishu.cn/base",
    )
    feishu_bitable_pull_sync_enabled: bool = _get_bool_env(
        "FEISHU_BITABLE_PULL_SYNC_ENABLED",
        False,
    )
    feishu_bitable_pull_sync_interval_seconds: int = _get_int_env(
        "FEISHU_BITABLE_PULL_SYNC_INTERVAL_SECONDS",
        30,
    )
    feishu_bitable_pull_sync_page_size: int = _get_int_env(
        "FEISHU_BITABLE_PULL_SYNC_PAGE_SIZE",
        100,
    )


settings = Settings()
