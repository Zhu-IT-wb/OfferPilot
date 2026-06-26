import os

from app.core import config


def test_parse_env_line_supports_common_dotenv_syntax() -> None:
    assert config._parse_env_line("DEEPSEEK_API_KEY=sk-test") == (
        "DEEPSEEK_API_KEY",
        "sk-test",
    )
    assert config._parse_env_line('export OFFERPILOT_ENV="local" # comment') == (
        "OFFERPILOT_ENV",
        "local",
    )
    assert config._parse_env_line("# comment") is None
    assert config._parse_env_line("not a valid line") is None


def test_load_env_file_does_not_override_existing_env(monkeypatch, tmp_path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "OFFERPILOT_TEST_FROM_FILE=from_file",
                "OFFERPILOT_TEST_EXISTING=from_file",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("OFFERPILOT_TEST_FROM_FILE", raising=False)
    monkeypatch.setenv("OFFERPILOT_TEST_EXISTING", "from_shell")

    config._load_env_file(env_file)

    assert os.getenv("OFFERPILOT_TEST_FROM_FILE") == "from_file"
    assert os.getenv("OFFERPILOT_TEST_EXISTING") == "from_shell"


def test_get_bool_env_parses_common_values(monkeypatch) -> None:
    monkeypatch.setenv("OFFERPILOT_TEST_BOOL", "true")
    assert config._get_bool_env("OFFERPILOT_TEST_BOOL", default=False) is True

    monkeypatch.setenv("OFFERPILOT_TEST_BOOL", "0")
    assert config._get_bool_env("OFFERPILOT_TEST_BOOL", default=True) is False

    monkeypatch.delenv("OFFERPILOT_TEST_BOOL", raising=False)
    assert config._get_bool_env("OFFERPILOT_TEST_BOOL", default=True) is True
