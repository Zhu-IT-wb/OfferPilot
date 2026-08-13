import asyncio
import pytest

from app.project_analysis.repository_command_executor import (
    RepositoryCommandExecutor,
    RepositoryCommandRejectedError,
)


def test_executor_runs_allowed_command_inside_repository(
    tmp_path,
) -> None:
    """允许的只读命令只能在仓库根目录中读取文件。"""

    readme = tmp_path / "README.md"
    readme.write_text(
        "first line\nsecond line\n",
        encoding="utf-8",
    )
    executor = RepositoryCommandExecutor(
        repository_dir=str(tmp_path),
    )

    result = asyncio.run(
        executor.execute(
            ["head", "-n", "1", "README.md"]
        )
    )

    assert result.exit_code == 0
    assert result.stdout == "first line\n"
    assert result.stderr == ""
    assert result.timed_out is False
    assert result.truncated is False


@pytest.mark.parametrize(
    "argv",
    [
        ["rm", "README.md"],
        ["git", "checkout", "--", "README.md"],
        ["git", "diff", "--no-index", "one", "two"],
        ["git", "grep", "-Oless", "needle"],
        ["rg", "--pre=sh", "needle"],
        ["rg", "--follow", "needle"],
        ["rg", "needle", "/etc/passwd"],
        ["rg", "--files", "/etc"],
        ["rg", "--regexp", "needle", "/etc"],
        ["tail", "--follow", "README.md"],
        ["head", "/etc/passwd"],
        ["wc", "../outside.txt"],
    ],
)
def test_executor_rejects_dangerous_commands_and_paths(
    tmp_path,
    argv,
) -> None:
    """拒绝非白名单命令、可执行选项和仓库外路径。"""

    executor = RepositoryCommandExecutor(
        repository_dir=str(tmp_path),
    )

    with pytest.raises(RepositoryCommandRejectedError):
        asyncio.run(executor.execute(argv))


def test_executor_does_not_treat_search_pattern_as_a_path(
    tmp_path,
) -> None:
    """以斜杠开头的代码搜索模式不是宿主绝对路径。"""

    executor = RepositoryCommandExecutor(
        repository_dir=str(tmp_path),
    )
    (tmp_path / "routes.py").write_text(
        "route = '/health'\n",
        encoding="utf-8",
    )

    result = asyncio.run(
        executor.execute(["rg", "/api/v1"])
    )

    assert result.exit_code == 1
    assert result.timed_out is False


def test_executor_rejects_symlink_that_leaves_repository(
    tmp_path,
) -> None:
    """即使参数是相对路径，也不能通过符号链接读取宿主文件。"""

    outside_file = tmp_path.parent / "outside-secret.txt"
    outside_file.write_text("secret", encoding="utf-8")
    link = tmp_path / "linked-secret.txt"
    link.symlink_to(outside_file)
    executor = RepositoryCommandExecutor(
        repository_dir=str(tmp_path),
    )

    with pytest.raises(RepositoryCommandRejectedError):
        asyncio.run(
            executor.execute(["head", "linked-secret.txt"])
        )


def test_executor_uses_minimal_environment(
    tmp_path,
    monkeypatch,
) -> None:
    """Agent 命令看不到运行 OfferPilot 服务所使用的环境变量。"""

    executable = tmp_path / "fake-rg"
    executable.write_text(
        "#!/bin/sh\nenv\n",
        encoding="utf-8",
    )
    executable.chmod(0o700)
    monkeypatch.setenv(
        "OFFERPILOT_TEST_SECRET",
        "must-not-leak",
    )
    executor = RepositoryCommandExecutor(
        repository_dir=str(tmp_path),
        executable_paths={"rg": str(executable)},
    )

    result = asyncio.run(executor.execute(["rg", "needle"]))

    assert "OFFERPILOT_TEST_SECRET" not in result.stdout
    assert "must-not-leak" not in result.stdout


def test_executor_limits_output_without_blocking_process(
    tmp_path,
) -> None:
    """超出限制的输出仍会被排空，但不会全部保留进内存。"""

    executable = tmp_path / "fake-rg"
    executable.write_text(
        "#!/bin/sh\ni=0\n"
        "while [ $i -lt 1000 ]; do "
        "printf x; i=$((i + 1)); done\n",
        encoding="utf-8",
    )
    executable.chmod(0o700)
    executor = RepositoryCommandExecutor(
        repository_dir=str(tmp_path),
        executable_paths={"rg": str(executable)},
    )

    result = asyncio.run(
        executor.execute(
            ["rg", "needle"],
            max_output_bytes=100,
        )
    )

    assert len(result.stdout) == 100
    assert result.truncated is True


def test_executor_shares_output_limit_between_stdout_and_stderr(
    tmp_path,
) -> None:
    """标准输出与错误输出共享同一个内存预算。"""

    executable = tmp_path / "fake-rg"
    executable.write_text(
        "#!/bin/sh\ni=0\n"
        "while [ $i -lt 80 ]; do "
        "printf x; printf y >&2; i=$((i + 1)); done\n",
        encoding="utf-8",
    )
    executable.chmod(0o700)
    executor = RepositoryCommandExecutor(
        repository_dir=str(tmp_path),
        executable_paths={"rg": str(executable)},
    )

    result = asyncio.run(
        executor.execute(
            ["rg", "needle"],
            max_output_bytes=100,
        )
    )

    assert len(result.stdout) + len(result.stderr) == 100
    assert result.truncated is True


def test_executor_times_out_and_terminates_process_group(
    tmp_path,
) -> None:
    """命令超时后返回受控结果，并终止对应进程组。"""

    executable = tmp_path / "fake-rg"
    executable.write_text(
        "#!/bin/sh\nsleep 10\n",
        encoding="utf-8",
    )
    executable.chmod(0o700)
    executor = RepositoryCommandExecutor(
        repository_dir=str(tmp_path),
        executable_paths={"rg": str(executable)},
    )

    result = asyncio.run(
        executor.execute(
            ["rg", "needle"],
            timeout_seconds=1,
        )
    )

    assert result.exit_code is None
    assert result.timed_out is True
    assert result.duration_ms < 3000
