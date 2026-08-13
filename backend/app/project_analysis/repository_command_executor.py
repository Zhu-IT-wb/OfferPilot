import asyncio
import os
import shutil
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple


class RepositoryCommandRejectedError(ValueError):
    """命令不符合仓库只读执行策略。"""


@dataclass(frozen=True)
class RepositoryCommandResult:
    """一次受控仓库命令的可序列化执行结果。"""

    argv: Tuple[str, ...]
    exit_code: Optional[int]
    stdout: str
    stderr: str
    timed_out: bool
    truncated: bool
    duration_ms: int

    def to_dict(self) -> dict:
        """返回适合放入 Tool 消息的字典。"""

        return {
            "argv": list(self.argv),
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "timed_out": self.timed_out,
            "truncated": self.truncated,
            "duration_ms": self.duration_ms,
        }


@dataclass
class _OutputBudget:
    """在 stdout 与 stderr 读取任务之间共享保留字节预算。"""

    remaining: int
    truncated: bool = False

    def retain(self, chunk: bytes) -> bytes:
        """消费剩余预算，并标记被丢弃的超额输出。"""

        retained = chunk[:self.remaining]
        self.remaining -= len(retained)
        if len(retained) < len(chunk):
            self.truncated = True
        return retained


class RepositoryCommandExecutor:
    """在固定仓库目录中执行严格白名单内的只读命令。"""

    MIN_TIMEOUT_SECONDS = 1
    DEFAULT_TIMEOUT_SECONDS = 20
    MAX_TIMEOUT_SECONDS = 30
    DEFAULT_MAX_OUTPUT_BYTES = 30_000
    MIN_OUTPUT_BYTES = 100
    MAX_OUTPUT_BYTES = 100_000
    MIN_ARGV_ITEMS = 1
    MAX_ARGV_ITEMS = 64
    MAX_ARGUMENT_CHARS = 1000

    _ALLOWED_PROGRAMS: Set[str] = {
        "git",
        "head",
        "rg",
        "tail",
        "wc",
    }
    _ALLOWED_GIT_SUBCOMMANDS: Set[str] = {
        "diff",
        "grep",
        "log",
        "ls-files",
        "rev-parse",
        "show",
        "status",
    }
    _BLOCKED_RG_OPTIONS: Tuple[str, ...] = (
        "--file",
        "--hostname-bin",
        "--follow",
        "--ignore-file",
        "--pre",
        "--pre-glob",
        "-f",
        "-L",
    )
    _BLOCKED_GIT_OPTIONS: Tuple[str, ...] = (
        "--ext-diff",
        "--no-index",
        "--open-files-in-pager",
        "--output",
        "--textconv",
        "-O",
    )
    _BLOCKED_TAIL_OPTIONS: Tuple[str, ...] = (
        "--follow",
        "--pid",
        "--retry",
        "-F",
        "-f",
    )
    _RG_OPTIONS_WITH_VALUES: Set[str] = {
        "--after-context",
        "--before-context",
        "--color",
        "--colors",
        "--context",
        "--context-separator",
        "--encoding",
        "--engine",
        "--field-context-separator",
        "--field-match-separator",
        "--glob",
        "--max-columns",
        "--max-count",
        "--path-separator",
        "--replace",
        "--sort",
        "--sortr",
        "--type",
        "--type-add",
        "--type-clear",
        "--type-not",
        "-A",
        "-B",
        "-C",
        "-E",
        "-M",
        "-T",
        "-g",
        "-m",
        "-r",
        "-t",
    }
    _RG_EXPLICIT_PATTERN_OPTIONS: Set[str] = {
        "--regexp",
        "-e",
    }
    _RG_PATH_ONLY_MODES: Set[str] = {
        "--files",
        "--type-list",
    }

    def __init__(
        self,
        repository_dir: str,
        executable_paths: Optional[Mapping[str, str]] = None,
        default_timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        default_max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
    ) -> None:
        """固定仓库根目录、程序路径和单次执行资源上限。"""

        repository_path = Path(repository_dir).resolve()
        if not repository_path.is_dir():
            raise ValueError(
                "repository_dir must be an existing directory."
            )
        if not (
            self.MIN_TIMEOUT_SECONDS
            <= default_timeout_seconds
            <= self.MAX_TIMEOUT_SECONDS
        ):
            raise ValueError(
                "default_timeout_seconds is outside the allowed range."
            )
        if not (
            self.MIN_OUTPUT_BYTES
            <= default_max_output_bytes
            <= self.MAX_OUTPUT_BYTES
        ):
            raise ValueError(
                "default_max_output_bytes is outside the allowed range."
            )

        self._repository_dir = repository_path
        self._default_timeout_seconds = default_timeout_seconds
        self._default_max_output_bytes = default_max_output_bytes
        self._executable_paths = self._resolve_executables(
            executable_paths
        )

    async def execute(
        self,
        argv: Sequence[str],
        timeout_seconds: Optional[int] = None,
        max_output_bytes: Optional[int] = None,
    ) -> RepositoryCommandResult:
        """校验参数后无 Shell 地运行命令并限制时间与输出。"""

        normalized_argv = self._validate_argv(argv)
        timeout = (
            timeout_seconds
            if timeout_seconds is not None
            else self._default_timeout_seconds
        )
        output_limit = (
            max_output_bytes
            if max_output_bytes is not None
            else self._default_max_output_bytes
        )

        if (
            timeout < self.MIN_TIMEOUT_SECONDS
            or timeout > self.MAX_TIMEOUT_SECONDS
        ):
            raise RepositoryCommandRejectedError(
                "timeout_seconds must be between "
                f"{self.MIN_TIMEOUT_SECONDS} and "
                f"{self.MAX_TIMEOUT_SECONDS}."
            )
        if (
            output_limit < self.MIN_OUTPUT_BYTES
            or output_limit > self.MAX_OUTPUT_BYTES
        ):
            raise RepositoryCommandRejectedError(
                "max_output_bytes must be between "
                f"{self.MIN_OUTPUT_BYTES} and "
                f"{self.MAX_OUTPUT_BYTES}."
            )

        program = normalized_argv[0]
        executable = self._executable_paths.get(program)
        if executable is None:
            raise RepositoryCommandRejectedError(
                f"Allowed command is unavailable: {program}."
            )

        command = [executable, *normalized_argv[1:]]
        if program == "git":
            command.insert(1, "--no-pager")

        started_at = time.monotonic()
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=str(self._repository_dir),
            env=self._minimal_environment(),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        output_budget = _OutputBudget(output_limit)
        stdout_task = asyncio.create_task(
            self._read_limited(
                process.stdout,
                output_budget,
            )
        )
        stderr_task = asyncio.create_task(
            self._read_limited(
                process.stderr,
                output_budget,
            )
        )
        timed_out = False

        try:
            await asyncio.wait_for(
                process.wait(),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            timed_out = True
            await self._terminate_process(process)
        except asyncio.CancelledError:
            await self._terminate_process(process)
            await asyncio.gather(
                stdout_task,
                stderr_task,
                return_exceptions=True,
            )
            raise

        stdout_bytes = await stdout_task
        stderr_bytes = await stderr_task
        duration_ms = int(
            (time.monotonic() - started_at) * 1000
        )

        return RepositoryCommandResult(
            argv=tuple(normalized_argv),
            exit_code=(
                None if timed_out else process.returncode
            ),
            stdout=stdout_bytes.decode(
                "utf-8",
                errors="replace",
            ),
            stderr=stderr_bytes.decode(
                "utf-8",
                errors="replace",
            ),
            timed_out=timed_out,
            truncated=output_budget.truncated,
            duration_ms=duration_ms,
        )

    def _validate_argv(
        self,
        argv: Sequence[str],
    ) -> List[str]:
        """校验通用参数、命令白名单和命令专属危险选项。"""

        if isinstance(argv, (str, bytes)):
            raise RepositoryCommandRejectedError(
                "argv must be an array of strings."
            )

        normalized = list(argv)
        if not (
            self.MIN_ARGV_ITEMS
            <= len(normalized)
            <= self.MAX_ARGV_ITEMS
        ):
            raise RepositoryCommandRejectedError(
                "argv must contain between "
                f"{self.MIN_ARGV_ITEMS} and "
                f"{self.MAX_ARGV_ITEMS} items."
            )

        for argument in normalized:
            if not isinstance(argument, str):
                raise RepositoryCommandRejectedError(
                    "Every argv item must be a string."
                )
            if (
                not argument
                or len(argument) > self.MAX_ARGUMENT_CHARS
            ):
                raise RepositoryCommandRejectedError(
                    "Command arguments cannot be empty or oversized."
                )
            if any(
                character in argument
                for character in ("\0", "\n", "\r")
            ):
                raise RepositoryCommandRejectedError(
                    "Command arguments contain unsupported characters."
                )
        program = normalized[0]
        if program not in self._ALLOWED_PROGRAMS:
            raise RepositoryCommandRejectedError(
                f"Command is not allowed: {program}."
            )

        if program == "git":
            self._validate_git(normalized)
        elif program == "rg":
            self._validate_rg(normalized[1:])
        elif program == "tail":
            self._reject_options(
                normalized[1:],
                self._BLOCKED_TAIL_OPTIONS,
            )
            self._validate_file_operands(
                normalized[1:],
                value_options={"-c", "-n", "--bytes", "--lines"},
            )
        elif program == "head":
            self._validate_file_operands(
                normalized[1:],
                value_options={"-c", "-n", "--bytes", "--lines"},
            )
        elif program == "wc":
            self._reject_options(
                normalized[1:],
                ("--files0-from",),
            )
            self._validate_file_operands(
                normalized[1:],
                value_options=set(),
            )

        return normalized

    def _validate_rg(self, arguments: Sequence[str]) -> None:
        """允许搜索模式包含路径字符，但严格校验实际搜索路径。"""

        self._reject_options(
            arguments,
            self._BLOCKED_RG_OPTIONS,
        )
        consume_value = False
        consume_pattern = False
        pattern_supplied = False
        after_separator = False

        for argument in arguments:
            if consume_value:
                consume_value = False
                continue
            if consume_pattern:
                consume_pattern = False
                pattern_supplied = True
                continue
            if not after_separator and argument == "--":
                after_separator = True
                continue
            if (
                not after_separator
                and argument in self._RG_OPTIONS_WITH_VALUES
            ):
                consume_value = True
                continue
            if (
                not after_separator
                and argument in self._RG_EXPLICIT_PATTERN_OPTIONS
            ):
                consume_pattern = True
                continue
            if not after_separator and (
                argument.startswith("--regexp=")
                or (
                    argument.startswith("-e")
                    and argument != "-e"
                )
            ):
                pattern_supplied = True
                continue
            if (
                not after_separator
                and argument in self._RG_PATH_ONLY_MODES
            ):
                pattern_supplied = True
                continue
            if not after_separator and argument.startswith("-"):
                continue
            if not pattern_supplied:
                pattern_supplied = True
                continue
            self._require_repository_path(argument)

    def _validate_git(self, argv: Sequence[str]) -> None:
        """Git 只允许明确的只读子命令及无执行能力的参数。"""

        if len(argv) < 2:
            raise RepositoryCommandRejectedError(
                "git requires an allowed read-only subcommand."
            )
        subcommand = argv[1]
        if subcommand not in self._ALLOWED_GIT_SUBCOMMANDS:
            raise RepositoryCommandRejectedError(
                f"Git subcommand is not allowed: {subcommand}."
            )
        self._reject_options(
            argv[2:],
            self._BLOCKED_GIT_OPTIONS,
        )

    @staticmethod
    def _reject_options(
        arguments: Sequence[str],
        blocked_options: Sequence[str],
    ) -> None:
        """拒绝精确命中或以等号携带值的危险选项。"""

        for argument in arguments:
            for option in blocked_options:
                if (
                    argument == option
                    or argument.startswith(option + "=")
                    or (
                        option.startswith("-")
                        and not option.startswith("--")
                        and argument.startswith(option)
                    )
                ):
                    raise RepositoryCommandRejectedError(
                        f"Command option is not allowed: {option}."
                    )

    def _validate_file_operands(
        self,
        arguments: Sequence[str],
        value_options: Set[str],
    ) -> None:
        """确认文件工具的所有路径操作数仍位于仓库内。"""

        consume_as_value = False
        after_separator = False
        for argument in arguments:
            if consume_as_value:
                consume_as_value = False
                continue
            if not after_separator and argument == "--":
                after_separator = True
                continue
            if not after_separator and argument in value_options:
                consume_as_value = True
                continue
            if not after_separator and argument.startswith("-"):
                continue
            if not after_separator and argument.startswith("+"):
                continue
            if argument == "-":
                continue
            self._require_repository_path(argument)

    def _require_repository_path(self, value: str) -> None:
        """解析真实路径，阻止通过符号链接逃出仓库目录。"""

        candidate = (
            self._repository_dir / value
        ).resolve()
        try:
            candidate.relative_to(self._repository_dir)
        except ValueError as exc:
            raise RepositoryCommandRejectedError(
                "Command path must stay inside the repository."
            ) from exc

    @classmethod
    def _resolve_executables(
        cls,
        executable_paths: Optional[Mapping[str, str]],
    ) -> Dict[str, str]:
        """解析固定程序路径，避免运行时受仓库内容或 PATH 劫持。"""

        if executable_paths is None:
            return {
                program: executable
                for program in cls._ALLOWED_PROGRAMS
                for executable in [shutil.which(program)]
                if executable is not None
            }

        unknown = set(executable_paths) - cls._ALLOWED_PROGRAMS
        if unknown:
            raise ValueError(
                "Executable mapping contains unsupported programs: "
                + ", ".join(sorted(unknown))
            )

        resolved = {}
        for program, executable in executable_paths.items():
            path = Path(executable).resolve()
            if not path.is_file() or not os.access(path, os.X_OK):
                raise ValueError(
                    f"Executable is unavailable: {program}."
                )
            resolved[program] = str(path)
        return resolved

    @staticmethod
    def _minimal_environment() -> Dict[str, str]:
        """仅传递命令稳定运行所需变量，不泄露服务端 secrets。"""

        return {
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_EXTERNAL_DIFF": "/usr/bin/false",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_PAGER": "cat",
            "GIT_TERMINAL_PROMPT": "0",
            "PAGER": "cat",
        }

    @staticmethod
    async def _read_limited(
        stream: Optional[asyncio.StreamReader],
        budget: _OutputBudget,
    ) -> bytes:
        """持续排空进程输出，但只在内存中保留限制以内的数据。"""

        if stream is None:
            return b""

        chunks = []
        while True:
            chunk = await stream.read(8192)
            if not chunk:
                break
            retained = budget.retain(chunk)
            if retained:
                chunks.append(retained)

        return b"".join(chunks)

    @staticmethod
    async def _terminate_process(
        process: asyncio.subprocess.Process,
    ) -> None:
        """终止整个子进程组，避免超时后遗留后代进程。"""

        if process.returncode is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await process.wait()
