import asyncio
import os
import shutil
import subprocess
import tempfile
from pathlib import PurePosixPath
from typing import List, Optional, Tuple

from app.services.github_repository_source import (
    is_safe_analysis_path,
    normalize_public_github_url,
)


class RepositoryWorkspaceError(RuntimeError):
    """Repository workspace operation failed."""


class RepositoryAccessError(RepositoryWorkspaceError):
    """A repository path is unsafe or unavailable."""


class GitHubRepositoryWorkspace:
    """管理公开 GitHub 仓库的临时克隆及受限只读访问。"""

    def __init__(
    self,
    repository_url: str,
    clone_timeout_seconds: int = 120,
    max_workspace_mb: int = 200,
    max_file_bytes: int = 512 * 1024,
    max_read_lines: int = 400,
    max_read_chars: int = 32 * 1024,
    ) -> None:
        """初始化仓库地址、资源限制和隔离后的 Git 环境变量。"""

        self.repository_url = normalize_public_github_url(
            repository_url
        )
        self.clone_timeout_seconds = (
            clone_timeout_seconds
        )
        self.max_workspace_bytes = (
            max_workspace_mb * 1024 * 1024
        )
        if max_file_bytes < 1:
            raise ValueError(
                "max_file_bytes must be positive."
            )

        if max_read_lines < 1:
            raise ValueError(
                "max_read_lines must be positive."
            )

        if max_read_chars < 1:
            raise ValueError(
                "max_read_chars must be positive."
            )

        self.max_file_bytes = max_file_bytes
        self.max_read_lines = max_read_lines
        self.max_read_chars = max_read_chars

        self.commit_sha = ""
        self.default_branch = ""

        self._temporary_root: Optional[str] = None
        self._repository_dir: Optional[str] = None
        self._all_paths: Tuple[str, ...] = ()
        self._opened = False

        self._environment = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("GIT_")
        }
        self._environment.update(
            {
                "GIT_TERMINAL_PROMPT": "0",
                "GIT_LFS_SKIP_SMUDGE": "1",
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_GLOBAL": os.devnull,
                "GIT_ASKPASS": "/usr/bin/false",
            }
        )

    async def __aenter__(
        self,
    ) -> "GitHubRepositoryWorkspace":
        """克隆默认分支、记录提交信息并建立安全文件路径索引。"""

        if self._opened:
            raise RepositoryWorkspaceError(
                "Repository workspace is already open."
            )

        self._temporary_root = tempfile.mkdtemp(
            prefix="offerpilot-project-agent-"
        )
        self._repository_dir = os.path.join(
            self._temporary_root,
            "repository",
        )

        try:
            await self._run_command(
                [
                    "git",
                    "-c",
                    "protocol.file.allow=never",
                    "-c",
                    "credential.helper=",
                    "-c",
                    "core.askPass=/usr/bin/false",
                    "-c",
                    "http.extraHeader=",
                    "clone",
                    "--depth",
                    "1",
                    "--single-branch",
                    "--no-tags",
                    self.repository_url,
                    self._repository_dir,
                ],
                monitor_directory=self._temporary_root,
            )

            self.commit_sha = (
                await self._git(
                    "rev-parse",
                    "HEAD",
                )
            ).strip()

            branch = (
                await self._git(
                    "symbolic-ref",
                    "--short",
                    "HEAD",
                )
            ).strip()
            self.default_branch = branch.split(
                "/",
                1,
            )[-1]

            raw_paths = await self._git(
                "ls-files",
                "-z",
                max_output_bytes=10 * 1024 * 1024,
            )

            self._all_paths = tuple(
                sorted(
                    path
                    for path in raw_paths.split("\0")
                    if path
                    and is_safe_analysis_path(path)
                )
            )
            self._opened = True

            return self

        except Exception:
            await self.close()
            raise

    async def __aexit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:
        """退出异步上下文时清理临时仓库，无论分析是否成功。"""

        await self.close()

    @property
    def all_paths(self) -> Tuple[str, ...]:
        """返回当前仓库中允许 Agent 查看和读取的全部文件路径。"""

        self._require_open()
        return self._all_paths

    @property
    def repository_dir(self) -> str:
        """返回当前已打开临时仓库的根目录。"""

        self._require_open()
        if self._repository_dir is None:
            raise RepositoryWorkspaceError(
                "Repository workspace is not open."
            )
        return self._repository_dir

    def list_files(
        self,
        prefix: str = "",
        limit: int = 200,
    ) -> List[str]:
        """按可选目录前缀列出安全文件，并限制单次返回数量。"""

        self._require_open()

        if limit < 1 or limit > 1000:
            raise RepositoryAccessError(
                "File list limit must be between 1 and 1000."
            )

        normalized_prefix = self._normalize_prefix(
            prefix
        )

        if not normalized_prefix:
            matches = list(self._all_paths)
        else:
            directory_prefix = (
                normalized_prefix.rstrip("/") + "/"
            )
            matches = [
                path
                for path in self._all_paths
                if path == normalized_prefix
                or path.startswith(directory_prefix)
            ]

        return matches[:limit]


    async def read_file(
        self,
        path: str,
        start_line: int = 1,
        end_line: Optional[int] = None,
        start_column: int = 1,
    ) -> dict:
        """校验并读取指定文本文件的行范围，返回可序列化结果。"""

        self._require_open()

        normalized_path = self._normalize_file_path(
            path
        )

        if normalized_path not in self._all_paths:
            raise RepositoryAccessError(
                "Repository file does not exist "
                "or is blocked."
            )

        if start_line < 1:
            raise RepositoryAccessError(
                "start_line must be at least 1."
            )

        if start_column < 1:
            raise RepositoryAccessError(
                "start_column must be at least 1."
            )

        if (
            end_line is not None
            and end_line < start_line
        ):
            raise RepositoryAccessError(
                "end_line cannot be smaller "
                "than start_line."
            )

        size_text = await self._git(
            "cat-file",
            "-s",
            f"HEAD:{normalized_path}",
        )

        try:
            file_size = int(size_text.strip())
        except ValueError as exc:
            raise RepositoryAccessError(
                "Repository file size is invalid."
            ) from exc

        if file_size > self.max_file_bytes:
            raise RepositoryAccessError(
                "Repository file exceeds the "
                "size limit."
            )

        raw_content = await self._git_bytes(
            "show",
            f"HEAD:{normalized_path}",
            max_output_bytes=self.max_file_bytes,
        )

        if b"\0" in raw_content:
            raise RepositoryAccessError(
                "Binary repository files cannot be read."
            )

        try:
            text = raw_content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RepositoryAccessError(
                "Repository file is not valid UTF-8 text."
            ) from exc

        lines = text.splitlines()
        total_lines = len(lines)

        if total_lines == 0:
            return {
                "path": normalized_path,
                "content": "",
                "start_line": 1,
                "start_column": 1,
                "end_line": 0,
                "total_lines": 0,
                "truncated": False,
                "has_more": False,
                "next_start_line": None,
                "next_start_column": None,
            }

        if start_line > total_lines:
            raise RepositoryAccessError(
                "start_line exceeds the file length."
            )

        if start_column > len(lines[start_line - 1]) + 1:
            raise RepositoryAccessError(
                "start_column exceeds the line length."
            )

        requested_end = (
            end_line
            if end_line is not None
            else total_lines
        )

        line_limited_end = min(
            requested_end,
            total_lines,
            start_line + self.max_read_lines - 1,
        )

        candidate_lines = lines[
            start_line - 1:line_limited_end
        ]
        selected_lines = []
        selected_char_count = 0

        returned_line_was_cut = False
        for index, line in enumerate(candidate_lines):
            remaining_line = (
                line[start_column - 1:]
                if index == 0
                else line
            )
            separator_size = (
                1 if selected_lines else 0
            )
            available_chars = (
                self.max_read_chars
                - selected_char_count
                - separator_size
            )

            if available_chars <= 0:
                break

            selected_line = remaining_line[:available_chars]
            selected_lines.append(selected_line)
            selected_char_count += (
                separator_size
                + len(selected_line)
            )

            if len(selected_line) < len(remaining_line):
                returned_line_was_cut = True
                break

        effective_end = (
            start_line + len(selected_lines) - 1
        )

        has_more = (
            effective_end < requested_end
            and effective_end < total_lines
        ) or returned_line_was_cut
        if not has_more:
            next_start_line = None
            next_start_column = None
        elif returned_line_was_cut:
            next_start_line = effective_end
            line_start_column = (
                start_column if len(selected_lines) == 1 else 1
            )
            next_start_column = (
                line_start_column + len(selected_lines[-1])
            )
        else:
            next_start_line = effective_end + 1
            next_start_column = 1

        return {
            "path": normalized_path,
            "content": "\n".join(selected_lines),
            "start_line": start_line,
            "start_column": start_column,
            "end_line": effective_end,
            "total_lines": total_lines,
            "truncated": has_more,
            "has_more": has_more,
            "next_start_line": next_start_line,
            "next_start_column": next_start_column,
        }
    async def search_code(
        self,
        query: str,
        path_prefix: str = "",
        limit: int = 50,
    ) -> dict:
        """在安全文件范围内搜索固定字符串，并返回路径、行号和代码行。"""

        self._require_open()

        normalized_query = str(query or "").strip()

        if not normalized_query:
            raise RepositoryAccessError(
                "Search query cannot be empty."
            )

        if len(normalized_query) > 200:
            raise RepositoryAccessError(
                "Search query exceeds the length limit."
            )

        if (
            "\0" in normalized_query
            or "\n" in normalized_query
            or "\r" in normalized_query
        ):
            raise RepositoryAccessError(
                "Search query contains unsupported characters."
            )

        if limit < 1 or limit > 200:
            raise RepositoryAccessError(
                "Search result limit must be "
                "between 1 and 200."
            )

        normalized_prefix = self._normalize_prefix(
            path_prefix
        )

        arguments = [
            "grep",
            "-n",
            "-I",
            "-F",
            "--null",
            "-e",
            normalized_query,
            "--",
        ]

        if normalized_prefix:
            arguments.append(normalized_prefix)

        raw_output = await self._git_bytes(
            *arguments,
            max_output_bytes=2 * 1024 * 1024,
            allowed_return_codes=(0, 1),
        )

        matches = []
        truncated = False

        for raw_record in raw_output.splitlines():
            parts = raw_record.split(b"\0", 2)

            if len(parts) != 3:
                continue

            raw_path, raw_line_number, raw_line = parts

            try:
                path = raw_path.decode("utf-8")
                line_number = int(
                    raw_line_number.decode("ascii")
                )
                line = raw_line.decode(
                    "utf-8",
                    errors="replace",
                )
            except (UnicodeDecodeError, ValueError):
                continue

            if path not in self._all_paths:
                continue

            matches.append(
                {
                    "path": path,
                    "line_number": line_number,
                    "line": line[:1000],
                }
            )

            if len(matches) > limit:
                truncated = True
                break

        return {
            "query": normalized_query,
            "path_prefix": normalized_prefix,
            "matches": matches[:limit],
            "truncated": truncated,
        }

    async def locate_exact_quote(
        self,
        path: str,
        quote: str,
        near_line: int = 1,
    ) -> Optional[Tuple[int, int]]:
        """在同一安全文件中精确定位原文，并优先选择靠近声明行的位置。"""

        self._require_open()
        normalized_path = self._normalize_file_path(path)
        if normalized_path not in self._all_paths:
            raise RepositoryAccessError(
                "Repository file does not exist or is blocked."
            )
        if not quote:
            raise RepositoryAccessError("quote cannot be empty.")

        size_text = await self._git(
            "cat-file",
            "-s",
            f"HEAD:{normalized_path}",
        )
        try:
            file_size = int(size_text.strip())
        except ValueError as exc:
            raise RepositoryAccessError(
                "Repository file size is invalid."
            ) from exc
        if file_size > self.max_file_bytes:
            raise RepositoryAccessError(
                "Repository file exceeds the size limit."
            )

        raw_content = await self._git_bytes(
            "show",
            f"HEAD:{normalized_path}",
            max_output_bytes=self.max_file_bytes,
        )
        if b"\0" in raw_content:
            raise RepositoryAccessError(
                "Binary repository files cannot be read."
            )
        try:
            text = raw_content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RepositoryAccessError(
                "Repository file is not valid UTF-8 text."
            ) from exc

        occurrences = []
        offset = 0
        while True:
            position = text.find(quote, offset)
            if position < 0:
                break
            start_line = text.count("\n", 0, position) + 1
            end_line = start_line + quote.count("\n")
            occurrences.append((start_line, end_line))
            offset = position + max(1, len(quote))
        if not occurrences:
            return None
        return min(
            occurrences,
            key=lambda lines: abs(lines[0] - near_line),
        )

    async def close(self) -> None:
        """重置 Workspace 状态并从磁盘删除临时仓库目录。"""

        temporary_root = self._temporary_root

        self._temporary_root = None
        self._repository_dir = None
        self._all_paths = ()
        self._opened = False

        if temporary_root:
            await asyncio.to_thread(
                shutil.rmtree,
                temporary_root,
                True,
            )


    async def _git(
        self,
        *arguments: str,
        max_output_bytes: int = 1024 * 1024,
    ) -> str:
        """执行受控 Git 命令，并把输出解码为文本。"""

        raw = await self._git_bytes(
            *arguments,
            max_output_bytes=max_output_bytes,
        )

        return raw.decode(
            "utf-8",
            errors="replace",
        )

    async def _git_bytes(
        self,
        *arguments: str,
        max_output_bytes: int = 1024 * 1024,
        allowed_return_codes: Tuple[int, ...] = (0,),
    ) -> bytes:
        """在当前仓库中执行受控 Git 命令并返回原始字节。"""

        if not self._repository_dir:
            raise RepositoryWorkspaceError(
                "Repository workspace is not open."
            )

        return await self._run_command(
            [
                "git",
                "-C",
                self._repository_dir,
                *arguments,
            ],
            max_output_bytes=max_output_bytes,
            allowed_return_codes=allowed_return_codes,
        )
    async def _run_command(
        self,
        command: List[str],
        monitor_directory: Optional[str] = None,
        max_output_bytes: int = 1024 * 1024,
        allowed_return_codes: Tuple[int, ...] = (0,),
    ) -> bytes:
        """运行子进程，并统一处理超时、失败、输出和目录大小限制。"""

        process = await asyncio.create_subprocess_exec(
            *command,
            env=self._environment,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        monitor_task = None
        if monitor_directory:
            monitor_task = asyncio.create_task(
                self._monitor_workspace_size(
                    process,
                    monitor_directory,
                )
            )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.clone_timeout_seconds,
            )

            if (
                monitor_task is not None
                and monitor_task.done()
                and monitor_task.exception() is not None
            ):
                raise monitor_task.exception()

            if process.returncode not in allowed_return_codes:
                error_text = stderr.decode(
                    "utf-8",
                    errors="replace",
                ).strip()

                raise RepositoryWorkspaceError(
                    "Git command failed: "
                    f"{error_text[:300]}"
                )

            if len(stdout) > max_output_bytes:
                raise RepositoryWorkspaceError(
                    "Git command output exceeded the limit."
                )

            return stdout

        except asyncio.TimeoutError as exc:
            if process.returncode is None:
                process.kill()
                await process.wait()

            raise RepositoryWorkspaceError(
                "Git command timed out."
            ) from exc

        except asyncio.CancelledError:
            if process.returncode is None:
                process.kill()
                await process.wait()
            raise

        finally:
            if monitor_task is not None:
                monitor_task.cancel()
                await asyncio.gather(
                    monitor_task,
                    return_exceptions=True,
                )

    async def _monitor_workspace_size(
        self,
        process,
        directory: str,
    ) -> None:
        """在 Git 进程运行期间监控临时目录，超限时终止进程。"""

        while process.returncode is None:
            size = await asyncio.to_thread(
                self._directory_size,
                directory,
            )

            if size > self.max_workspace_bytes:
                process.kill()
                await process.wait()

                raise RepositoryWorkspaceError(
                    "Repository workspace exceeded "
                    "the size limit."
                )

            await asyncio.sleep(0.1)

    def _require_open(self) -> None:
        """确保调用发生在已成功打开的 Workspace 生命周期内。"""

        if not self._opened:
            raise RepositoryWorkspaceError(
                "Repository workspace is not open."
            )
    @staticmethod
    def _normalize_file_path(path: str) -> str:
        """规范化仓库文件路径，并拒绝绝对路径和目录穿越。"""

        value = str(path or "").strip()

        if not value:
            raise RepositoryAccessError(
                "Repository path cannot be empty."
            )

        if value.startswith("/") or "\\" in value:
            raise RepositoryAccessError(
                "Repository path must be relative "
                "and use forward slashes."
            )

        normalized = PurePosixPath(value)

        if (
            normalized.is_absolute()
            or ".." in normalized.parts
        ):
            raise RepositoryAccessError(
                "Repository path is unsafe."
            )

        result = normalized.as_posix()

        if result in {"", "."}:
            raise RepositoryAccessError(
                "Repository path must point to a file."
            )

        return result

    @staticmethod
    def _normalize_prefix(prefix: str) -> str:
        """规范化文件列表前缀，并拒绝不安全的路径表达。"""

        value = str(prefix or "").strip()

        if not value:
            return ""

        if value.startswith("/") or "\\" in value:
            raise RepositoryAccessError(
                "Repository prefix must be relative "
                "and use forward slashes."
            )

        normalized = PurePosixPath(value)

        if (
            normalized.is_absolute()
            or ".." in normalized.parts
        ):
            raise RepositoryAccessError(
                "Repository prefix is unsafe."
            )

        result = normalized.as_posix()

        if result == ".":
            return ""

        return result.rstrip("/")

    @staticmethod
    def _directory_size(directory: str) -> int:
        """递归计算目录当前占用的总字节数，忽略瞬时文件错误。"""

        total = 0

        for root, _, files in os.walk(directory):
            for filename in files:
                path = os.path.join(
                    root,
                    filename,
                )

                try:
                    total += os.path.getsize(path)
                except OSError:
                    continue

        return total
