import asyncio
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Dict, List, Optional
from urllib.parse import urlsplit


MAX_ANALYSIS_FILES = 30
MAX_ANALYSIS_BYTES = 180 * 1024
MAX_FILE_BYTES = 24 * 1024


class GitRepositoryError(RuntimeError):
    pass


@dataclass(frozen=True)
class RepositoryFile:
    path: str
    content: str


@dataclass(frozen=True)
class RepositorySnapshot:
    repository_url: str
    commit_sha: str
    default_branch: str
    files: Dict[str, RepositoryFile]
    all_paths: List[str]
    metadata_bytes: int


def normalize_public_github_url(raw_url: str) -> str:
    value = str(raw_url or "").strip()
    parsed = urlsplit(value)
    if parsed.scheme != "https" or parsed.hostname is None:
        raise ValueError("只支持 https://github.com/owner/repository 公开仓库地址。")
    if parsed.hostname.lower() != "github.com" or parsed.port is not None:
        raise ValueError("只支持 github.com 的公开仓库。")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("GitHub 地址不能包含凭据、查询参数或锚点。")
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) != 2:
        raise ValueError("请填写仓库根地址，不支持分支或子目录地址。")
    owner, repo = parts
    if repo.endswith(".git"):
        repo = repo[:-4]
    component = re.compile(r"^[A-Za-z0-9_.-]+$")
    if not owner or not repo or not component.fullmatch(owner) or not component.fullmatch(repo):
        raise ValueError("GitHub 仓库地址格式无效。")
    return f"https://github.com/{owner}/{repo}"


class GitHubRepositorySource:
    def __init__(self, clone_timeout_seconds: int = 120,
                 max_git_metadata_mb: int = 100):
        self.clone_timeout_seconds = clone_timeout_seconds
        self.max_git_metadata_bytes = max_git_metadata_mb * 1024 * 1024

    async def fetch(self, repository_url: str,
                    selected_paths: Optional[List[str]] = None) -> RepositorySnapshot:
        return await self._fetch(repository_url, selected_paths)

    async def _fetch(self, repository_url, selected_paths):
        canonical = normalize_public_github_url(repository_url)
        temporary_root = tempfile.mkdtemp(prefix="offerpilot-discovery-")
        repository_dir = os.path.join(temporary_root, "repository")
        environment = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("GIT_")
        }
        environment.update({
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_LFS_SKIP_SMUDGE": "1",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_ASKPASS": "/usr/bin/false",
        })
        try:
            await self._git(
                [
                    "git",
                    "-c", "protocol.file.allow=never",
                    "-c", "credential.helper=",
                    "-c", "core.askPass=/usr/bin/false",
                    "-c", "http.extraHeader=",
                    "clone",
                    "--depth", "1",
                    "--filter=blob:none",
                    "--no-checkout",
                    "--no-tags",
                    canonical,
                    repository_dir,
                ],
                environment,
                monitor_directory=os.path.join(repository_dir, ".git"),
            )
            metadata_bytes = _directory_size(os.path.join(repository_dir, ".git"))
            if metadata_bytes > self.max_git_metadata_bytes:
                raise GitRepositoryError("Git 元数据超过允许的大小。")
            commit_sha = (
                await self._git(
                    ["git", "-C", repository_dir, "rev-parse", "HEAD"],
                    environment,
                )
            ).strip()
            branch_ref = (
                await self._git(
                    [
                        "git", "-C", repository_dir, "symbolic-ref",
                        "--short", "HEAD",
                    ],
                    environment,
                )
            ).strip()
            default_branch = branch_ref.split("/", 1)[-1]
            tree = await self._git(
                [
                    "git", "-C", repository_dir, "ls-tree", "-r",
                    "--name-only", "-z", "HEAD",
                ],
                environment,
                max_output_bytes=10 * 1024 * 1024,
            )
            all_paths = [item for item in tree.split("\0") if item]
            requested = selected_paths if selected_paths is not None else _initial_inventory_paths(all_paths)
            files = await self._read_selected(
                repository_dir, environment, all_paths, requested
            )
            return RepositorySnapshot(canonical, commit_sha, default_branch, files, all_paths, metadata_bytes)
        except subprocess.TimeoutExpired as exc:
            raise GitRepositoryError("克隆公开仓库超时，请稍后重试。") from exc
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr or b""
            message = (
                stderr.decode("utf-8", errors="replace")
                if isinstance(stderr, bytes)
                else str(stderr)
            ).strip()
            raise GitRepositoryError("无法读取公开仓库，请确认地址存在且仓库公开。" + (f" ({message[:160]})" if message else "")) from exc
        finally:
            shutil.rmtree(temporary_root, ignore_errors=True)

    async def _read_selected(
        self, repository_dir, environment, all_paths, requested
    ):
        allowed = set(all_paths)
        result: Dict[str, RepositoryFile] = {}
        total = 0
        for path in requested:
            if path not in allowed or not is_safe_analysis_path(path) or len(result) >= MAX_ANALYSIS_FILES:
                continue
            size_text = await self._git(
                [
                    "git", "-C", repository_dir, "cat-file", "-s",
                    f"HEAD:{path}",
                ],
                environment,
            )
            try:
                size = int(size_text.strip())
            except ValueError:
                continue
            if size > MAX_FILE_BYTES or total + size > MAX_ANALYSIS_BYTES:
                continue
            raw = await self._git_bytes(
                ["git", "-C", repository_dir, "show", f"HEAD:{path}"],
                environment,
                max_output_bytes=MAX_FILE_BYTES,
            )
            if b"\0" in raw or _contains_secret(raw) or _contains_prompt_injection(raw):
                continue
            try:
                content = raw.decode("utf-8")
            except UnicodeDecodeError:
                continue
            result[path] = RepositoryFile(path, content)
            total += len(raw)
        return result

    async def _git(self, command, environment, **kwargs):
        value = await self._git_bytes(command, environment, **kwargs)
        return value.decode("utf-8", errors="replace")

    async def _git_bytes(
        self,
        command,
        environment,
        monitor_directory=None,
        max_output_bytes=None,
    ):
        output_file = tempfile.TemporaryFile() if max_output_bytes else None
        process = await asyncio.create_subprocess_exec(
            *command,
            env=environment,
            stdout=output_file if output_file else asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        monitor = asyncio.create_task(
            _monitor_process_limits(
                process,
                monitor_directory=monitor_directory,
                max_directory_bytes=self.max_git_metadata_bytes,
                output_file=output_file,
                max_output_bytes=max_output_bytes,
            )
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=self.clone_timeout_seconds
            )
            if monitor.done() and monitor.exception() is not None:
                raise monitor.exception()
            if process.returncode:
                raise subprocess.CalledProcessError(
                    process.returncode, command, stderr=stderr
                )
            if output_file is not None:
                output_file.seek(0)
                value = output_file.read(max_output_bytes + 1)
                if len(value) > max_output_bytes:
                    raise GitRepositoryError("仓库文件树超过允许的大小。")
                return value
            return stdout
        except asyncio.TimeoutError as exc:
            if process.returncode is None:
                process.kill()
                await process.wait()
            raise subprocess.TimeoutExpired(
                command, self.clone_timeout_seconds
            ) from exc
        except asyncio.CancelledError:
            if process.returncode is None:
                process.kill()
                await process.wait()
            raise
        finally:
            monitor.cancel()
            await asyncio.gather(monitor, return_exceptions=True)
            if output_file is not None:
                output_file.close()


async def _monitor_process_limits(
    process,
    monitor_directory,
    max_directory_bytes,
    output_file,
    max_output_bytes,
):
    while process.returncode is None:
        if (
            monitor_directory
            and _directory_size(monitor_directory) > max_directory_bytes
        ):
            process.kill()
            await process.wait()
            raise GitRepositoryError("Git 元数据超过允许的大小。")
        if output_file is not None and output_file.tell() > max_output_bytes:
            process.kill()
            await process.wait()
            raise GitRepositoryError("仓库文件树超过允许的大小。")
        await asyncio.sleep(0.05)


_BLOCKED_NAMES = {
    ".env", ".npmrc", ".pypirc", "credentials", "credentials.json",
    "id_rsa", "id_ed25519", "secrets.yml", "secrets.yaml",
}
_BLOCKED_EXTENSIONS = {
    ".pem", ".key", ".p12", ".pfx", ".cer", ".crt", ".db", ".sqlite",
    ".sqlite3", ".csv", ".tsv", ".jsonl", ".zip", ".tar", ".gz", ".7z",
    ".rar", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".mp3", ".mp4",
    ".mov", ".avi", ".pdf", ".woff", ".woff2", ".class", ".jar", ".exe",
}
_BLOCKED_DIRS = {"node_modules", "vendor", ".venv", "venv", "dist", "build",
                 "target", ".next", "coverage", "__pycache__", ".git"}


def is_safe_analysis_path(path: str) -> bool:
    pure = PurePosixPath(path)
    lowered_parts = [part.lower() for part in pure.parts]
    name = pure.name.lower()
    if any(part in _BLOCKED_DIRS for part in lowered_parts):
        return False
    if name in _BLOCKED_NAMES or name.startswith(".env."):
        return False
    if pure.suffix.lower() in _BLOCKED_EXTENSIONS:
        return False
    return not any(token in name for token in ("secret", "credential", "private_key"))


def _contains_secret(raw: bytes) -> bool:
    text = raw.decode("utf-8", errors="ignore")
    patterns = (
        r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
        r"gh[opsu]_[A-Za-z0-9]{30,}",
        r"github_pat_[A-Za-z0-9_]{30,}",
        r"AKIA[0-9A-Z]{16}",
        r"xox[baprs]-[A-Za-z0-9-]{20,}",
        r"sk_(?:live|test)_[A-Za-z0-9]{16,}",
        r"(?i)(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?)://[^\s:@]+:[^\s@]+@",
        r"eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{10,}",
        r"(?i)(?:api[_-]?key|client[_-]?secret|password)\s*[:=]\s*['\"][^'\"]{12,}['\"]",
    )
    return any(re.search(pattern, text) for pattern in patterns)


def _contains_prompt_injection(raw: bytes) -> bool:
    text = raw.decode("utf-8", errors="ignore")[:100000].lower()
    suspicious = (
        "ignore previous instructions", "ignore all previous instructions",
        "disregard previous instructions", "reveal the system prompt",
        "you are chatgpt", "assistant: execute", "system: execute",
        "忽略之前的指令", "忽略以上指令", "泄露系统提示词",
    )
    return any(marker in text for marker in suspicious)


def _initial_inventory_paths(paths):
    return sorted(
        (path for path in paths if is_safe_analysis_path(path)),
        key=analysis_path_score,
        reverse=True,
    )[:MAX_ANALYSIS_FILES]


def analysis_path_score(path):
    lowered = path.lower()
    name = PurePosixPath(path).name.lower()
    score = 0
    if name.startswith("readme"):
        score += 100
    if name in {
        "pyproject.toml", "package.json", "go.mod", "pom.xml", "cargo.toml",
        "requirements.txt",
    }:
        score += 90
    if name in {"main.py", "app.py", "server.py", "index.ts", "index.js"}:
        score += 80
    if any(
        token in lowered
        for token in ("domain", "service", "controller", "repository", "model")
    ):
        score += 60
    if "test" in lowered:
        score += 45
    if any(token in lowered for token in ("docker", "deploy", "k8s", "workflow")):
        score += 40
    return score - len(PurePosixPath(path).parts)


def _directory_size(path):
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total
