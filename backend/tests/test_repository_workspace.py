import asyncio

import pytest

from app.project_analysis.repository_workspace import (
    GitHubRepositoryWorkspace,
    RepositoryAccessError,
    RepositoryWorkspaceError,
)


def _opened_workspace(
    *,
    max_read_lines: int = 400,
    max_read_chars: int = 32 * 1024,
) -> GitHubRepositoryWorkspace:
    """创建无需访问网络的已打开 Workspace 测试替身。"""

    workspace = GitHubRepositoryWorkspace(
        "https://github.com/example/project",
        max_read_lines=max_read_lines,
        max_read_chars=max_read_chars,
    )
    workspace._opened = True
    workspace._repository_dir = "/tmp/fake-repository"
    workspace._all_paths = (
        "README.md",
        "src/app.py",
        "src/service.py",
    )
    return workspace


def test_list_files_filters_by_safe_prefix_and_limit() -> None:
    """验证文件列表支持安全前缀过滤和数量限制。"""

    workspace = _opened_workspace()

    assert workspace.list_files(
        prefix="src",
        limit=1,
    ) == ["src/app.py"]

    with pytest.raises(RepositoryAccessError):
        workspace.list_files(prefix="../outside")


def test_repository_dir_is_exposed_only_while_workspace_is_open() -> None:
    """命令执行器只能取得当前已打开临时仓库的根目录。"""

    workspace = _opened_workspace()

    assert workspace.repository_dir == "/tmp/fake-repository"

    workspace._opened = False
    with pytest.raises(RepositoryWorkspaceError):
        _ = workspace.repository_dir


def test_read_file_returns_requested_lines_and_truncation() -> None:
    """验证文本读取会返回正确行范围并标记自动截断。"""

    workspace = _opened_workspace(max_read_lines=2)
    raw_content = b"line 1\nline 2\nline 3\nline 4\n"

    async def fake_git(
        *arguments,
        max_output_bytes=1024 * 1024,
    ):
        """模拟 Git 返回文件字节大小。"""

        assert arguments == (
            "cat-file",
            "-s",
            "HEAD:README.md",
        )
        return str(len(raw_content))

    async def fake_git_bytes(
        *arguments,
        max_output_bytes=1024 * 1024,
        allowed_return_codes=(0,),
    ):
        """模拟 Git 返回文件原始内容。"""

        assert arguments == (
            "show",
            "HEAD:README.md",
        )
        return raw_content

    workspace._git = fake_git
    workspace._git_bytes = fake_git_bytes

    result = asyncio.run(
        workspace.read_file("README.md")
    )

    assert result == {
        "path": "README.md",
        "content": "line 1\nline 2",
        "start_line": 1,
        "start_column": 1,
        "end_line": 2,
        "total_lines": 4,
        "truncated": True,
        "has_more": True,
        "next_start_line": 3,
        "next_start_column": 1,
    }


def test_read_file_reports_no_more_content_for_final_page() -> None:
    """从中间开始读取并到达文件末尾时不得诱导模型继续分页。"""

    workspace = _opened_workspace(max_read_lines=2)
    raw_content = b"line 1\nline 2\nline 3\nline 4\n"

    async def fake_git(*arguments, max_output_bytes=1024 * 1024):
        return str(len(raw_content))

    async def fake_git_bytes(
        *arguments,
        max_output_bytes=1024 * 1024,
        allowed_return_codes=(0,),
    ):
        return raw_content

    workspace._git = fake_git
    workspace._git_bytes = fake_git_bytes

    result = asyncio.run(
        workspace.read_file(
            "README.md",
            start_line=3,
        )
    )

    assert result["content"] == "line 3\nline 4"
    assert result["has_more"] is False
    assert result["next_start_line"] is None
    assert result["next_start_column"] is None
    assert result["truncated"] is False


def test_read_file_blocks_path_traversal_and_binary_content() -> None:
    """验证文件读取拒绝目录穿越和二进制内容。"""

    workspace = _opened_workspace()

    with pytest.raises(RepositoryAccessError):
        asyncio.run(
            workspace.read_file("../secret.txt")
        )

    raw_content = b"hello\0world"

    async def fake_git(
        *arguments,
        max_output_bytes=1024 * 1024,
    ):
        """模拟二进制文件的字节大小。"""

        return str(len(raw_content))

    async def fake_git_bytes(
        *arguments,
        max_output_bytes=1024 * 1024,
        allowed_return_codes=(0,),
    ):
        """模拟包含 NUL 字节的二进制文件内容。"""

        return raw_content

    workspace._git = fake_git
    workspace._git_bytes = fake_git_bytes

    with pytest.raises(
        RepositoryAccessError,
        match="Binary",
    ):
        asyncio.run(
            workspace.read_file("README.md")
        )


def test_read_file_limits_returned_characters_on_long_files() -> None:
    """验证代码读取按字符预算截断，避免工具结果撑大上下文。"""

    workspace = _opened_workspace(
        max_read_lines=20,
        max_read_chars=11,
    )
    raw_content = b"12345\n67890\nabcde\n"

    async def fake_git(
        *arguments,
        max_output_bytes=1024 * 1024,
    ):
        """模拟 Git 返回文件字节大小。"""

        return str(len(raw_content))

    async def fake_git_bytes(
        *arguments,
        max_output_bytes=1024 * 1024,
        allowed_return_codes=(0,),
    ):
        """模拟 Git 返回超过单次工具字符预算的文本。"""

        return raw_content

    workspace._git = fake_git
    workspace._git_bytes = fake_git_bytes

    result = asyncio.run(
        workspace.read_file("README.md")
    )

    assert result["content"] == "12345\n67890"
    assert result["end_line"] == 2
    assert result["truncated"] is True
    assert result["has_more"] is True
    assert result["next_start_line"] == 3
    assert result["next_start_column"] == 1


def test_read_file_resumes_character_truncated_line_without_skipping() -> None:
    """超长单行使用行列游标续读时不得遗漏中间字符。"""

    workspace = _opened_workspace(max_read_chars=5)
    raw_content = b"abcdefghijkl\nnext\n"

    async def fake_git(*arguments, max_output_bytes=1024 * 1024):
        return str(len(raw_content))

    async def fake_git_bytes(
        *arguments,
        max_output_bytes=1024 * 1024,
        allowed_return_codes=(0,),
    ):
        return raw_content

    workspace._git = fake_git
    workspace._git_bytes = fake_git_bytes

    first = asyncio.run(workspace.read_file("README.md"))
    second = asyncio.run(
        workspace.read_file(
            "README.md",
            start_line=first["next_start_line"],
            start_column=first["next_start_column"],
        )
    )
    third = asyncio.run(
        workspace.read_file(
            "README.md",
            start_line=second["next_start_line"],
            start_column=second["next_start_column"],
        )
    )

    assert first["content"] == "abcde"
    assert (first["next_start_line"], first["next_start_column"]) == (1, 6)
    assert second["content"] == "fghij"
    assert (second["next_start_line"], second["next_start_column"]) == (1, 11)
    assert third["content"] == "kl\nne"


def test_search_code_parses_matches_filters_paths_and_truncates() -> None:
    """验证代码搜索解析结果、过滤屏蔽路径并限制返回数量。"""

    workspace = _opened_workspace()
    captured = {}

    async def fake_git_bytes(
        *arguments,
        max_output_bytes=1024 * 1024,
        allowed_return_codes=(0,),
    ):
        """模拟 git grep 的 NUL 分隔输出。"""

        captured["arguments"] = arguments
        captured["max_output_bytes"] = max_output_bytes
        captured["allowed_return_codes"] = (
            allowed_return_codes
        )
        return (
            b"src/app.py\x0012\x00class ProjectService:\n"
            b".env\x001\x00API_KEY=secret\n"
            b"src/service.py\x0040\x00service = ProjectService()\n"
        )

    workspace._git_bytes = fake_git_bytes

    result = asyncio.run(
        workspace.search_code(
            "ProjectService",
            path_prefix="src",
            limit=1,
        )
    )

    assert captured["arguments"] == (
        "grep",
        "-n",
        "-I",
        "-F",
        "--null",
        "-e",
        "ProjectService",
        "--",
        "src",
    )
    assert captured["allowed_return_codes"] == (0, 1)
    assert result == {
        "query": "ProjectService",
        "path_prefix": "src",
        "matches": [
            {
                "path": "src/app.py",
                "line_number": 12,
                "line": "class ProjectService:",
            }
        ],
        "truncated": True,
    }


def test_search_code_returns_empty_result() -> None:
    """验证 git grep 没有匹配时返回空列表而不是抛错。"""

    workspace = _opened_workspace()

    async def fake_git_bytes(
        *arguments,
        max_output_bytes=1024 * 1024,
        allowed_return_codes=(0,),
    ):
        """模拟 git grep 没有找到匹配。"""

        assert allowed_return_codes == (0, 1)
        return b""

    workspace._git_bytes = fake_git_bytes

    result = asyncio.run(
        workspace.search_code("MissingSymbol")
    )

    assert result["matches"] == []
    assert result["truncated"] is False
