import asyncio
import json

import pytest

from app.project_analysis.repository_analysis_result import (
    RepositoryAnalysisResultParser,
    RepositoryAnalysisValidationError,
)
from app.project_analysis.repository_workspace import (
    RepositoryAccessError,
)


class FakeWorkspace:
    """提供确定文件内容并记录读取范围的 Workspace 测试替身。"""

    repository_url = "https://github.com/example/offerpilot"

    def __init__(self) -> None:
        """初始化测试文件和读取调用记录。"""

        self.files = {
            "app/main.py": (
                "from fastapi import FastAPI\n"
                "app = FastAPI()\n"
                "\n"
                "@app.get('/health')\n"
                "def health():\n"
                "    return {'status': 'ok'}"
            ),
            "app/service.py": (
                "class ProjectService:\n"
                "    def analyze(self):\n"
                "        return 'complete'"
            ),
        }
        self.calls = []

    async def read_file(
        self,
        path,
        start_line=1,
        end_line=None,
    ):
        """返回指定文件行范围，并对不存在的路径模拟安全拒绝。"""

        self.calls.append(
            (path, start_line, end_line)
        )

        content = self.files.get(path)

        if content is None:
            raise RepositoryAccessError(
                "Repository file does not exist."
            )

        lines = content.splitlines()
        requested_end = end_line or len(lines)
        selected = lines[
            start_line - 1:requested_end
        ]

        return {
            "path": path,
            "content": "\n".join(selected),
            "start_line": start_line,
            "end_line": requested_end,
            "total_lines": len(lines),
            "truncated": requested_end < len(lines),
        }

    async def locate_exact_quote(self, path, quote, near_line=1):
        """在测试文件中精确定位 quote 的真实行范围。"""

        content = self.files.get(path)
        if content is None:
            raise RepositoryAccessError("Repository file does not exist.")
        position = content.find(quote)
        if position < 0:
            return None
        start_line = content.count("\n", 0, position) + 1
        return start_line, start_line + quote.count("\n")


def _finding(
    *,
    finding_id="F1",
    claim="项目使用 FastAPI 提供 HTTP 接口。",
    target_field="tech_stack",
    path="app/main.py",
    start_line=1,
    end_line=2,
    quote="from fastapi import FastAPI",
    confidence=0.9,
):
    """创建可以按测试场景覆盖字段的 Finding 字典。"""

    return {
        "id": finding_id,
        "claim": claim,
        "topic": "architecture",
        "target_field": target_field,
        "path": path,
        "start_line": start_line,
        "end_line": end_line,
        "quote": quote,
        "confidence": confidence,
    }


def _payload(
    findings,
    *,
    evidence_by_field=None,
    warnings=None,
    project_profile=None,
):
    """把测试数据编码成模型最终返回的 JSON 文本。"""

    value = {
        "project_profile": project_profile or {},
        "findings": findings,
        "warnings": warnings or [],
    }

    if evidence_by_field is not None:
        value["evidence_by_field"] = (
            evidence_by_field
        )

    return json.dumps(
        value,
        ensure_ascii=False,
    )


def test_parser_keeps_grounded_findings_and_builds_draft() -> None:
    """验证真实路径、行号和原文能够生成安全档案草稿。"""

    workspace = FakeWorkspace()
    parser = RepositoryAnalysisResultParser()
    findings = [
        _finding(),
        _finding(
            finding_id="F2",
            claim=(
                "项目暴露了健康检查接口。"
            ),
            target_field="architecture",
            start_line=4,
            end_line=6,
            quote="@app.get('/health')",
        ),
    ]

    result = asyncio.run(
        parser.parse(
            _payload(
                findings,
                evidence_by_field={
                    "tech_stack": ["F1"],
                    "architecture": ["F2"],
                },
            ),
            workspace,
        )
    )

    assert result.draft["name"] == "offerpilot"
    assert result.draft["tech_stack"] == [
        "项目使用 FastAPI 提供 HTTP 接口。"
    ]
    assert result.draft["architecture"] == (
        "项目暴露了健康检查接口。"
    )
    assert result.draft["responsibilities"] == []
    assert result.draft["metrics"] == []
    assert result.evidence_by_field == {
        "tech_stack": ["F1"],
        "architecture": ["F2"],
    }
    assert workspace.calls == [
        ("app/main.py", 1, 2),
        ("app/main.py", 4, 6),
    ]


def test_parser_accepts_json_wrapped_in_markdown_fence() -> None:
    """模型偶尔添加 Markdown 围栏时仍能解析唯一的分析对象。"""

    parser = RepositoryAnalysisResultParser()
    workspace = FakeWorkspace()
    wrapped = (
        "```json\n"
        + _payload([_finding()])
        + "\n```"
    )

    result = asyncio.run(
        parser.parse(wrapped, workspace)
    )

    assert [
        finding["id"]
        for finding in result.findings
    ] == ["F1"]


def test_parser_accepts_one_analysis_object_surrounded_by_text() -> None:
    """最终文本夹带说明时只提取唯一的分析 JSON 对象。"""

    parser = RepositoryAnalysisResultParser()
    workspace = FakeWorkspace()
    wrapped = (
        "分析结果如下：\n"
        + _payload([_finding()])
        + "\n以上为代码分析结果。"
    )

    result = asyncio.run(
        parser.parse(wrapped, workspace)
    )

    assert result.findings[0]["id"] == "F1"


@pytest.mark.parametrize(
    "raw_content, expected_message",
    [
        (
            "not-json",
            r"line 1, column 1: Expecting value",
        ),
        ("[]", "must be a JSON object"),
        ("{}", "findings must be a list"),
    ],
)
def test_parser_rejects_invalid_top_level_results(
    raw_content,
    expected_message,
) -> None:
    """验证非法 JSON 和错误顶层结构会立即终止解析。"""

    parser = RepositoryAnalysisResultParser()

    with pytest.raises(
        RepositoryAnalysisValidationError,
        match=expected_message,
    ):
        asyncio.run(
            parser.parse(
                raw_content,
                FakeWorkspace(),
            )
        )


def test_parser_rejects_ambiguous_multiple_analysis_objects() -> None:
    """存在多个分析对象时拒绝猜测模型真正想返回哪一个。"""

    parser = RepositoryAnalysisResultParser()
    content = (
        _payload([_finding()])
        + "\n"
        + _payload([_finding(finding_id="F2")])
    )

    with pytest.raises(
        RepositoryAnalysisValidationError,
        match="multiple JSON objects",
    ):
        asyncio.run(
            parser.parse(content, FakeWorkspace())
        )


def test_parser_reports_embedded_json_syntax_location() -> None:
    """说明文字后的畸形 JSON 应报告 JSON 内真正失败的位置。"""

    parser = RepositoryAnalysisResultParser()

    with pytest.raises(
        RepositoryAnalysisValidationError,
        match=(
            r"line 2, column 15: "
            r"Expecting value"
        ),
    ):
        asyncio.run(
            parser.parse(
                '分析结果：\n{"findings": [}',
                FakeWorkspace(),
            )
        )


def test_parser_discards_fake_path_and_wrong_quote() -> None:
    """验证伪造路径和不匹配原文会被丢弃并产生警告。"""

    parser = RepositoryAnalysisResultParser()
    findings = [
        _finding(),
        _finding(
            finding_id="F2",
            path="missing.py",
        ),
        _finding(
            finding_id="F3",
            quote="Django application",
        ),
    ]

    result = asyncio.run(
        parser.parse(
            _payload(findings),
            FakeWorkspace(),
        )
    )

    assert [
        finding["id"]
        for finding in result.findings
    ] == ["F1"]
    assert any(
        "无法读取证据" in warning
        for warning in result.warnings
    )
    assert any(
        "原文不匹配" in warning
        for warning in result.warnings
    )


def test_parser_rejects_result_without_grounded_findings() -> None:
    """验证所有证据均无效时不会生成看似正常的空档案。"""

    parser = RepositoryAnalysisResultParser()

    with pytest.raises(
        RepositoryAnalysisValidationError,
        match="No grounded findings",
    ):
        asyncio.run(
            parser.parse(
                _payload(
                    [
                        _finding(
                            path="missing.py",
                        )
                    ]
                ),
                FakeWorkspace(),
            )
        )


def test_parser_ignores_duplicate_and_invalid_findings() -> None:
    """验证重复 ID、非法字段和各种非法置信度不会进入结果。"""

    parser = RepositoryAnalysisResultParser()
    findings = [
        _finding(),
        _finding(
            finding_id="F1",
            claim="重复结论",
        ),
        _finding(
            finding_id="F2",
            target_field="metrics",
        ),
        _finding(
            finding_id="F3",
            confidence=True,
        ),
        _finding(
            finding_id="F4",
            confidence=-0.1,
        ),
        _finding(
            finding_id="F5",
            confidence=1.1,
        ),
        _finding(
            finding_id="F6",
            confidence="0.9",
        ),
    ]

    result = asyncio.run(
        parser.parse(
            _payload(findings),
            FakeWorkspace(),
        )
    )

    assert [
        finding["id"]
        for finding in result.findings
    ] == ["F1"]
    assert any(
        "重复" in warning
        for warning in result.warnings
    )
    assert any(
        "target_field" in warning
        for warning in result.warnings
    )
    assert any(
        "confidence" in warning
        for warning in result.warnings
    )


def test_parser_reports_invalid_declared_evidence() -> None:
    """验证不存在或字段不匹配的证据引用只产生警告。"""

    parser = RepositoryAnalysisResultParser()

    result = asyncio.run(
        parser.parse(
            _payload(
                [_finding()],
                evidence_by_field={
                    "architecture": ["F1"],
                    "tech_stack": ["missing"],
                    "unknown_field": ["F1"],
                },
            ),
            FakeWorkspace(),
        )
    )

    assert result.evidence_by_field == {
        "tech_stack": ["F1"],
    }
    assert any(
        "不匹配" in warning
        for warning in result.warnings
    )
    assert any(
        "无效证据" in warning
        for warning in result.warnings
    )
    assert any(
        "未知档案字段" in warning
        for warning in result.warnings
    )


def test_parser_enforces_finding_and_line_limits() -> None:
    """验证 Finding 数量和单条证据行数都受后端限制。"""

    parser = RepositoryAnalysisResultParser(
        max_findings=2,
        max_evidence_lines=3,
    )
    findings = [
        _finding(
            finding_id="F1",
            start_line=1,
            end_line=6,
        ),
        _finding(
            finding_id="F2",
        ),
        _finding(
            finding_id="F3",
        ),
    ]

    result = asyncio.run(
        parser.parse(
            _payload(findings),
            FakeWorkspace(),
        )
    )

    assert [
        finding["id"]
        for finding in result.findings
    ] == ["F2"]
    assert any(
        "证据范围过大" in warning
        for warning in result.warnings
    )
    assert any(
        "超过数量限制" in warning
        for warning in result.warnings
    )


def test_parser_defaults_to_twenty_four_findings() -> None:
    """验证默认结果预算只保留二十四条高价值代码结论。"""

    parser = RepositoryAnalysisResultParser()
    findings = [
        _finding(
            finding_id=f"F{index}",
        )
        for index in range(1, 26)
    ]

    result = asyncio.run(
        parser.parse(
            _payload(findings),
            FakeWorkspace(),
        )
    )

    assert len(result.findings) == 24
    assert result.findings[-1]["id"] == "F24"
    assert any(
        "超过数量限制" in warning
        for warning in result.warnings
    )


def test_parser_discards_quotes_longer_than_four_hundred_chars() -> None:
    """验证过长代码原文不会进入最终证据和项目档案。"""

    workspace = FakeWorkspace()
    workspace.files["app/long.py"] = "x" * 401
    parser = RepositoryAnalysisResultParser()

    result = asyncio.run(
        parser.parse(
            _payload(
                [
                    _finding(),
                    _finding(
                        finding_id="F2",
                        path="app/long.py",
                        start_line=1,
                        end_line=1,
                        quote="x" * 401,
                    ),
                ]
            ),
            workspace,
        )
    )

    assert [
        finding["id"]
        for finding in result.findings
    ] == ["F1"]
    assert any(
        "quote is too long" in warning
        for warning in result.warnings
    )


def test_parser_limits_final_warnings_after_validation() -> None:
    """验证模型警告和后端校验警告合并后仍受最终输出预算限制。"""

    parser = RepositoryAnalysisResultParser()
    result = asyncio.run(
        parser.parse(
            _payload(
                [_finding()],
                warnings=[
                    (f"warning-{index}-" + "x" * 250)
                    for index in range(8)
                ],
            ),
            FakeWorkspace(),
        )
    )

    assert len(result.warnings) == 8
    assert all(
        len(warning) <= 200
        for warning in result.warnings
    )


def test_parser_does_not_trust_raw_project_profile() -> None:
    """验证模型直接生成但没有 Finding 支撑的档案内容不会入库。"""

    parser = RepositoryAnalysisResultParser()

    result = asyncio.run(
        parser.parse(
            _payload(
                [_finding()],
                project_profile={
                    "name": "伪造的项目名",
                    "background": "伪造的项目背景",
                    "tech_stack": ["Django"],
                    "architecture": "伪造的系统架构",
                    "key_decisions": [
                        "伪造的技术选型"
                    ],
                    "metrics": [
                        "每秒处理十万请求"
                    ],
                    "outcomes": [
                        "收入增长百分之五百"
                    ],
                },
            ),
            FakeWorkspace(),
        )
    )

    assert result.draft["name"] == "offerpilot"
    assert result.draft["background"] == ""
    assert result.draft["tech_stack"] == [
        "项目使用 FastAPI 提供 HTTP 接口。"
    ]
    assert result.draft["architecture"] == ""
    assert result.draft["key_decisions"] == []
    assert result.draft["metrics"] == []
    assert result.draft["outcomes"] == []


def test_submission_rejects_unknown_coverage_area() -> None:
    """terminal coverage 必须严格拒绝 Schema 未声明的额外维度。"""

    parser = RepositoryAnalysisResultParser()
    coverage = {
        area: {
            "status": "covered" if area == "architecture" else "not_applicable",
            "evidence_ids": ["F1"] if area == "architecture" else [],
        }
        for area in (
            "project_overview",
            "tech_stack",
            "architecture",
            "business_flows",
            "data_and_integrations",
            "testing_and_reliability",
            "deployment",
        )
    }
    coverage["security"] = {"status": "not_applicable", "evidence_ids": []}

    with pytest.raises(
        RepositoryAnalysisValidationError,
        match="unknown areas: security",
    ):
        asyncio.run(
            parser.parse_submission(
                {
                    "findings": [_finding()],
                    "evidence_by_field": {"tech_stack": ["F1"]},
                    "coverage": coverage,
                    "warnings": [],
                },
                FakeWorkspace(),
            )
        )


def test_submission_requires_warning_for_each_not_found_area() -> None:
    """每个 not_found 覆盖维度都必须有可机器关联的 warning。"""

    parser = RepositoryAnalysisResultParser()
    coverage = {
        area: {
            "status": "covered" if area == "architecture" else "not_applicable",
            "evidence_ids": ["F1"] if area == "architecture" else [],
        }
        for area in (
            "project_overview",
            "tech_stack",
            "architecture",
            "business_flows",
            "data_and_integrations",
            "testing_and_reliability",
            "deployment",
        )
    }
    coverage["deployment"] = {"status": "not_found", "evidence_ids": []}

    with pytest.raises(
        RepositoryAnalysisValidationError,
        match="deployment",
    ):
        asyncio.run(
            parser.parse_submission(
                {
                    "findings": [_finding()],
                    "evidence_by_field": {"tech_stack": ["F1"]},
                    "coverage": coverage,
                    "warnings": ["一条无法关联到维度的说明。"],
                },
                FakeWorkspace(),
            )
        )


def test_submission_relocates_exact_quote_with_wrong_line_range() -> None:
    """quote 在同一文件中真实存在时，terminal 校验应安全修正错误行号。"""

    parser = RepositoryAnalysisResultParser()
    finding = _finding(
        start_line=1,
        end_line=1,
        quote="@app.get('/health')",
    )
    coverage = {
        area: {
            "status": "covered" if area == "architecture" else "not_applicable",
            "evidence_ids": ["F1"] if area == "architecture" else [],
        }
        for area in (
            "project_overview",
            "tech_stack",
            "architecture",
            "business_flows",
            "data_and_integrations",
            "testing_and_reliability",
            "deployment",
        )
    }

    result = asyncio.run(
        parser.parse_submission(
            {
                "findings": [finding],
                "evidence_by_field": {"tech_stack": ["F1"]},
                "coverage": coverage,
                "warnings": [],
            },
            FakeWorkspace(),
        )
    )

    assert result.findings[0]["start_line"] == 4
    assert result.findings[0]["end_line"] == 4
