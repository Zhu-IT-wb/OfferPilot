import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.project_analysis.repository_workspace import (
    GitHubRepositoryWorkspace,
    RepositoryAccessError,
)


ALLOWED_TARGET_FIELDS = {
    "name",
    "background",
    "tech_stack",
    "architecture",
    "key_decisions",
    "technical_challenges",
    "resume_description",
    "supplemental_text",
}

REQUIRED_COVERAGE_AREAS = (
    "project_overview",
    "tech_stack",
    "architecture",
    "business_flows",
    "data_and_integrations",
    "testing_and_reliability",
    "deployment",
)
ALLOWED_COVERAGE_STATUSES = {
    "covered",
    "not_found",
    "not_applicable",
}

DEFAULT_MAX_FINDINGS = 24
DEFAULT_MAX_QUOTE_CHARS = 400
DEFAULT_MAX_WARNINGS = 8
DEFAULT_MAX_WARNING_CHARS = 200
SUBMISSION_FIELDS = {
    "findings",
    "evidence_by_field",
    "coverage",
    "warnings",
}
FINDING_FIELDS = {
    "id",
    "claim",
    "topic",
    "target_field",
    "path",
    "start_line",
    "end_line",
    "quote",
    "confidence",
}



class RepositoryAnalysisValidationError(ValueError):
    """模型返回的项目分析结果不符合后端协议。"""


@dataclass(frozen=True)
class RepositoryAnalysisResult:
    """保存经过代码证据验证的项目分析结果。"""

    draft: Dict[str, Any]
    findings: List[Dict[str, Any]]
    evidence_by_field: Dict[str, List[str]]
    warnings: List[str]
    coverage: Optional[Dict[str, Dict[str, Any]]] = None


class RepositoryAnalysisResultParser:
    """解析模型输出并验证每条代码证据。"""

    def __init__(
        self,
        max_findings: int = DEFAULT_MAX_FINDINGS,
        max_evidence_lines: int = 80,
        max_quote_chars: int = (
            DEFAULT_MAX_QUOTE_CHARS
        ),
    ) -> None:
        """设置 Finding 数量、证据行数和原文字符数上限。"""

        if max_findings < 1:
            raise ValueError(
                "max_findings must be positive."
            )

        if max_evidence_lines < 1:
            raise ValueError(
                "max_evidence_lines must be positive."
            )

        if max_quote_chars < 1:
            raise ValueError(
                "max_quote_chars must be positive."
            )

        self._max_findings = max_findings
        self._max_evidence_lines = (
            max_evidence_lines
        )
        self._max_quote_chars = max_quote_chars

    async def parse(
        self,
        raw_content: str,
        workspace: GitHubRepositoryWorkspace,
    ) -> RepositoryAnalysisResult:
        """解析最终 JSON，并只保留能够被仓库内容证明的结论。"""

        payload = self._decode_payload(raw_content)
        return await self._parse_payload(
            payload,
            workspace,
            require_coverage=False,
            strict_submission=False,
        )

    async def parse_submission(
        self,
        payload: Dict[str, Any],
        workspace: GitHubRepositoryWorkspace,
    ) -> RepositoryAnalysisResult:
        """验证 terminal 工具提交，包括显式分析覆盖状态。"""

        if not isinstance(payload, dict):
            raise RepositoryAnalysisValidationError(
                "submit_analysis arguments must be an object."
            )
        missing = SUBMISSION_FIELDS - payload.keys()
        unknown = payload.keys() - SUBMISSION_FIELDS
        if missing:
            raise RepositoryAnalysisValidationError(
                "submit_analysis is missing required fields: "
                + ", ".join(sorted(missing))
            )
        if unknown:
            raise RepositoryAnalysisValidationError(
                "submit_analysis contains unknown fields: "
                + ", ".join(sorted(unknown))
            )
        return await self._parse_payload(
            payload,
            workspace,
            require_coverage=True,
            strict_submission=True,
        )

    async def _parse_payload(
        self,
        payload: Dict[str, Any],
        workspace: GitHubRepositoryWorkspace,
        require_coverage: bool,
        strict_submission: bool,
    ) -> RepositoryAnalysisResult:
        """校验已解码的分析载荷，并重新读取代码证据。"""

        warnings = self._parse_warnings(
            payload.get("warnings"),
            strict=strict_submission,
        )
        coverage = self._parse_coverage(
            payload.get("coverage"),
            warnings,
            require=require_coverage,
        )

        raw_findings = payload.get("findings")

        if not isinstance(raw_findings, list):
            raise RepositoryAnalysisValidationError(
                "findings must be a list."
            )

        if strict_submission and len(raw_findings) > self._max_findings:
            raise RepositoryAnalysisValidationError(
                f"findings exceeds the limit of {self._max_findings}."
            )

        grounded_findings = []
        seen_finding_ids = set()

        for index, raw_finding in enumerate(
            raw_findings[:self._max_findings],
            start=1,
        ):
            try:
                finding = self._parse_finding(
                    raw_finding
                )
            except RepositoryAnalysisValidationError as exc:
                if strict_submission:
                    raise RepositoryAnalysisValidationError(
                        f"findings[{index - 1}] is invalid: {exc}"
                    ) from exc
                warnings.append(
                    f"已忽略第 {index} 条无效结论：{exc}"
                )
                continue

            finding_id = finding["id"]

            if finding_id in seen_finding_ids:
                if strict_submission:
                    raise RepositoryAnalysisValidationError(
                        f"findings[{index - 1}] duplicates ID {finding_id}."
                    )
                warnings.append(
                    f"已忽略重复的 Finding ID："
                    f"{finding_id}"
                )
                continue

            seen_finding_ids.add(finding_id)

            if (
                finding["end_line"]
                - finding["start_line"]
                + 1
                > self._max_evidence_lines
            ):
                if strict_submission:
                    raise RepositoryAnalysisValidationError(
                        f"finding {finding_id} evidence range exceeds "
                        f"{self._max_evidence_lines} lines."
                    )
                warnings.append(
                    f"已忽略证据范围过大的结论："
                    f"{finding_id}"
                )
                continue

            try:
                file_region = await workspace.read_file(
                    path=finding["path"],
                    start_line=finding["start_line"],
                    end_line=finding["end_line"],
                )
            except RepositoryAccessError as exc:
                if strict_submission:
                    raise RepositoryAnalysisValidationError(
                        f"finding {finding_id} evidence cannot be read: {exc}"
                    ) from exc
                warnings.append(
                    f"已忽略无法读取证据的结论 "
                    f"{finding_id}：{exc}"
                )
                continue

            actual_content = str(
                file_region.get("content") or ""
            ).strip()

            if finding["quote"] not in actual_content:
                if strict_submission:
                    raise RepositoryAnalysisValidationError(
                        f"finding {finding_id} quote does not match "
                        "the declared source range."
                    )
                warnings.append(
                    f"已忽略原文不匹配的结论："
                    f"{finding_id}"
                )
                continue

            grounded_findings.append(
                finding
            )

        if len(raw_findings) > self._max_findings:
            warnings.append(
                "模型返回的 Findings 超过数量限制，"
                "多余部分已忽略。"
            )

        if not grounded_findings:
            raise RepositoryAnalysisValidationError(
                "No grounded findings were found."
            )

        evidence_by_field = (
            self._build_evidence_by_field(
                grounded_findings
            )
        )

        if require_coverage:
            self._validate_coverage_evidence(
                coverage,
                grounded_findings,
            )

        declared_evidence_warnings = self._check_declared_evidence(
            payload.get("evidence_by_field"),
            grounded_findings,
        )
        if strict_submission and declared_evidence_warnings:
            raise RepositoryAnalysisValidationError(
                "evidence_by_field is invalid: "
                + "; ".join(declared_evidence_warnings)
            )
        warnings.extend(declared_evidence_warnings)

        draft = self._build_draft(
            workspace=workspace,
            findings=grounded_findings,
        )

        return RepositoryAnalysisResult(
            draft=draft,
            findings=grounded_findings,
            evidence_by_field=evidence_by_field,
            warnings=self._limit_warnings(
                self._deduplicate(warnings)
            ),
            coverage=coverage,
        )

    @staticmethod
    def _parse_coverage(
        raw_coverage: Any,
        warnings: List[str],
        require: bool,
    ) -> Dict[str, Dict[str, Any]]:
        """读取模型对每个分析维度的明确完成判断。"""

        if raw_coverage is None and not require:
            return {}
        if not isinstance(raw_coverage, dict):
            raise RepositoryAnalysisValidationError(
                "coverage must be an object."
            )
        missing = [
            area
            for area in REQUIRED_COVERAGE_AREAS
            if area not in raw_coverage
        ]
        if missing:
            raise RepositoryAnalysisValidationError(
                "coverage is missing required areas: "
                + ", ".join(missing)
            )
        unknown_areas = raw_coverage.keys() - set(REQUIRED_COVERAGE_AREAS)
        if unknown_areas:
            raise RepositoryAnalysisValidationError(
                "coverage contains unknown areas: "
                + ", ".join(sorted(unknown_areas))
            )
        normalized: Dict[str, Dict[str, Any]] = {}
        for area in REQUIRED_COVERAGE_AREAS:
            entry = raw_coverage[area]
            if not isinstance(entry, dict):
                raise RepositoryAnalysisValidationError(
                    f"coverage.{area} must be an object."
                )
            unknown = entry.keys() - {"status", "evidence_ids"}
            if unknown:
                raise RepositoryAnalysisValidationError(
                    f"coverage.{area} contains unknown fields: "
                    + ", ".join(sorted(unknown))
                )
            status = entry.get("status")
            evidence_ids = entry.get("evidence_ids")
            if status not in ALLOWED_COVERAGE_STATUSES:
                raise RepositoryAnalysisValidationError(
                    f"coverage.{area}.status is invalid."
                )
            if not isinstance(evidence_ids, list) or not all(
                isinstance(item, str) for item in evidence_ids
            ):
                raise RepositoryAnalysisValidationError(
                    f"coverage.{area}.evidence_ids must be a string list."
                )
            if status == "covered" and not evidence_ids:
                raise RepositoryAnalysisValidationError(
                    f"coverage.{area} is covered but has no evidence_ids."
                )
            if status != "covered" and evidence_ids:
                raise RepositoryAnalysisValidationError(
                    f"coverage.{area} may only reference evidence when covered."
                )
            normalized[area] = {
                "status": status,
                "evidence_ids": list(dict.fromkeys(evidence_ids)),
            }
        unexplained = [
            area
            for area, entry in normalized.items()
            if entry["status"] == "not_found"
            and not any(
                warning.startswith(f"[{area}]")
                for warning in warnings
            )
        ]
        if unexplained:
            raise RepositoryAnalysisValidationError(
                "coverage not_found areas require matching warnings using "
                "the [area] prefix: "
                + ", ".join(unexplained)
            )
        return normalized

    @staticmethod
    def _validate_coverage_evidence(
        coverage: Dict[str, Dict[str, Any]],
        findings: List[Dict[str, Any]],
    ) -> None:
        """确保 covered 状态只引用已通过代码校验的 Finding。"""

        finding_ids = {finding["id"] for finding in findings}
        for area, entry in coverage.items():
            invalid = [
                finding_id
                for finding_id in entry["evidence_ids"]
                if finding_id not in finding_ids
            ]
            if invalid:
                raise RepositoryAnalysisValidationError(
                    f"coverage.{area} references invalid evidence: "
                    + ", ".join(invalid)
                )

    @staticmethod
    def _decode_payload(
        raw_content: str,
    ) -> Dict[str, Any]:
        """把模型最终文本解析成顶层 JSON 对象。"""

        if not isinstance(raw_content, str):
            raise RepositoryAnalysisValidationError(
                "Analysis result must be a string."
            )

        normalized = raw_content.strip()

        if not normalized:
            raise RepositoryAnalysisValidationError(
                "Analysis result cannot be empty."
            )

        lines = normalized.splitlines()
        if (
            len(lines) >= 3
            and lines[0].strip().lower()
            in {"```", "```json"}
            and lines[-1].strip() == "```"
        ):
            normalized = "\n".join(
                lines[1:-1]
            ).strip()

        try:
            payload = json.loads(normalized)
        except json.JSONDecodeError as exc:
            payload = (
                RepositoryAnalysisResultParser
                ._extract_unique_analysis_object(
                    normalized
                )
            )
            if payload is None:
                raise RepositoryAnalysisValidationError(
                    "Analysis result is not valid JSON "
                    f"at line {exc.lineno}, "
                    f"column {exc.colno}: {exc.msg}."
                ) from exc

        if not isinstance(payload, dict):
            raise RepositoryAnalysisValidationError(
                "Analysis result must be a JSON object."
            )

        return payload

    @staticmethod
    def _extract_unique_analysis_object(
        content: str,
    ) -> Optional[Dict[str, Any]]:
        """从说明文字中提取唯一且具有 findings 字段的 JSON 对象。"""

        decoder = json.JSONDecoder()
        candidates = []
        best_failure = None

        for index, character in enumerate(content):
            if character != "{":
                continue
            try:
                value, _ = decoder.raw_decode(
                    content[index:]
                )
            except json.JSONDecodeError as exc:
                if (
                    best_failure is None
                    or exc.pos > best_failure[0]
                ):
                    best_failure = (
                        exc.pos,
                        index + exc.pos,
                        exc.msg,
                    )
                continue
            if (
                isinstance(value, dict)
                and "findings" in value
            ):
                candidates.append(value)

        if len(candidates) == 1:
            return candidates[0]

        if len(candidates) > 1:
            raise RepositoryAnalysisValidationError(
                "Analysis result contains multiple JSON "
                "objects with findings."
            )

        if best_failure is not None:
            _, absolute_position, message = (
                best_failure
            )
            line = (
                content.count(
                    "\n",
                    0,
                    absolute_position,
                )
                + 1
            )
            previous_newline = content.rfind(
                "\n",
                0,
                absolute_position,
            )
            column = (
                absolute_position
                - previous_newline
            )
            raise RepositoryAnalysisValidationError(
                "Analysis result is not valid JSON "
                f"at line {line}, column {column}: "
                f"{message}."
            )

        return None

    def _parse_finding(
        self,
        raw_finding: Any,
    ) -> Dict[str, Any]:
        """校验并标准化一条模型生成的 Finding。"""

        if not isinstance(raw_finding, dict):
            raise RepositoryAnalysisValidationError(
                "Finding must be an object."
            )

        unknown = raw_finding.keys() - FINDING_FIELDS
        if unknown:
            raise RepositoryAnalysisValidationError(
                "Finding contains unknown fields: "
                + ", ".join(sorted(unknown))
            )

        finding_id = self._required_text(
            raw_finding,
            "id",
            max_length=64,
        )
        claim = self._required_text(
            raw_finding,
            "claim",
            max_length=300,
        )
        topic = self._required_text(
            raw_finding,
            "topic",
            max_length=100,
        )
        target_field = self._required_text(
            raw_finding,
            "target_field",
            max_length=100,
        )
        path = self._required_text(
            raw_finding,
            "path",
            max_length=1000,
        )
        quote = self._required_text(
            raw_finding,
            "quote",
            max_length=self._max_quote_chars,
        )

        if target_field not in ALLOWED_TARGET_FIELDS:
            raise RepositoryAnalysisValidationError(
                "Finding target_field is not allowed."
            )

        start_line = self._positive_integer(
            raw_finding,
            "start_line",
        )
        end_line = self._positive_integer(
            raw_finding,
            "end_line",
        )

        if end_line < start_line:
            raise RepositoryAnalysisValidationError(
                "Finding end_line cannot be smaller "
                "than start_line."
            )

        confidence = self._confidence(
            raw_finding.get("confidence")
        )

        return {
            "id": finding_id,
            "claim": claim,
            "topic": topic,
            "target_field": target_field,
            "path": path,
            "start_line": start_line,
            "end_line": end_line,
            "quote": quote,
            "confidence": confidence,
        }

    @staticmethod
    def _required_text(
        payload: Dict[str, Any],
        field_name: str,
        max_length: int,
    ) -> str:
        """读取必填文本字段，并限制空值和最大长度。"""

        value = payload.get(field_name)

        if not isinstance(value, str):
            raise RepositoryAnalysisValidationError(
                f"{field_name} must be a string."
            )

        normalized = value.strip()

        if not normalized:
            raise RepositoryAnalysisValidationError(
                f"{field_name} cannot be empty."
            )

        if len(normalized) > max_length:
            raise RepositoryAnalysisValidationError(
                f"{field_name} is too long."
            )

        return normalized

    @staticmethod
    def _positive_integer(
        payload: Dict[str, Any],
        field_name: str,
    ) -> int:
        """读取大于零的整数，并拒绝被当作整数的布尔值。"""

        value = payload.get(field_name)

        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 1
        ):
            raise RepositoryAnalysisValidationError(
                f"{field_name} must be a positive integer."
            )

        return value

    @staticmethod
    def _confidence(
        value: Any,
    ) -> float:
        """校验并标准化零到一之间的置信度。"""

        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
        ):
            raise RepositoryAnalysisValidationError(
                "confidence must be a number."
            )

        normalized = float(value)

        if normalized < 0 or normalized > 1:
            raise RepositoryAnalysisValidationError(
                "confidence must be between 0 and 1."
            )

        return normalized

    @staticmethod
    def _parse_warnings(
        raw_warnings: Any,
        strict: bool = False,
    ) -> List[str]:
        """读取模型警告，并忽略空值和非字符串内容。"""

        if raw_warnings is None:
            return []

        if not isinstance(raw_warnings, list):
            raise RepositoryAnalysisValidationError(
                "warnings must be a list."
            )

        if strict and len(raw_warnings) > DEFAULT_MAX_WARNINGS:
            raise RepositoryAnalysisValidationError(
                f"warnings exceeds the limit of {DEFAULT_MAX_WARNINGS}."
            )

        warnings = []

        for index, item in enumerate(raw_warnings[
            :DEFAULT_MAX_WARNINGS
        ]):
            if not isinstance(item, str):
                if strict:
                    raise RepositoryAnalysisValidationError(
                        f"warnings[{index}] must be a string."
                    )
                continue

            normalized = item.strip()

            if not normalized:
                if strict:
                    raise RepositoryAnalysisValidationError(
                        f"warnings[{index}] cannot be empty."
                    )
                continue
            if strict and len(normalized) > DEFAULT_MAX_WARNING_CHARS:
                raise RepositoryAnalysisValidationError(
                    f"warnings[{index}] is too long."
                )
            warnings.append(normalized[:DEFAULT_MAX_WARNING_CHARS])

        return warnings

    @staticmethod
    def _limit_warnings(
        warnings: List[str],
    ) -> List[str]:
        """在全部校验结束后统一限制最终警告数量和单条长度。"""

        return [
            warning[:DEFAULT_MAX_WARNING_CHARS]
            for warning in warnings[:DEFAULT_MAX_WARNINGS]
        ]

    @staticmethod
    def _build_evidence_by_field(
        findings: List[Dict[str, Any]],
    ) -> Dict[str, List[str]]:
        """根据已验证 Finding 重新生成字段与证据的关系。"""

        evidence_by_field: Dict[
            str,
            List[str],
        ] = {}

        for finding in findings:
            target_field = finding[
                "target_field"
            ]
            evidence_by_field.setdefault(
                target_field,
                [],
            ).append(
                finding["id"]
            )

        return evidence_by_field

    @staticmethod
    def _check_declared_evidence(
        raw_mapping: Any,
        findings: List[Dict[str, Any]],
    ) -> List[str]:
        """检查模型声明的证据关系，并报告错误引用。"""

        if raw_mapping is None:
            return [
                "模型没有返回 evidence_by_field，"
                "后端已根据 Findings 自动重建。"
            ]

        if not isinstance(raw_mapping, dict):
            return [
                "模型返回的 evidence_by_field 无效，"
                "后端已根据 Findings 自动重建。"
            ]

        findings_by_id = {
            finding["id"]: finding
            for finding in findings
        }
        warnings = []

        for field_name, finding_ids in raw_mapping.items():
            if field_name not in ALLOWED_TARGET_FIELDS:
                warnings.append(
                    f"模型声明了未知档案字段："
                    f"{field_name}"
                )
                continue

            if not isinstance(finding_ids, list):
                warnings.append(
                    f"字段 {field_name} 的证据引用"
                    "不是列表。"
                )
                continue

            for finding_id in finding_ids:
                finding = findings_by_id.get(
                    finding_id
                )

                if finding is None:
                    warnings.append(
                        f"字段 {field_name} 引用了"
                        f"无效证据：{finding_id}"
                    )
                    continue

                if (
                    finding["target_field"]
                    != field_name
                ):
                    warnings.append(
                        f"证据 {finding_id} 与字段 "
                        f"{field_name} 不匹配。"
                    )

        return warnings

    @staticmethod
    def _build_draft(
        workspace: GitHubRepositoryWorkspace,
        findings: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """仅使用已验证结论重建安全的项目档案草稿。"""

        claims_by_field: Dict[
            str,
            List[str],
        ] = {}

        for finding in findings:
            claims_by_field.setdefault(
                finding["target_field"],
                [],
            ).append(
                finding["claim"]
            )

        for field_name, claims in (
            claims_by_field.items()
        ):
            claims_by_field[field_name] = (
                RepositoryAnalysisResultParser
                ._deduplicate(claims)
            )

        repository_name = (
            workspace.repository_url
            .rstrip("/")
            .rsplit("/", 1)[-1]
        )

        name_claims = claims_by_field.get(
            "name",
            [],
        )

        return {
            "name": (
                name_claims[0]
                if name_claims
                else repository_name
            ),
            "background": "\n".join(
                claims_by_field.get(
                    "background",
                    [],
                )
            ),
            "responsibilities": [],
            "tech_stack": list(
                claims_by_field.get(
                    "tech_stack",
                    [],
                )
            ),
            "architecture": "\n".join(
                claims_by_field.get(
                    "architecture",
                    [],
                )
            ),
            "key_decisions": list(
                claims_by_field.get(
                    "key_decisions",
                    [],
                )
            ),
            "technical_challenges": list(
                claims_by_field.get(
                    "technical_challenges",
                    [],
                )
            ),
            "metrics": [],
            "outcomes": [],
            "resume_description": "\n".join(
                claims_by_field.get(
                    "resume_description",
                    [],
                )
            ),
            "supplemental_text": "\n".join(
                claims_by_field.get(
                    "supplemental_text",
                    [],
                )
            ),
        }

    @staticmethod
    def _deduplicate(
        values: List[str],
    ) -> List[str]:
        """按原始顺序删除重复字符串。"""

        return list(
            dict.fromkeys(values)
        )
