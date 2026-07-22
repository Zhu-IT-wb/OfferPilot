import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.models.leetcode import LeetCodeDifficulty, LeetCodeProblem


HOT100_SOURCE_URL = "https://leetcode.cn/studyplan/top-100-liked/"
HOT100_SOURCE_NAME = "leetcode_hot_100"
_DEFAULT_SNAPSHOT_PATH = Path(__file__).resolve().parents[1] / "data" / "leetcode_hot100.json"


class LeetCodeCatalogError(ValueError):
    pass


@dataclass(frozen=True)
class LeetCodeCatalogSnapshot:
    source_url: str
    source_version: str
    problems: List[LeetCodeProblem]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_url": self.source_url,
            "source_version": self.source_version,
            "problems": [problem.to_dict() for problem in self.problems],
        }


def load_hot100_snapshot(path: Optional[Path] = None) -> LeetCodeCatalogSnapshot:
    snapshot_path = path or _DEFAULT_SNAPSHOT_PATH
    try:
        payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LeetCodeCatalogError(f"Unable to load Hot 100 snapshot: {snapshot_path}") from exc

    problems = [_problem_from_snapshot(item) for item in payload.get("problems", [])]
    _validate_hot100(problems)
    return LeetCodeCatalogSnapshot(
        source_url=str(payload.get("source_url") or HOT100_SOURCE_URL),
        source_version=str(payload.get("source_version") or "unknown"),
        problems=problems,
    )


def parse_hot100_html(html: str, source_version: str = "manual") -> LeetCodeCatalogSnapshot:
    match = re.search(
        r'<script[^>]+id=["\']__NEXT_DATA__["\'][^>]*>(.*?)</script>',
        html,
        flags=re.DOTALL,
    )
    if match is None:
        raise LeetCodeCatalogError("LeetCode page does not contain __NEXT_DATA__.")
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        raise LeetCodeCatalogError("LeetCode __NEXT_DATA__ is invalid JSON.") from exc

    plan = _find_plan(payload)
    if plan is None:
        raise LeetCodeCatalogError("LeetCode page does not contain a Hot 100 study plan.")

    problems: List[LeetCodeProblem] = []
    problem_order = 0
    for category_order, group in enumerate(plan.get("planSubGroups", [])):
        category = str(group.get("name") or "未分类").strip()
        for question in group.get("questions", []):
            problem_order += 1
            slug = str(question.get("titleSlug") or "").strip()
            frontend_id = str(question.get("questionFrontendId") or "").strip()
            topics = [
                str(tag.get("nameTranslated") or tag.get("name") or tag.get("slug") or "").strip()
                for tag in question.get("topicTags", [])
                if isinstance(tag, dict)
            ]
            problems.append(
                LeetCodeProblem(
                    id=f"leetcode_{frontend_id}",
                    frontend_id=frontend_id,
                    title_zh=str(question.get("translatedTitle") or question.get("title") or "").strip(),
                    title_en=str(question.get("title") or "").strip(),
                    slug=slug,
                    difficulty=_difficulty_from_value(question.get("difficulty")),
                    topics=[topic for topic in topics if topic],
                    category=category,
                    category_order=category_order,
                    problem_order=problem_order,
                    url=f"https://leetcode.cn/problems/{slug}/",
                )
            )

    _validate_hot100(problems)
    return LeetCodeCatalogSnapshot(
        source_url=HOT100_SOURCE_URL,
        source_version=source_version,
        problems=problems,
    )


def _find_plan(value: Any) -> Optional[Dict[str, Any]]:
    if isinstance(value, dict):
        groups = value.get("planSubGroups")
        if isinstance(groups, list):
            return value
        for nested in value.values():
            found = _find_plan(nested)
            if found is not None:
                return found
    elif isinstance(value, list):
        for nested in value:
            found = _find_plan(nested)
            if found is not None:
                return found
    return None


def _difficulty_from_value(value: Any) -> LeetCodeDifficulty:
    normalized = str(value or "").strip().lower()
    try:
        return LeetCodeDifficulty(normalized)
    except ValueError as exc:
        raise LeetCodeCatalogError(f"Unknown LeetCode difficulty: {value}") from exc


def _problem_from_snapshot(item: Dict[str, Any]) -> LeetCodeProblem:
    try:
        return LeetCodeProblem(
            id=str(item["id"]),
            frontend_id=str(item["frontend_id"]),
            title_zh=str(item["title_zh"]),
            title_en=str(item["title_en"]),
            slug=str(item["slug"]),
            difficulty=_difficulty_from_value(item["difficulty"]),
            topics=[str(topic) for topic in item["topics"]],
            category=str(item["category"]),
            category_order=int(item["category_order"]),
            problem_order=int(item["problem_order"]),
            url=str(item["url"]),
            source=str(item.get("source") or HOT100_SOURCE_NAME),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise LeetCodeCatalogError("Hot 100 snapshot contains an invalid problem.") from exc


def _validate_hot100(problems: List[LeetCodeProblem]) -> None:
    if len(problems) != 100:
        raise LeetCodeCatalogError(f"Hot 100 catalog must contain exactly 100 problems, got {len(problems)}.")
    if len({problem.frontend_id for problem in problems}) != 100:
        raise LeetCodeCatalogError("Hot 100 frontend IDs must be unique.")
    if len({problem.slug for problem in problems}) != 100:
        raise LeetCodeCatalogError("Hot 100 slugs must be unique.")
    for problem in problems:
        if not all((problem.frontend_id, problem.title_zh, problem.slug, problem.category, problem.url)):
            raise LeetCodeCatalogError("Hot 100 problem metadata is incomplete.")
        if not problem.topics:
            raise LeetCodeCatalogError(f"Hot 100 problem has no topics: {problem.slug}")
