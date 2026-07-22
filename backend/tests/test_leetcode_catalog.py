import json

import pytest

from app.services.leetcode_catalog import (
    LeetCodeCatalogError,
    load_hot100_snapshot,
    parse_hot100_html,
)


def test_checked_in_hot100_snapshot_contains_100_unique_problems() -> None:
    snapshot = load_hot100_snapshot()

    assert len(snapshot.problems) == 100
    assert len({problem.frontend_id for problem in snapshot.problems}) == 100
    assert len({problem.slug for problem in snapshot.problems}) == 100
    assert snapshot.problems[0].title_zh == "两数之和"
    assert snapshot.problems[0].url == "https://leetcode.cn/problems/two-sum/"
    assert all(problem.category for problem in snapshot.problems)
    assert all(problem.topics for problem in snapshot.problems)


def test_hot100_html_parser_rejects_incomplete_catalog() -> None:
    page_data = {
        "props": {
            "pageProps": {
                "studyPlanDetail": {
                    "planSubGroups": [
                        {
                            "name": "哈希",
                            "questions": [
                                {
                                    "translatedTitle": "两数之和",
                                    "title": "Two Sum",
                                    "titleSlug": "two-sum",
                                    "questionFrontendId": "1",
                                    "difficulty": "EASY",
                                    "topicTags": [
                                        {"slug": "array", "nameTranslated": "数组", "name": "Array"}
                                    ],
                                }
                            ],
                        }
                    ]
                }
            }
        }
    }
    html = f'<html><script id="__NEXT_DATA__" type="application/json">{json.dumps(page_data)}</script></html>'

    with pytest.raises(LeetCodeCatalogError, match="exactly 100"):
        parse_hot100_html(html)


def test_hot100_snapshot_loader_rejects_incomplete_file(tmp_path) -> None:
    snapshot = tmp_path / "hot100.json"
    snapshot.write_text(
        json.dumps(
            {
                "source_url": "https://leetcode.cn/studyplan/top-100-liked/",
                "source_version": "broken",
                "problems": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(LeetCodeCatalogError, match="exactly 100"):
        load_hot100_snapshot(snapshot)
