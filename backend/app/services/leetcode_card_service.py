from datetime import date
from typing import Any, Dict

from app.models.leetcode import LeetCodeDeliveryType
from app.repositories.leetcode_repository import LeetCodeRepository
from app.schemas.tool import ToolResult
from app.services.leetcode_messages import build_leetcode_card
from app.services.leetcode_recommendation import LeetCodeRecommendationWorkflow
from app.tools.leetcode_tools import record_leetcode_result


class LeetCodeCardService:
    def __init__(self, repository: LeetCodeRepository) -> None:
        self.repository = repository

    def build_today_card(
        self,
        owner_id: str,
        today: date,
        delivery_type: LeetCodeDeliveryType = LeetCodeDeliveryType.MORNING,
    ) -> Dict[str, Any]:
        recommendations = LeetCodeRecommendationWorkflow(self.repository).get_today(
            owner_id=owner_id,
            today=today,
        )
        return build_leetcode_card(recommendations, delivery_type)

    def record_result(
        self,
        owner_id: str,
        assignment_id: str,
        result: str,
    ) -> ToolResult:
        return record_leetcode_result(
            self.repository,
            {
                "owner_id": owner_id,
                "assignment_id": assignment_id,
                "result": result,
            },
        )
