from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# 枚举 ApplicationStatus 的可选值。
class ApplicationStatus(str, Enum):
    PLANNED = "planned"
    SUBMITTED = "submitted"
    WRITTEN_TEST = "written_test"
    WRITTEN_TEST_PASSED = "written_test_passed"
    INTERVIEW_1 = "interview_1"
    INTERVIEW_1_PASSED = "interview_1_passed"
    INTERVIEW_2 = "interview_2"
    INTERVIEW_2_PASSED = "interview_2_passed"
    INTERVIEW_3 = "interview_3"
    INTERVIEW_3_PASSED = "interview_3_passed"
    HR = "hr"
    INTERVIEW_SCHEDULED = "interview_scheduled"
    OFFER = "offer"
    REJECTED = "rejected"
    NO_RESPONSE = "no_response"
    WITHDRAWN = "withdrawn"


APPLICATION_STATUS_LABELS = {
    ApplicationStatus.PLANNED: "待投递/待确认",
    ApplicationStatus.SUBMITTED: "已投递",
    ApplicationStatus.WRITTEN_TEST: "笔试阶段",
    ApplicationStatus.WRITTEN_TEST_PASSED: "笔试通过",
    ApplicationStatus.INTERVIEW_1: "一面阶段",
    ApplicationStatus.INTERVIEW_1_PASSED: "一面通过",
    ApplicationStatus.INTERVIEW_2: "二面阶段",
    ApplicationStatus.INTERVIEW_2_PASSED: "二面通过",
    ApplicationStatus.INTERVIEW_3: "三面阶段",
    ApplicationStatus.INTERVIEW_3_PASSED: "三面通过",
    ApplicationStatus.HR: "HR 面阶段",
    ApplicationStatus.INTERVIEW_SCHEDULED: "面试已安排",
    ApplicationStatus.OFFER: "Offer",
    ApplicationStatus.REJECTED: "已拒绝",
    ApplicationStatus.NO_RESPONSE: "暂无反馈",
    ApplicationStatus.WITHDRAWN: "已放弃",
}

INTERVIEW_APPLICATION_STATUSES = {
    ApplicationStatus.WRITTEN_TEST,
    ApplicationStatus.WRITTEN_TEST_PASSED,
    ApplicationStatus.INTERVIEW_1,
    ApplicationStatus.INTERVIEW_1_PASSED,
    ApplicationStatus.INTERVIEW_2,
    ApplicationStatus.INTERVIEW_2_PASSED,
    ApplicationStatus.INTERVIEW_3,
    ApplicationStatus.INTERVIEW_3_PASSED,
    ApplicationStatus.HR,
    ApplicationStatus.INTERVIEW_SCHEDULED,
}


# 定义 Application 相关的数据结构或领域对象。
@dataclass
class Application:
    id: str
    company: str
    role: str
    base_location: Optional[str] = None
    owner_id: str = "local_user"
    status: ApplicationStatus = ApplicationStatus.PLANNED
    interview_time: Optional[str] = None
    round: Optional[str] = None
    jd_keywords: List[str] = field(default_factory=list)

    # 将当前领域对象转换为可序列化字典。
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "owner_id": self.owner_id,
            "company": self.company,
            "role": self.role,
            "base_location": self.base_location,
            "status": self.status.value,
            "interview_time": self.interview_time,
            "round": self.round,
            "jd_keywords": list(self.jd_keywords),
        }


# 将投递状态转换为用户可读的中文标签。
def application_status_label(status: ApplicationStatus) -> str:
    return APPLICATION_STATUS_LABELS.get(status, status.value)


# 根据面试轮次推断投递状态。
def application_status_from_round(round_name: Optional[str]) -> ApplicationStatus:
    status_by_round = {
        "笔试": ApplicationStatus.WRITTEN_TEST,
        "一面": ApplicationStatus.INTERVIEW_1,
        "二面": ApplicationStatus.INTERVIEW_2,
        "三面": ApplicationStatus.INTERVIEW_3,
        "hr": ApplicationStatus.HR,
        "hr面": ApplicationStatus.HR,
    }
    if round_name:
        return status_by_round.get(_normalize_round(round_name), ApplicationStatus.INTERVIEW_SCHEDULED)
    return ApplicationStatus.PLANNED


# 根据通过的面试轮次推断下一阶段状态。
def application_passed_status_from_round(round_name: Optional[str]) -> Optional[ApplicationStatus]:
    status_by_round = {
        "笔试": ApplicationStatus.WRITTEN_TEST_PASSED,
        "一面": ApplicationStatus.INTERVIEW_1_PASSED,
        "二面": ApplicationStatus.INTERVIEW_2_PASSED,
        "三面": ApplicationStatus.INTERVIEW_3_PASSED,
        "hr": ApplicationStatus.OFFER,
        "hr面": ApplicationStatus.OFFER,
    }
    if not round_name:
        return None
    return status_by_round.get(_normalize_round(round_name))


# 标准化 round。
def _normalize_round(round_name: str) -> str:
    return round_name.replace(" ", "").lower()
