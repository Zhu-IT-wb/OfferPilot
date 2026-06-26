from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ApplicationStatus(str, Enum):
    PLANNED = "planned"
    SUBMITTED = "submitted"
    WRITTEN_TEST = "written_test"
    INTERVIEW_1 = "interview_1"
    INTERVIEW_2 = "interview_2"
    INTERVIEW_3 = "interview_3"
    HR = "hr"
    INTERVIEW_SCHEDULED = "interview_scheduled"
    OFFER = "offer"
    REJECTED = "rejected"
    NO_RESPONSE = "no_response"
    WITHDRAWN = "withdrawn"


@dataclass
class Application:
    id: str
    company: str
    role: str
    status: ApplicationStatus = ApplicationStatus.PLANNED
    interview_time: Optional[str] = None
    round: Optional[str] = None
    jd_keywords: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "company": self.company,
            "role": self.role,
            "status": self.status.value,
            "interview_time": self.interview_time,
            "round": self.round,
            "jd_keywords": list(self.jd_keywords),
        }


def application_status_from_round(round_name: Optional[str]) -> ApplicationStatus:
    status_by_round = {
        "笔试": ApplicationStatus.WRITTEN_TEST,
        "一面": ApplicationStatus.INTERVIEW_1,
        "二面": ApplicationStatus.INTERVIEW_2,
        "三面": ApplicationStatus.INTERVIEW_3,
        "HR 面": ApplicationStatus.HR,
        "hr 面": ApplicationStatus.HR,
        "HR面": ApplicationStatus.HR,
        "hr面": ApplicationStatus.HR,
    }
    if round_name:
        return status_by_round.get(round_name, ApplicationStatus.INTERVIEW_SCHEDULED)
    return ApplicationStatus.PLANNED
