

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from enum import Enum


class ActionStatus(str, Enum):
    OK = "ok"
    MISSING_OWNER = "missing_owner"
    UNPARSEABLE_DATE = "unparseable_date"
    DUPLICATE = "duplicate"


@dataclass
class ActionItem:
    task: str
    owner: str | None = None
    deadline_raw: str | None = None
    deadline: date | None = None
    confidence: float = 0.5
    source_quote: str | None = None
    status: ActionStatus = ActionStatus.OK

    def __post_init__(self):
        self.confidence = round(max(0.0, min(1.0, float(self.confidence))), 2)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["deadline"] = self.deadline.isoformat() if self.deadline else None
        d["status"] = self.status.value
        return d


@dataclass
class ExtractionResult:
    items: list[ActionItem] = field(default_factory=list)
    transcript_id: str | None = None
    model_used: str | None = None
