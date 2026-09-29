

from __future__ import annotations

from datetime import date, timedelta
from difflib import SequenceMatcher

from dateutil import parser as date_parser

from .schema import ActionItem, ActionStatus, ExtractionResult

DUPLICATE_SIMILARITY_THRESHOLD = 0.85  # 0-1, difflib ratio


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def normalize_deadline(deadline_raw: str | None, reference_date: date) -> date | None:
    """Best-effort conversion of a natural-language deadline into an ISO date."""
    if not deadline_raw:
        return None

    text = deadline_raw.lower().strip()
    weekdays = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

    if "tomorrow" in text:
        return reference_date + timedelta(days=1)
    if "end of day" in text or "eod" in text:
        return reference_date
    if "end of week" in text:
        return reference_date + timedelta(days=(4 - reference_date.weekday()) % 7)
    if "end of month" in text:
        next_month = reference_date.replace(day=28) + timedelta(days=4)
        return next_month.replace(day=1) - timedelta(days=1)

    for i, wd in enumerate(weekdays):
        if wd in text:
            days_ahead = (i - reference_date.weekday()) % 7
            if "next" in text or days_ahead == 0:
                days_ahead += 7
            return reference_date + timedelta(days=days_ahead)

    try:
        return date_parser.parse(text, fuzzy=True, default=reference_date).date()
    except (ValueError, OverflowError):
        return None


def validate_items(result: ExtractionResult, reference_date: date | None = None) -> ExtractionResult:
    """Mutate items in place: fill normalized deadlines, flag issues, tag duplicates."""
    reference_date = reference_date or date.today()
    seen: list[ActionItem] = []

    for item in result.items:
        if item.deadline_raw:
            item.deadline = normalize_deadline(item.deadline_raw, reference_date)
            if item.deadline is None:
                item.status = ActionStatus.UNPARSEABLE_DATE

        if not item.owner and item.status == ActionStatus.OK:
            item.status = ActionStatus.MISSING_OWNER

        if item.status == ActionStatus.OK:
            for prior in seen:
                if _similarity(item.task, prior.task) >= DUPLICATE_SIMILARITY_THRESHOLD:
                    item.status = ActionStatus.DUPLICATE
                    break

        seen.append(item)

    return result
