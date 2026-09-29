

from __future__ import annotations

import json
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from .schema import ActionItem

MATCH_THRESHOLD = 0.60  


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


@dataclass
class EvalReport:
    precision: float
    recall: float
    f1: float
    owner_accuracy: float
    deadline_accuracy: float
    n_predicted: int
    n_ground_truth: int
    n_matched: int


def load_ground_truth(path: str | Path) -> list[dict]:
    return json.loads(Path(path).read_text())["action_items"]


def evaluate(predicted: list[ActionItem], ground_truth: list[dict]) -> EvalReport:
    matched_gt_idx: set[int] = set()
    owner_hits = 0
    deadline_hits = 0
    n_matched = 0

    for item in predicted:
        best_idx, best_score = -1, 0
        for i, gt in enumerate(ground_truth):
            if i in matched_gt_idx:
                continue
            score = _similarity(item.task, gt["task"])
            if score > best_score:
                best_idx, best_score = i, score

        if best_score >= MATCH_THRESHOLD:
            matched_gt_idx.add(best_idx)
            n_matched += 1
            gt = ground_truth[best_idx]
            if gt.get("owner") and item.owner and gt["owner"].lower() == item.owner.lower():
                owner_hits += 1
            if gt.get("deadline_raw") and item.deadline_raw:
                deadline_hits += 1

    n_predicted = len(predicted)
    n_gt = len(ground_truth)
    precision = n_matched / n_predicted if n_predicted else 0.0
    recall = n_matched / n_gt if n_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return EvalReport(
        precision=round(precision, 3),
        recall=round(recall, 3),
        f1=round(f1, 3),
        owner_accuracy=round(owner_hits / n_matched, 3) if n_matched else 0.0,
        deadline_accuracy=round(deadline_hits / n_matched, 3) if n_matched else 0.0,
        n_predicted=n_predicted,
        n_ground_truth=n_gt,
        n_matched=n_matched,
    )
