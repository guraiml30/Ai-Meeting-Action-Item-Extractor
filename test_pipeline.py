"""Run with: python -m pytest tests/ -v (or just `python tests/test_pipeline.py`)"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluate import evaluate, load_ground_truth
from src.extract import extract_action_items
from src.preprocess import segment_transcript
from src.validate import validate_items

SAMPLE_DIR = Path(__file__).resolve().parents[1] / "sample_data"


def run_on(transcript_name: str):
    text = (SAMPLE_DIR / f"{transcript_name}.txt").read_text()
    turns = segment_transcript(text)
    assert turns, "segmentation produced no turns"

    result = extract_action_items(turns, transcript_id=transcript_name)
    result = validate_items(result)

    gt = load_ground_truth(SAMPLE_DIR / f"{transcript_name}.groundtruth.json")
    report = evaluate(result.items, gt)
    print(f"\n{transcript_name} ({result.model_used}): {report}")
    return result, report


def test_transcript_1_extracts_items():
    result, report = run_on("transcript_1")
    assert len(result.items) > 0
    assert report.recall > 0


def test_transcript_2_flags_something():
    result, report = run_on("transcript_2")
    assert len(result.items) > 0


if __name__ == "__main__":
    run_on("transcript_1")
    run_on("transcript_2")
