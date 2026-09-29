


from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

from src.evaluate import evaluate, load_ground_truth
from src.extract import extract_action_items
from src.preprocess import segment_transcript
from src.validate import validate_items

PROJECT_ROOT = Path(__file__).resolve().parent
SAMPLE_DIR = PROJECT_ROOT / "sample_data"
APP_PATH = PROJECT_ROOT / "app.py"


def run_one(transcript_path: Path) -> None:
    name = transcript_path.stem
    gt_path = transcript_path.with_suffix(".groundtruth.json")

    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")

    text = transcript_path.read_text(encoding="utf-8")
    turns = segment_transcript(text)
    result = extract_action_items(turns, transcript_id=name)
    result = validate_items(result)

    print(f"Model used: {result.model_used}")
    print(f"Extracted {len(result.items)} item(s):\n")
    for i, item in enumerate(result.items, start=1):
        deadline = item.deadline.isoformat() if item.deadline else "—"
        print(
            f"  {i}. [{item.status.value:>16}] (conf {item.confidence:.2f}) "
            f"{item.task}"
        )
        print(f"     owner: {item.owner or '—'}   deadline: {item.deadline_raw or '—'} -> {deadline}")

    if gt_path.exists():
        gt = load_ground_truth(gt_path)
        report = evaluate(result.items, gt)
        print(f"\nEval vs. ground truth ({len(gt)} annotated items):")
        print(
            f"  precision={report.precision}  recall={report.recall}  f1={report.f1}  "
            f"owner_acc={report.owner_accuracy}  deadline_acc={report.deadline_accuracy}"
        )
    else:
        print("\n(no ground-truth file found, skipping evaluation)")


def launch_streamlit_app() -> None:
    """Launch streamlit run app.py and block until the user stops it (Ctrl+C)."""
    if importlib.util.find_spec("streamlit") is None:
        print(
            "\nStreamlit isn't installed, so the app can't be launched.\n"
            "Install it with:  pip install -r requirements.txt\n"
            "Then run:         streamlit run app.py"
        )
        return

    print(f"\n{'=' * 60}\nLaunching Streamlit app (Ctrl+C to stop)...\n{'=' * 60}")
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(APP_PATH)], cwd=str(PROJECT_ROOT))


def main() -> None:
    launch_app = "--no-app" not in sys.argv

    transcripts = sorted(SAMPLE_DIR.glob("*.txt"))
    if not transcripts:
        print(f"No .txt transcripts found in {SAMPLE_DIR}")
    else:
        print(f"Found {len(transcripts)} transcript(s) in {SAMPLE_DIR}")
        for path in transcripts:
            run_one(path)
        print(f"\n{'=' * 60}\nDone. Ran {len(transcripts)} transcript(s).\n{'=' * 60}")

    if launch_app:
        launch_streamlit_app()


if __name__ == "__main__":
    main()