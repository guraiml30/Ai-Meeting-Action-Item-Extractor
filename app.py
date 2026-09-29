"""
Step 6 in the plan: a simple upload-to-results interface.

Run with:  streamlit run app.py
Optional:  export ANTHROPIC_API_KEY=sk-...   (otherwise uses the rule-based fallback)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.extract import extract_action_items
from src.preprocess import segment_transcript
from src.validate import validate_items

APP_DIR = Path(__file__).resolve().parent
SAMPLE_DIR = APP_DIR / "sample_data"

st.set_page_config(page_title="Meeting Action-Item Extractor", layout="wide")
st.title("🗒️ AI Meeting Action-Item Extractor")
st.caption("Upload a transcript → get structured action items with task, owner, deadline, and confidence.")

SAMPLE_LABELS = {
    "(none)": None,
    "transcript_1.txt": SAMPLE_DIR / "transcript_1.txt",
    "transcript_2.txt": SAMPLE_DIR / "transcript_2.txt",
    "transcript_3.txt": SAMPLE_DIR / "transcript_3.txt",
}

with st.sidebar:
    st.header("Input")
    uploaded = st.file_uploader("Meeting transcript (.txt)", type=["txt"])
    sample_label = st.selectbox("...or try a sample", list(SAMPLE_LABELS.keys()))
    min_confidence = st.slider("Minimum confidence to display", 0.0, 1.0, 0.4, 0.05)
    run_button = st.button("Extract action items", type="primary")

sample_path = SAMPLE_LABELS[sample_label]

transcript_text = None
if uploaded is not None:
    transcript_text = uploaded.read().decode("utf-8", errors="ignore")
elif sample_path is not None:
    transcript_text = sample_path.read_text(encoding="utf-8")

if transcript_text:
    with st.expander("Transcript preview", expanded=False):
        st.text(transcript_text)

if run_button:
    if not transcript_text:
        st.warning("Upload a transcript or pick a sample first.")
        st.stop()

    with st.spinner("Segmenting and extracting..."):
        turns = segment_transcript(transcript_text)
        result = extract_action_items(turns)
        result = validate_items(result)

    st.success(f"Extracted {len(result.items)} candidate action item(s) using *{result.model_used}*.")

    rows = [
        {
            "Task": it.task,
            "Owner": it.owner or "—",
            "Deadline (raw)": it.deadline_raw or "—",
            "Deadline (parsed)": it.deadline.isoformat() if it.deadline else "—",
            "Confidence": it.confidence,
            "Status": it.status.value,
            "Source": it.source_quote or "—",
        }
        for it in result.items
        if it.confidence >= min_confidence
    ]

    if not rows:
        st.info("No action items met the confidence threshold.")
    else:
        df = pd.DataFrame(rows)

        def highlight_status(row):
            color = {
                "missing_owner": "background-color: #fff3cd; color: #856404",
                "unparseable_date": "background-color: #f8d7da; color: #721c24",
                "duplicate": "background-color: #e2e3e5; color: #383d41",
            }.get(row["Status"], "")
            return [color] * len(row)

        st.dataframe(df.style.apply(highlight_status, axis=1), use_container_width=True)

        col1, col2 = st.columns(2)
        col1.download_button(
            "Download CSV", df.to_csv(index=False).encode("utf-8"), "action_items.csv", "text/csv"
        )
        col2.download_button(
            "Download JSON", df.to_json(orient="records", indent=2), "action_items.json", "application/json"
        )
else:
    st.info("Upload a transcript, or pick a sample from the sidebar, then click *Extract action items*.")