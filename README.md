# AI Meeting Action-Item Extractor

Converts a meeting transcript into structured action items — `task`,
`owner`, `deadline`, and `confidence` — with validation and evaluation
built in. Built with Python end to end; Streamlit for the UI, per the
suggested stack.

## How the six steps map to this repo

| Step | Where |
|---|---|
| 1. Gather/generate annotated transcripts | `sample_data/*.txt` + `*.groundtruth.json` |
| 2. Clean and segment by speaker/sentence | `src/preprocess.py` |
| 3. LLM extraction | `src/extract.py` |
| 4. Structured output schema | `src/schema.py` |
| 5. Validation rules | `src/validate.py` |
| 6. Evaluation + upload interface | `src/evaluate.py`, `app.py` |

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-...   # optional — see fallback below
```

## Run the app

```bash
streamlit run app.py
```

Upload a `.txt` transcript (or pick one of the two samples in the
sidebar) and click **Extract action items**. Results are shown in a
table color-coded by validation status, with CSV/JSON download.

## Run the pipeline from the command line / tests

```bash
python tests/test_pipeline.py
```

This runs extraction + validation + evaluation against both sample
transcripts and prints precision/recall/F1.

## Extraction: LLM vs. fallback

`extract_action_items()` calls Claude (`claude-sonnet-4-6`) with a prompt
that constrains the output to a JSON array matching the schema, when
`ANTHROPIC_API_KEY` is set. Without a key (or if the call fails), it
falls back to a small regex/keyword extractor (`_rule_based_extract`) so
the whole pipeline still runs offline for demos, CI, or before you've
wired up API billing. The fallback is deliberately crude — it exists to
keep the pipeline runnable, not to compete with the LLM path on
accuracy.

## Validation rules (`src/validate.py`)

- Normalizes natural-language deadlines ("next Friday", "EOD", "end of
  month") to ISO dates where possible; flags `unparseable_date`
  otherwise.
- Flags items with no stated owner as `missing_owner`.
- Flags near-duplicate tasks (`difflib` similarity ≥ 0.85) as
  `duplicate`, keeping the first occurrence as canonical.

## Evaluation (`src/evaluate.py`)

Matches predicted items to a ground-truth list by text similarity, then
reports precision, recall, F1, and owner/deadline accuracy on matched
pairs. Add more `sample_data/*.groundtruth.json` files (same shape as
the existing ones) to grow the eval set.

## Extending

- Swap the fallback regex patterns in `extract.py` for a small
  fine-tuned classifier if you outgrow prompting.
- The Streamlit app is intentionally single-file; a FastAPI wrapper
  around `src/` would be a small addition if you need an API instead of
  a UI (the plan's "simple Streamlit/API layer" note).
- Git is already initialized in this folder — push to GitHub with
  `git remote add origin <url> && git push -u origin main`.
