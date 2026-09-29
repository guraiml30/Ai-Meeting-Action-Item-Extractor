

from __future__ import annotations

import re
from dataclasses import dataclass

SPEAKER_LINE_RE = re.compile(r"^\s*([A-Za-z][\w .'-]{0,40}):\s*(.+)$")

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")

FILLER_WORDS = {"um", "uh", "erm", "you know", "like,"}


@dataclass
class Turn:
    speaker: str
    sentence: str
    line_no: int


def clean_line(text: str) -> str:
    """Strip timestamps, filler tokens, and extra whitespace."""
   
    text = re.sub(r"^\s*[\[(]\d{1,2}:\d{2}(:\d{2})?[\])]\s*", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    for filler in FILLER_WORDS:
        text = re.sub(rf"\b{re.escape(filler)}\b,?", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def segment_transcript(raw_text: str) -> list[Turn]:
    """Parse a raw transcript into (speaker, sentence) turns."""
    turns: list[Turn] = []
    current_speaker = "Unknown"

    for line_no, raw_line in enumerate(raw_text.splitlines(), start=1):
        if not raw_line.strip():
            continue

        match = SPEAKER_LINE_RE.match(raw_line)
        if match:
            current_speaker, body = match.group(1).strip(), match.group(2)
        else:
            body = raw_line  

        body = clean_line(body)
        if not body:
            continue

        for sentence in SENTENCE_SPLIT_RE.split(body):
            sentence = sentence.strip()
            if sentence:
                turns.append(Turn(speaker=current_speaker, sentence=sentence, line_no=line_no))

    return turns


def turns_to_prompt_text(turns: list[Turn]) -> str:
    """Re-render segmented turns as a compact speaker-tagged block for the LLM prompt."""
    return "\n".join(f"[{i}] {t.speaker}: {t.sentence}" for i, t in enumerate(turns))
