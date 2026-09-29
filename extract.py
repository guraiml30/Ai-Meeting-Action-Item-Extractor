

from __future__ import annotations

import json
import os
import re

from .preprocess import Turn, turns_to_prompt_text
from .schema import ActionItem, ExtractionResult

MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """You extract action items from meeting transcripts.

You will be given numbered, speaker-tagged sentences. Return ONLY a JSON array
(no prose, no markdown fences) of objects with exactly these keys:
- task: short imperative description (string)
- owner: person responsible, or null if not stated
- deadline_raw: the deadline exactly as mentioned (e.g. "by Friday"), or null
- confidence: your confidence 0.0-1.0 that this is a genuine, actionable commitment
- source_line: the integer index [N] of the sentence it came from

Only include genuine commitments or assigned tasks - skip general discussion,
questions, and past-tense statements about work already done."""


def _call_claude(prompt_text: str) -> str:
    
    import importlib

    anthropic = importlib.import_module("anthropic")

    client = anthropic.Anthropic()  
    response = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt_text}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def _parse_llm_json(raw: str, turns: list[Turn]) -> list[ActionItem]:
    cleaned = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    data = json.loads(cleaned)
    items = []
    for obj in data:
        line_idx = obj.get("source_line")
        quote = turns[line_idx].sentence if isinstance(line_idx, int) and 0 <= line_idx < len(turns) else None
        items.append(
            ActionItem(
                task=obj["task"],
                owner=obj.get("owner"),
                deadline_raw=obj.get("deadline_raw"),
                confidence=float(obj.get("confidence", 0.5)),
                source_quote=quote,
            )
        )
    return items


# --- Rule-based fallback -----------------------------------------------

COMMITMENT_PATTERNS = [
    re.compile(r"\bi(?:'ll| will)\b (?P<task>.+)", re.IGNORECASE),
    re.compile(r"\b(?P<owner>[A-Z][a-z]+) (?:will|is going to|needs to|should)\b (?P<task>.+)"),
    re.compile(r"\blet'?s\b (?P<task>.+)", re.IGNORECASE),
    re.compile(r"\baction item[:\-]?\s*(?P<task>.+)", re.IGNORECASE),
]
DEADLINE_PATTERN = re.compile(
    r"\b(by|before|due)\s+([A-Za-z]+ \d{1,2}(st|nd|rd|th)?|next \w+|tomorrow|"
    r"end of (day|week|month)|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.IGNORECASE,
)


def _rule_based_extract(turns: list[Turn]) -> list[ActionItem]:
    items: list[ActionItem] = []
    for turn in turns:
        for pattern in COMMITMENT_PATTERNS:
            match = pattern.search(turn.sentence)
            if not match:
                continue
            groups = match.groupdict()
            task = groups.get("task", turn.sentence).strip().rstrip(".")
            owner = groups.get("owner") or (turn.speaker if turn.speaker != "Unknown" else None)
            deadline_match = DEADLINE_PATTERN.search(turn.sentence)
            deadline_raw = deadline_match.group(0) if deadline_match else None
            items.append(
                ActionItem(
                    task=task,
                    owner=owner,
                    deadline_raw=deadline_raw,
                    confidence=0.55,  
                    source_quote=turn.sentence,
                )
            )
            break  
    return items


# --- Public entry point --------------------------------------------------


def extract_action_items(turns: list[Turn], transcript_id: str | None = None) -> ExtractionResult:
    """Try the LLM path; fall back to rules if no API key or the call fails."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            prompt_text = turns_to_prompt_text(turns)
            raw = _call_claude(prompt_text)
            items = _parse_llm_json(raw, turns)
            return ExtractionResult(items=items, transcript_id=transcript_id, model_used=MODEL)
        except Exception:
            pass  

    items = _rule_based_extract(turns)
    return ExtractionResult(items=items, transcript_id=transcript_id, model_used="rule-based-fallback")
