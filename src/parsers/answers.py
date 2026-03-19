from __future__ import annotations

import re


def normalize_answer(text: str) -> str:
    return " ".join(text.strip().lower().split())


def extract_final_answer(text: str) -> str:
    match = re.search(r"final answer\s*[:\-]\s*(.+)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip().splitlines()[-1].strip() if text.strip() else ""


def answers_match(prediction: str, target: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(target)
