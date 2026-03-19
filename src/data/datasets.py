from __future__ import annotations

import json
from pathlib import Path

from src.types import Example


def dataset_root(config: dict) -> Path:
    return Path(config["datasets"]["root_dir"])


def _read_examples(path: Path) -> list[Example]:
    if not path.exists():
        return []

    examples = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            examples.append(
                Example(
                    example_id=row["example_id"],
                    task_name=row["task_name"],
                    question=row["question"],
                    answer=row["answer"],
                    metadata=row.get("metadata", {}),
                )
            )
    return examples


def _load_split(config: dict, split_name: str) -> list[Example]:
    root = dataset_root(config) / "splits" / split_name
    if not root.exists():
        return []

    examples: list[Example] = []
    for path in sorted(root.glob("*.jsonl")):
        examples.extend(_read_examples(path))
    return examples


def load_dev_examples(config: dict) -> list[Example]:
    return _load_split(config, "dev")


def load_eval_examples(config: dict) -> list[Example]:
    return _load_split(config, "eval")
