from __future__ import annotations

from pathlib import Path

from src.types import Example


def dataset_root(config: dict) -> Path:
    return Path(config["datasets"]["root_dir"])


def load_dev_examples(config: dict) -> list[Example]:
    """Replace with your frozen development split loader."""
    _ = dataset_root(config)
    return []


def load_eval_examples(config: dict) -> list[Example]:
    """Replace with your frozen evaluation split loader."""
    _ = dataset_root(config)
    return []
