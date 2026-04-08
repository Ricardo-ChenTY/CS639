from __future__ import annotations

"""Download proposal datasets from Hugging Face and freeze fixed JSONL splits."""

import argparse
import json
import random
from pathlib import Path
from typing import Iterable

from datasets import load_dataset


MMLU_SUBJECTS = [
    "high_school_mathematics",
    "formal_logic",
    "computer_security",
    "college_biology",
]

HF_SOURCES = {
    "gsm8k": ("openai/gsm8k", "main"),
    "bbh": ("lukaemon/bbh", None),
    "mmlu": ("cais/mmlu", None),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and freeze the CS639 benchmark subsets."
    )
    parser.add_argument(
        "--root-dir",
        default="dataset",
        help="Dataset root directory. In Colab this can be /content/drive/MyDrive/CS639/dataset.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for frozen splits.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing frozen split files.",
    )
    parser.add_argument(
        "--hf-cache-dir",
        default=None,
        help="Optional Hugging Face datasets cache directory.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sample_records(records: list[dict], count: int, seed: int) -> list[dict]:
    if count > len(records):
        raise ValueError(f"Requested {count} records from a pool of {len(records)}.")
    rng = random.Random(seed)
    indices = list(range(len(records)))
    rng.shuffle(indices)
    return [records[index] for index in indices[:count]]


def save_jsonl(path: Path, rows: Iterable[dict], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists. Use --overwrite to replace it.")

    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_gsm8k(split_name: str, row: dict, index: int) -> dict:
    answer_text = row["answer"]
    final_answer = answer_text.split("####")[-1].strip() if "####" in answer_text else answer_text.strip()
    return {
        "example_id": f"gsm8k_{split_name}_{index:04d}",
        "task_name": "gsm8k",
        "question": row["question"].strip(),
        "answer": final_answer,
        "metadata": {
            "source_dataset": "openai/gsm8k",
            "source_config": "main",
            "source_split": split_name,
            "raw_answer": answer_text,
        },
    }


def normalize_bbh(task_name: str, row: dict, index: int) -> dict:
    return {
        "example_id": f"{task_name}_{index:04d}",
        "task_name": task_name,
        "question": row["input"].strip(),
        "answer": row["target"].strip(),
        "metadata": {
            "source_dataset": "lukaemon/bbh",
            "source_config": task_name,
            "source_split": "test",
        },
    }


def normalize_mmlu(subject: str, split_name: str, row: dict, index: int) -> dict:
    answer_index = int(row["answer"])
    answer_label = ["A", "B", "C", "D"][answer_index]
    return {
        "example_id": f"mmlu_{subject}_{split_name}_{index:04d}",
        "task_name": f"mmlu_{subject}",
        "question": row["question"].strip(),
        "answer": answer_label,
        "metadata": {
            "source_dataset": "cais/mmlu",
            "source_config": subject,
            "source_split": split_name,
            "choices": row["choices"],
            "subject": subject,
        },
    }


def distribute_total(total: int, buckets: int) -> list[int]:
    base = total // buckets
    remainder = total % buckets
    return [base + (1 if idx < remainder else 0) for idx in range(buckets)]


def freeze_gsm8k(root_dir: Path, seed: int, overwrite: bool, hf_cache_dir: str | None) -> None:
    dataset = load_dataset("openai/gsm8k", "main", cache_dir=hf_cache_dir)
    dev_rows = [normalize_gsm8k("train", row, idx) for idx, row in enumerate(dataset["train"])]
    eval_rows = [normalize_gsm8k("test", row, idx) for idx, row in enumerate(dataset["test"])]

    save_jsonl(root_dir / "splits" / "dev" / "gsm8k.jsonl", sample_records(dev_rows, 10, seed), overwrite)
    save_jsonl(root_dir / "splits" / "eval" / "gsm8k.jsonl", sample_records(eval_rows, 100, seed), overwrite)


def freeze_bbh_task(
    root_dir: Path,
    task_name: str,
    dev_count: int,
    eval_count: int,
    seed: int,
    overwrite: bool,
    hf_cache_dir: str | None,
) -> None:
    dataset = load_dataset("lukaemon/bbh", task_name, cache_dir=hf_cache_dir)["test"]
    rows = [normalize_bbh(task_name, row, idx) for idx, row in enumerate(dataset)]

    if dev_count + eval_count > len(rows):
        raise ValueError(f"{task_name} needs {dev_count + eval_count} rows but only has {len(rows)}.")

    rng = random.Random(seed)
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    dev_sample = [rows[idx] for idx in indices[:dev_count]]
    eval_sample = [rows[idx] for idx in indices[dev_count : dev_count + eval_count]]

    save_jsonl(root_dir / "splits" / "dev" / f"{task_name}.jsonl", dev_sample, overwrite)
    save_jsonl(root_dir / "splits" / "eval" / f"{task_name}.jsonl", eval_sample, overwrite)


def freeze_mmlu(root_dir: Path, seed: int, overwrite: bool, hf_cache_dir: str | None) -> None:
    dev_counts = distribute_total(5, len(MMLU_SUBJECTS))
    eval_counts = distribute_total(100, len(MMLU_SUBJECTS))

    dev_rows: list[dict] = []
    eval_rows: list[dict] = []

    for subject_idx, subject in enumerate(MMLU_SUBJECTS):
        dataset = load_dataset("cais/mmlu", subject, cache_dir=hf_cache_dir)
        subject_dev = [
            normalize_mmlu(subject, "dev", row, idx)
            for idx, row in enumerate(dataset["dev"])
        ]
        subject_eval = [
            normalize_mmlu(subject, "test", row, idx)
            for idx, row in enumerate(dataset["test"])
        ]

        dev_rows.extend(sample_records(subject_dev, dev_counts[subject_idx], seed + subject_idx))
        eval_rows.extend(sample_records(subject_eval, eval_counts[subject_idx], seed + 100 + subject_idx))

    save_jsonl(root_dir / "splits" / "dev" / "mmlu.jsonl", dev_rows, overwrite)
    save_jsonl(root_dir / "splits" / "eval" / "mmlu.jsonl", eval_rows, overwrite)


def write_manifest(root_dir: Path, seed: int, overwrite: bool) -> None:
    manifest = {
        "seed": seed,
        "datasets": {
            "gsm8k": {"source": "openai/gsm8k", "dev": 10, "eval": 100},
            "date_understanding": {"source": "lukaemon/bbh", "dev": 10, "eval": 50},
            "logical_deduction_three_objects": {"source": "lukaemon/bbh", "dev": 5, "eval": 50},
            "mmlu": {
                "source": "cais/mmlu",
                "subjects": MMLU_SUBJECTS,
                "dev_total": 5,
                "eval_total": 100,
            },
        },
    }
    save_jsonl(root_dir / "manifest.jsonl", [manifest], overwrite)


def main() -> None:
    args = parse_args()
    root_dir = Path(args.root_dir)
    ensure_dir(root_dir)

    print("Downloading datasets from Hugging Face...")
    print(f"  GSM8K: {HF_SOURCES['gsm8k'][0]} / {HF_SOURCES['gsm8k'][1]}")
    print(f"  BBH:   {HF_SOURCES['bbh'][0]}")
    print(f"  MMLU:  {HF_SOURCES['mmlu'][0]}")
    if args.hf_cache_dir:
        print(f"  HF cache: {args.hf_cache_dir}")

    print("Downloading and freezing GSM8K from Hugging Face...")
    freeze_gsm8k(root_dir, args.seed, args.overwrite, args.hf_cache_dir)

    print("Downloading and freezing BBH date_understanding from Hugging Face...")
    freeze_bbh_task(
        root_dir,
        "date_understanding",
        10,
        50,
        args.seed,
        args.overwrite,
        args.hf_cache_dir,
    )

    print("Downloading and freezing BBH logical_deduction_three_objects from Hugging Face...")
    freeze_bbh_task(
        root_dir,
        "logical_deduction_three_objects",
        5,
        50,
        args.seed + 1,
        args.overwrite,
        args.hf_cache_dir,
    )

    print("Downloading and freezing MMLU from Hugging Face...")
    freeze_mmlu(root_dir, args.seed, args.overwrite, args.hf_cache_dir)

    print("Writing manifest...")
    write_manifest(root_dir, args.seed, args.overwrite)
    print(f"Done. Frozen subsets saved under {root_dir / 'splits'}.")


if __name__ == "__main__":
    main()
