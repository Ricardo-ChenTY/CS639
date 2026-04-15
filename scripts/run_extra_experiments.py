from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.data.datasets import load_eval_examples
from src.evaluation.runner import run_experiment
from src.llm.loader import build_llm


EXTRA_RUNS = [
    # SC-5: extend self-consistency scaling curve
    {
        "tag": "sc5",
        "method": "self_consistency",
        "overrides": {"reasoning": {"self_consistency_samples": 5}},
    },
    # Deliberation-only lambda sweep: get frontier points
    {
        "tag": "deliberation_lambda0.1",
        "method": "deliberation_only",
        "overrides": {"reasoning": {"lambda_cost": 0.1}},
    },
    {
        "tag": "deliberation_lambda0.2",
        "method": "deliberation_only",
        "overrides": {"reasoning": {"lambda_cost": 0.2}},
    },
    {
        "tag": "deliberation_lambda0.3",
        "method": "deliberation_only",
        "overrides": {"reasoning": {"lambda_cost": 0.3}},
    },
    # Task-aware static routing (oracle-guided upper bound)
    {
        "tag": "task_aware",
        "method": "task_aware",
        "overrides": {},
    },
    # Task-aware with optimal lambda (lambda=0.1 from sweep)
    {
        "tag": "task_aware_lambda0.1",
        "method": "task_aware",
        "overrides": {"reasoning": {"lambda_cost": 0.1}},
    },
    # Adaptive routing with sampling-consistency verifier
    {
        "tag": "adaptive_sc_verifier",
        "method": "adaptive_sc_verifier",
        "overrides": {},
    },
]


def apply_overrides(config: dict, overrides: dict) -> dict:
    cfg = copy.deepcopy(config)
    for section, values in overrides.items():
        cfg[section].update(values)
    return cfg


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run extra experiments for paper.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument(
        "--tags",
        nargs="+",
        default=None,
        help="Subset of run tags to execute (default: all).",
    )
    args = parser.parse_args()

    base_config = load_config(args.config)
    examples = load_eval_examples(base_config)
    llm = build_llm(base_config["model"])

    output_dir = Path(base_config["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = EXTRA_RUNS if args.tags is None else [r for r in EXTRA_RUNS if r["tag"] in args.tags]

    all_summaries: dict[str, dict] = {}
    for run in runs:
        tag, method = run["tag"], run["method"]
        print(f"\n=== Running {tag} ({method}) ===")
        cfg = apply_overrides(base_config, run["overrides"])
        # Save to tag-specific files by temporarily overriding output_dir
        cfg["experiment"]["name"] = tag
        # Patch runner to use tag as method name for file naming
        cfg["_extra_tag"] = tag
        summary = _run_tagged(cfg, examples, llm, method, tag, output_dir)
        all_summaries[tag] = summary
        print(summary)

    summary_path = output_dir / "extra_experiments_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nSaved to {summary_path}")


def _run_tagged(config, examples, llm, method, tag, output_dir):
    """Run experiment and save results with tag-based filenames."""
    from tqdm import tqdm
    from src.evaluation.metrics import build_record, summarize
    from src.evaluation.runner import METHODS

    runner = METHODS[method]
    records = []
    for example in tqdm(examples, desc=tag, leave=False):
        result = runner(example, llm, config)
        records.append(
            build_record(
                example=example,
                method=method,
                prediction=result.final_answer,
                total_tokens=result.total_tokens,
                latency_sec=result.latency_sec,
                metadata=result.metadata,
            )
        )

    summary = summarize(records)

    records_path = output_dir / f"{tag}_records.json"
    summary_path = output_dir / f"{tag}_summary.json"

    with records_path.open("w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "example_id": r.example.example_id,
                    "task_name": r.example.task_name,
                    "prediction": r.prediction,
                    "gold": r.example.answer,
                    "is_correct": r.is_correct,
                    "total_tokens": r.total_tokens,
                    "latency_sec": r.latency_sec,
                    "route": r.metadata.get("route"),
                }
                for r in records
            ],
            f,
            indent=2,
        )
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    main()
