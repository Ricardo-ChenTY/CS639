from __future__ import annotations

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


# Baselines first, then full adaptive, then ablations
METHODS = [
    "direct",
    "cot",
    "self_consistency",
    "adaptive",
    "meta_control_only",
    "deliberation_only",
]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run all CS639 experiment methods.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=METHODS,
        help="Subset of methods to run (default: all).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    examples = load_eval_examples(config)
    llm = build_llm(config["model"])

    all_summaries: dict[str, dict] = {}
    for method in args.methods:
        print(f"\n=== Running {method} ===")
        summary = run_experiment(config, examples, llm, method)
        all_summaries[method] = summary
        print(summary)

    output_dir = Path(config["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "all_methods_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(all_summaries, handle, indent=2)

    print(f"\nSaved combined summary to {summary_path}")


if __name__ == "__main__":
    main()
