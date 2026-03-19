from __future__ import annotations

import argparse

from src.config import load_config
from src.data.datasets import load_eval_examples
from src.evaluation.runner import run_experiment
from src.llm.loader import build_llm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CS639 reasoning experiments.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument(
        "--method",
        default="adaptive",
        choices=["direct", "cot", "self_consistency", "adaptive"],
        help="Inference method to run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    llm = build_llm(config["model"])
    examples = load_eval_examples(config)
    summary = run_experiment(config, examples, llm, args.method)
    print(summary)


if __name__ == "__main__":
    main()
