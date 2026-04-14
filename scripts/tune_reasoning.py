from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.data.datasets import load_dev_examples
from src.evaluation.runner import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune adaptive reasoning thresholds on the dev split.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument(
        "--output-config",
        default=None,
        help="Optional path to write the tuned YAML config. Defaults to <config stem>_tuned.yaml.",
    )
    return parser.parse_args()


def candidate_grid() -> dict[str, list[float]]:
    # With logprob-based A verifier, belief score is continuous in [0,1].
    # When V=1, G=1: belief = (2 + A_logprob) / 3 in [0.67, 1.0].
    # Denser grid in 0.66-0.84 range where the discriminating action is.
    return {
        "tau_0": [0.50, 0.66, 0.75, 0.84],
        "tau_1": [0.50, 0.66, 0.75, 0.84],
        "tau_expand": [0.0, 0.05, 0.1],
        "lambda_cost": [0.0, 0.1, 0.2, 0.3],
    }


def objective(summary: dict, reasoning_cfg: dict, max_new_tokens: int) -> float:
    normalized_cost = summary["avg_tokens"] / max(max_new_tokens, 1)
    return summary["accuracy"] - (reasoning_cfg["lambda_cost"] * normalized_cost)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    dev_examples = load_dev_examples(config)
    if not dev_examples:
        raise ValueError("No dev examples found under dataset/splits/dev. Run download_datasets.py first.")

    from src.llm.loader import build_llm

    llm = build_llm(config["model"])
    grid = candidate_grid()
    best_run: dict | None = None
    runs: list[dict] = []

    for tau_0 in grid["tau_0"]:
        for tau_1 in grid["tau_1"]:
            if tau_1 > tau_0:
                continue
            for tau_expand in grid["tau_expand"]:
                for lambda_cost in grid["lambda_cost"]:
                    trial = copy.deepcopy(config)
                    trial["reasoning"]["tau_0"] = tau_0
                    trial["reasoning"]["tau_1"] = tau_1
                    trial["reasoning"]["tau_expand"] = tau_expand
                    trial["reasoning"]["lambda_cost"] = lambda_cost

                    summary = run_experiment(trial, dev_examples, llm, "adaptive")
                    score = objective(summary, trial["reasoning"], trial["model"]["max_new_tokens"])
                    record = {
                        "score": score,
                        "summary": summary,
                        "reasoning": {
                            "tau_0": tau_0,
                            "tau_1": tau_1,
                            "tau_expand": tau_expand,
                            "lambda_cost": lambda_cost,
                        },
                    }
                    runs.append(record)

                    if best_run is None or (
                        score,
                        summary["accuracy"],
                        -summary["avg_tokens"],
                    ) > (
                        best_run["score"],
                        best_run["summary"]["accuracy"],
                        -best_run["summary"]["avg_tokens"],
                    ):
                        best_run = record

    assert best_run is not None

    tuned_config = copy.deepcopy(config)
    tuned_config["reasoning"].update(best_run["reasoning"])

    config_path = Path(args.config)
    output_config = Path(args.output_config) if args.output_config else config_path.with_name(
        f"{config_path.stem}_tuned.yaml"
    )
    with output_config.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(tuned_config, handle, sort_keys=False)

    output_dir = Path(config["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    tuning_path = output_dir / "adaptive_dev_tuning.json"
    with tuning_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "best": best_run,
                "num_trials": len(runs),
                "trials": runs,
            },
            handle,
            indent=2,
        )

    print("Best adaptive dev setting:")
    print(json.dumps(best_run, indent=2))
    print(f"Tuned config written to {output_config}")
    print(f"Full tuning log written to {tuning_path}")


if __name__ == "__main__":
    main()
