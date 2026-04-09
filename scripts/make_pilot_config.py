from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a small pilot config for server-side experiment smoke runs.")
    parser.add_argument("--base-config", default="configs/default.yaml", help="Base YAML config to copy.")
    parser.add_argument(
        "--output-config",
        default="configs/pilot_server.yaml",
        help="Output YAML path for the generated pilot config.",
    )
    parser.add_argument(
        "--model-path",
        default=str(PROJECT_ROOT / "models" / "Qwen3-4B-Instruct-2507"),
        help="Local model directory to write into the config.",
    )
    parser.add_argument(
        "--dataset-root",
        default=str(PROJECT_ROOT / "dataset"),
        help="Local dataset root to write into the config.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "outputs" / "pilot"),
        help="Output directory to write into the config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.base_config)

    config["model"]["model_name_or_path"] = args.model_path
    config["datasets"]["root_dir"] = args.dataset_root
    config["experiment"]["output_dir"] = args.output_dir

    config["datasets"]["dev_split"] = {
        "gsm8k": 3,
        "bbh_date_understanding": 2,
        "bbh_logical_deduction_three_objects": 2,
        "mmlu": 2,
    }
    config["datasets"]["eval_split"] = {
        "gsm8k": 10,
        "bbh_date_understanding": 5,
        "bbh_logical_deduction_three_objects": 5,
        "mmlu": 10,
    }
    config["reasoning"]["tau_0"] = 0.67
    config["reasoning"]["tau_1"] = 0.5
    config["reasoning"]["tau_expand"] = 0.05
    config["reasoning"]["lambda_cost"] = 0.1

    output_path = Path(args.output_config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    print(f"Wrote pilot config to {output_path}")


if __name__ == "__main__":
    main()
