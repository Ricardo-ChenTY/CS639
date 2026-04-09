from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a Hugging Face model snapshot for this project.")
    parser.add_argument(
        "--repo-id",
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="Hugging Face model repo to download.",
    )
    parser.add_argument(
        "--local-dir",
        default="models/Qwen3-4B-Instruct-2507",
        help="Local directory where the model snapshot will be stored.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional Hugging Face token. If omitted, uses the local HF login state or anonymous access.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from huggingface_hub import snapshot_download

    local_dir = Path(args.local_dir)
    local_dir.parent.mkdir(parents=True, exist_ok=True)

    path = snapshot_download(
        repo_id=args.repo_id,
        local_dir=str(local_dir),
        token=args.token,
    )
    print(f"Downloaded model to {path}")


if __name__ == "__main__":
    main()
