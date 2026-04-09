# CS639 Project

This repository implements the current proposal in [full_proposal_revised.tex](/Users/ricardochen/CS639/full_proposal_revised.tex), including direct answering, zero-shot CoT, self-consistency, and adaptive reasoning with cost-aware search.

## Directory Layout

- `models/`: reserved for local model weights, quantized checkpoints, or adapter files
- `dataset/`: reserved for raw data, frozen splits, and processed subsets
- `configs/`: experiment configuration
- `scripts/`: entry scripts for running experiments
- `src/`: core project code
- `outputs/`: predictions, metrics, and logs

## Quick Start

1. Create a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Put model assets under `models/`.
4. Put dataset files or frozen splits under `dataset/`.
5. Run the scaffold:

```bash
python scripts/run_experiment.py --config configs/default.yaml
```

To run all currently implemented methods in one shot:

```bash
python scripts/run_all_experiments.py --config configs/default.yaml
```

To tune the adaptive controller once on the frozen dev split and write a frozen config:

```bash
python scripts/tune_reasoning.py --config configs/default.yaml
python scripts/run_all_experiments.py --config configs/default_tuned.yaml
```

For a server-side pilot run without the notebook:

```bash
python scripts/download_model.py --local-dir models/Qwen3-4B-Instruct-2507
python scripts/download_datasets.py --root-dir dataset --hf-cache-dir dataset/hf_cache --overwrite
python scripts/make_pilot_config.py --output-config configs/pilot_server.yaml
python scripts/run_all_experiments.py --config configs/pilot_server.yaml
```

To download and freeze the proposal subsets with a fixed seed:

```bash
python scripts/download_datasets.py \
  --root-dir /content/drive/MyDrive/CS639/dataset \
  --hf-cache-dir /content/drive/MyDrive/CS639/dataset/hf_cache \
  --overwrite
```

The script downloads from Hugging Face `datasets`:
- `openai/gsm8k`
- `lukaemon/bbh`
- `cais/mmlu`

## Google Drive Layout

If you run in Google Colab, the default config already assumes this layout after mounting Drive:

```text
/content/drive/MyDrive/CS639/
  models/
  dataset/
  outputs/
```

Mount Drive in Colab before running experiments:

```python
from google.colab import drive
drive.mount('/content/drive')
```

## Current Workflow

- Use `scripts/download_datasets.py` to create frozen subsets under `dataset/splits/`
- Adjust `configs/default.yaml` for your model path, dataset root, and budget thresholds
- Tune `tau_0`, `tau_1`, `tau_expand`, and `lambda_cost` once on the dev split with `scripts/tune_reasoning.py`
- Run `scripts/run_all_experiments.py` to compare direct, CoT, self-consistency, and adaptive search
- Inspect `outputs/` for per-example records and summary metrics
