# CS639 Project Scaffold

This repository is a minimal experiment scaffold for the proposal in [full_proposal_revised.tex](/Users/ricardochen/CS639/full_proposal_revised.tex).

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

To download and freeze the proposal subsets with a fixed seed:

```bash
python scripts/download_datasets.py --root-dir /content/drive/MyDrive/CS639/dataset --overwrite
```

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

## Intended Workflow

- Use `scripts/download_datasets.py` to create frozen subsets under `dataset/splits/`
- Implement model loading and text generation in `src/llm/loader.py`
- Fill in prompting logic in `src/prompts/templates.py`
- Finish baseline methods in `src/baselines/`
- Extend adaptive routing in `src/routing/adaptive.py`
- Extend budget-aware search in `src/search/cost_aware.py`
- Use `outputs/` for saved predictions and metrics
