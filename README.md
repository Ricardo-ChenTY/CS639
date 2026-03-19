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

## Intended Workflow

- Add dataset loaders in `src/data/datasets.py`
- Implement model loading and text generation in `src/llm/loader.py`
- Fill in prompting logic in `src/prompts/templates.py`
- Finish baseline methods in `src/baselines/`
- Extend adaptive routing in `src/routing/adaptive.py`
- Extend budget-aware search in `src/search/cost_aware.py`
- Use `outputs/` for saved predictions and metrics
